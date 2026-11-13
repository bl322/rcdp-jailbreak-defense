"""Trajectory-policy selection for the multi-turn study.

Two procedures live here.

``search_trajectory_policy_r1`` (revision R1, the paper's description, Sec. 4):
    selects vartheta = (d_tau_c, d_tau_h, d_tau_q, eta, xi, lambda, k_down, k_up)
    from a finite candidate set on *calibration (selection-side) trajectories*,
    with the scorer, risk head and Platt layer fixed. The objective is the
    paper's operating-point criterion at session level,
        J = ABR_sess - alpha * FPR_sess   (alpha = 1.0),
    where a harmful session counts as blocked if any user turn is blocked and
    a benign session counts as preserved only if every user turn is allowed.
    Ties: max J, then max ABR, then min mean prompt tokens (quarantine
    terminal), then min L1 norm of the threshold offsets, then enumeration
    order. The caller supplies the selection side (seed-cluster split, see
    rcdp_guard.multiturn.split_by_cluster); nothing here sees evaluation data.
    The candidate is evaluated with use_trajectory_features=False, i.e. the
    routing risk is min(1, r_t + eta*rho_{t-1} + xi*max(0, r_t - r_{t-1}))
    and rho_t = lambda*rho_{t-1} + max(0, r_t - tau_c), the paper's equations,
    on the base variant's downgrade rule (4-state+memory: one level at a time
    after k_down safe turns).
    What the paper fixes and what it does not: Sec. 4 (4_method.tex) names the
    eight components of vartheta, "a finite candidate set", "calibration
    trajectories" and a fixed scorer/Platt layer, and defers "full details" to
    the supplementary material, which contains none (no objective, no candidate
    values, no split). The objective above is therefore the paper's only stated
    operating-point criterion (J with alpha = 1.0) applied at session level; the
    candidate values (POLICY_SPACE_R1) are the submitted grid's values mapped to
    vartheta, plus lambda = 0.80 (paper default) and eta = 0 / xi = 0 (term off).
    k_down and k_up only move non-blocking states (downgrades; N->C and C->H
    escalation), so they change tokens but never session ABR/BAR; they are
    decided by the token tie-break.

``search_trajectory_policy`` / ``replay_summary`` / ``split_examples_by_id``
(legacy, unchanged): the submitted procedure. It differs from the paper in
(i) the split: ord-sum of the *session* id mod 100 < 67 (sessions of one seed
fall on both sides); (ii) the objective: ABR + 0.2*BAR - 0.002*tokens on
turn-level rates; (iii) the candidate set: 10 knobs including
use_trajectory_features=True (dynamic threshold lowering, 0.45*rho gain and a
4-state-only escalation rule that the paper does not describe), and a
trajectory_decay that is ignored whenever use_risk_memory=True; (iv) the
replay never decays the history counters (not faithful when history_decay=1).
Kept only so the submitted numbers stay reproducible.

``replay_session`` is a faithful re-implementation of the routing part of
RCDPController.evaluate_example given per-turn calibrated probabilities (which
do not depend on the controller state, because the calibration features that
depend on it are always zero in the single-turn fit). Runners must check it
against the controller (scripts/run_multiturn_cases.py does, for every
variant and every session).
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .controller import ControllerConfig, ControllerState, _rank, select_controller_state

_Q = ControllerState.QUARANTINE
_H = ControllerState.HARDENED
_C = ControllerState.CAUTIOUS

# ---------------------------------------------------------------------------
# Faithful replay (revision R1)
# ---------------------------------------------------------------------------


def block_key(num_states: int) -> str:
    """Threshold that blocks in select_controller_state (2-state: tau_h, else tau_q)."""
    return "tau_h" if num_states == 2 else "tau_q"


def check_replay_supported(config: ControllerConfig) -> None:
    if config.use_terminal_semantic_review:
        raise ValueError("replay does not support use_terminal_semantic_review (needs a second scorer)")
    if config.use_learned_transition_policy:
        raise ValueError("replay does not support use_learned_transition_policy")


def replay_session(
    probs: Sequence[float],
    config: ControllerConfig,
    thresholds: Dict[str, float],
    *,
    stop_at_block: bool = False,
) -> List[tuple]:
    """Route one dialogue from its per-user-turn calibrated probabilities.

    Returns one tuple (state, decision, effective_risk, risk_memory_before_turn)
    per routed user turn; with ``stop_at_block`` routing stops after the first
    blocked turn (quarantine is terminal for the session).
    """
    out: List[tuple] = []
    tau_c = thresholds["tau_c"]
    prior_state: Optional[ControllerState] = None
    prior_high = 0
    prior_caut = 0
    risk_memory = 0.0
    safe_streak = 0
    risky_streak = 0
    latched = False
    prev_p = 0.0
    has_prev = False
    downgrade_run = 0
    total = len(probs) or 1
    traj = config.use_trajectory_features
    decay_counters = config.allow_state_downgrade and config.history_decay > 0
    latch_rank = _rank(config.hazard_latch_min_state)
    for i, p in enumerate(probs):
        if decay_counters:
            prior_high = max(0, prior_high - config.history_decay)
            prior_caut = max(0, prior_caut - config.history_decay)
        uncertainty = 1.0 - abs(2 * p - 1.0)
        ref = prev_p if (has_prev or config.legacy_first_turn_trend) else p
        trend = p - ref
        eff = p
        pressure = max(0.0, p - tau_c)
        safe_run = downgrade_run + 1 if p < tau_c else 0
        if config.use_risk_memory:
            eff = min(1.0, p + config.risk_memory_gain * risk_memory)
        if config.risk_trend_gain:
            eff = min(1.0, eff + config.risk_trend_gain * max(0.0, trend))
        if traj:
            eff = min(
                1.0,
                eff
                + config.trajectory_gain * risk_memory
                + config.trajectory_trend_gain * max(0.0, trend)
                - 0.05 * max(0, safe_streak - 1),
            )
        state = select_controller_state(
            calibrated=eff,
            uncertainty=uncertainty,
            prior_state=prior_state,
            history_flag=prior_high > 0,
            safe_streak=safe_streak,
            risky_streak=risky_streak,
            config=config,
            dialogue_progress=(i / max(total - 1, 1)) if traj else 0.0,
            risk_memory=risk_memory,
            prev_risk_level=prev_p if (has_prev or config.legacy_first_turn_trend) else eff,
            transition_score=0.5,
            thresholds=thresholds,
            safe_run=safe_run,
        )
        if config.use_hazard_latch and latched:
            state = _Q
        downgrade_run = 0 if (prior_state is not None and _rank(state) < _rank(prior_state)) else safe_run
        decision = "block" if state == _Q else "allow"
        out.append((state, decision, eff, risk_memory))
        if decision == "block" and stop_at_block:
            break
        if state == _H or state == _Q:
            prior_high += 1
        if state == _C:
            prior_caut += 1
        if _rank(state) >= latch_rank:
            latched = True
        if config.use_risk_memory or traj:
            decay = config.risk_memory_decay if config.use_risk_memory else config.trajectory_decay
            risk_memory = decay * risk_memory + pressure
        else:
            risk_memory = 0.0
        if p < tau_c:
            safe_streak += 1
            risky_streak = 0
        else:
            risky_streak += 1
            safe_streak = 0
        prior_state = state
        prev_p = p
        has_prev = True
    return out


def session_outcome(routed: Sequence[tuple], config: ControllerConfig) -> Dict[str, Any]:
    """Session-level outcome under the paper's rule (any blocked turn blocks the session).

    ``tokens_terminal`` counts the state-prompt tokens of the turns up to and
    including the first block (quarantine is terminal).
    """
    tokens = config.state_prompt_tokens
    first = None
    for i, r in enumerate(routed):
        if r[1] == "block":
            first = i
            break
    upto = routed if first is None else routed[: first + 1]
    return {
        "blocked": first is not None,
        "first_block_turn": first,
        "turns_terminal": len(upto),
        "tokens_terminal": sum(tokens[r[0]] for r in upto),
        "reached_hq": any(r[0] == _H or r[0] == _Q for r in upto),
    }


def replay_outcomes(
    sessions: Sequence[Dict[str, Any]],
    probs: Sequence[Sequence[float]],
    config: ControllerConfig,
    thresholds_fn: Callable[[str], Dict[str, float]],
    *,
    indices: Optional[Iterable[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Outcomes of ``sessions[i]`` (benchmark taken from the session) for i in ``indices``."""
    check_replay_supported(config)
    idx = range(len(sessions)) if indices is None else indices
    out = {}
    for i in idx:
        th = thresholds_fn(sessions[i]["benchmark"])
        out[i] = session_outcome(replay_session(probs[i], config, th, stop_at_block=True), config)
    return out


def summarize_outcomes(sessions: Sequence[Dict[str, Any]], outcomes: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    nh = bh = nb = ab = 0
    tok = turns = 0
    for i, o in outcomes.items():
        if sessions[i]["label"] == "harmful":
            nh += 1
            bh += int(o["blocked"])
        else:
            nb += 1
            ab += int(not o["blocked"])
        tok += o["tokens_terminal"]
        turns += o["turns_terminal"]
    abr = bh / nh if nh else None
    bar = ab / nb if nb else None
    return {
        "n_harmful": nh, "blocked_harmful": bh, "abr": abr,
        "n_benign": nb, "allowed_benign": ab, "bar": bar,
        "fpr": (1.0 - bar) if bar is not None else None,
        "mean_prompt_tokens_terminal": tok / turns if turns else None,
    }


# ---------------------------------------------------------------------------
# Threshold families
# ---------------------------------------------------------------------------


def shift_block_threshold(th: Dict[str, float], delta: float, bkey: str) -> Dict[str, float]:
    """Move the blocking threshold by ``delta`` (clipped to [0, 1]) and rescale the
    non-blocking ones by the same factor (the ratio rule of the calibration
    protocol). tau_u is unchanged. delta == 0 returns the thresholds unchanged."""
    if delta == 0:
        return dict(th)
    base = th[bkey]
    tau = min(1.0, max(0.0, round(base + delta, 6)))
    if base > 0:
        scale = tau / base
        out = {k: (min(1.0, v * scale) if k in ("tau_c", "tau_h", "tau_q") else v) for k, v in th.items()}
    else:  # degenerate base: fall back to the default ratios
        from .calibration_protocol import thresholds_for_block

        out = thresholds_for_block(tau, bkey)
        out["tau_u"] = th.get("tau_u", out["tau_u"])
    out[bkey] = tau
    return out


def offset_thresholds(th: Dict[str, float], d_c: float, d_h: float, d_q: float) -> Dict[str, float]:
    clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
    out = dict(th)
    out["tau_c"] = clip(th["tau_c"] + d_c)
    out["tau_h"] = clip(th["tau_h"] + d_h)
    out["tau_q"] = clip(th["tau_q"] + d_q)
    return out


def thresholds_function(config: ControllerConfig) -> Callable[[str], Dict[str, float]]:
    """Frozen copy of config.thresholds_for (later edits of config do not leak in)."""
    frozen = replace(
        config,
        thresholds=dict(config.thresholds),
        benchmark_thresholds=(
            {k: dict(v) for k, v in config.benchmark_thresholds.items()}
            if config.benchmark_thresholds else None
        ),
    )
    return frozen.thresholds_for


# ---------------------------------------------------------------------------
# Paper-aligned trajectory-policy search (revision R1)
# ---------------------------------------------------------------------------

POLICY_SPACE_R1 = {
    # threshold offsets relative to the single-turn operating point (as in the submitted grid)
    "d_tau_c": [-0.08, -0.05, -0.03, 0.0],
    "d_tau_h": [-0.08, -0.05, -0.03, 0.0],
    "d_tau_q": [-0.10, -0.07, -0.04, 0.0],
    "eta": [0.0, 0.25, 0.50, 0.90],      # memory gain (0 = memory off); paper default 0.90
    "xi": [0.0, 0.10, 0.18, 0.26],       # trend gain
    "lam": [0.62, 0.72, 0.80, 0.82],     # memory decay; paper default 0.80
    "k_down": [1, 2],                    # safe streak before a downgrade; paper default 2
    "k_up": [1, 2],                      # risky streak before an escalation
}
POLICY_SPACE_SMALL = {
    "d_tau_c": [-0.05, 0.0],
    "d_tau_h": [0.0],
    "d_tau_q": [-0.07, 0.0],
    "eta": [0.0, 0.90],
    "xi": [0.0, 0.18],
    "lam": [0.80],
    "k_down": [2],
    "k_up": [2],
}
POLICY_KEYS = ("d_tau_c", "d_tau_h", "d_tau_q", "eta", "xi", "lam", "k_down", "k_up")
R1_OBJECTIVE = "J = ABR_sess - alpha * FPR_sess on selection-side sessions (any-turn-block rule)"
R1_TIE_BREAK = ("max J; then max ABR; then min mean prompt tokens (quarantine terminal); "
                "then min |d_tau_c|+|d_tau_h|+|d_tau_q|; then enumeration order")


def policy_candidates(space: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Enumerate the grid; with eta == 0 the decay is irrelevant, so only its first value is kept."""
    missing = [k for k in POLICY_KEYS if k not in space]
    if missing:
        raise ValueError(f"policy space lacks {missing}")
    out = []
    for values in product(*(space[k] for k in POLICY_KEYS)):
        cand = dict(zip(POLICY_KEYS, values))
        if cand["eta"] == 0 and cand["lam"] != space["lam"][0]:
            continue
        out.append(cand)
    return out


def candidate_config(base: ControllerConfig, cand: Dict[str, Any]) -> ControllerConfig:
    return replace(
        base,
        num_states=4,
        use_trajectory_features=False,
        use_learned_transition_policy=False,
        use_risk_memory=cand["eta"] > 0,
        risk_memory_gain=float(cand["eta"]),
        risk_trend_gain=float(cand["xi"]),
        risk_memory_decay=float(cand["lam"]),
        safe_streak_to_downgrade=int(cand["k_down"]),
        risky_streak_to_escalate=int(cand["k_up"]),
    )


def candidate_thresholds_fn(base_fn: Callable[[str], Dict[str, float]], cand: Dict[str, Any]):
    cache: Dict[str, Dict[str, float]] = {}

    def fn(bench: str) -> Dict[str, float]:
        if bench not in cache:
            cache[bench] = offset_thresholds(base_fn(bench), cand["d_tau_c"], cand["d_tau_h"], cand["d_tau_q"])
        return cache[bench]

    return fn


def search_trajectory_policy_r1(
    sessions: Sequence[Dict[str, Any]],
    probs: Sequence[Sequence[float]],
    selection_indices: Sequence[int],
    base_config: ControllerConfig,
    base_thresholds_fn: Callable[[str], Dict[str, float]],
    *,
    space: Optional[Dict[str, Sequence[Any]]] = None,
    alpha: float = 1.0,
    top_k: int = 20,
) -> Dict[str, Any]:
    space = space or POLICY_SPACE_R1
    cands = policy_candidates(space)
    if not cands:
        raise ValueError("empty policy space")
    sel = list(selection_indices)
    if not any(sessions[i]["label"] == "harmful" for i in sel) or not any(sessions[i]["label"] == "benign" for i in sel):
        raise ValueError("selection side needs harmful and benign sessions")
    scored = []
    for order, cand in enumerate(cands):
        cfg = candidate_config(base_config, cand)
        summ = summarize_outcomes(sessions, replay_outcomes(sessions, probs, cfg, candidate_thresholds_fn(base_thresholds_fn, cand), indices=sel))
        j = summ["abr"] - alpha * summ["fpr"]
        l1 = abs(cand["d_tau_c"]) + abs(cand["d_tau_h"]) + abs(cand["d_tau_q"])
        key = (round(j, 12), round(summ["abr"], 12), -round(summ["mean_prompt_tokens_terminal"], 12), -round(l1, 12), -order)
        scored.append((key, cand, summ, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_key, best, best_summ, best_j = scored[0]
    n_tied_j = sum(1 for s in scored if s[0][0] == best_key[0])
    n_tied_j_abr = sum(1 for s in scored if s[0][:2] == best_key[:2])
    return {
        "procedure": "r1 (paper Sec. 4: finite candidate set on calibration trajectories; scorer, head and Platt fixed)",
        "objective": R1_OBJECTIVE,
        "alpha": alpha,
        "tie_break": R1_TIE_BREAK,
        "space": {k: list(space[k]) for k in POLICY_KEYS},
        "n_candidates": len(cands),
        "n_selection_sessions": len(sel),
        "best": dict(best),
        "best_J": best_j,
        "best_selection_summary": best_summ,
        "n_tied_at_best_J": n_tied_j,
        "n_tied_at_best_J_and_ABR": n_tied_j_abr,
        "top": [{"candidate": dict(c), "J": j, "selection": s} for _, c, s, j in scored[:top_k]],
    }


# ---------------------------------------------------------------------------
# Legacy (submitted) procedure -- unchanged
# ---------------------------------------------------------------------------


def split_examples_by_id(examples: list[dict[str, Any]], ratio: float = 0.67) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Legacy: deterministically split dialogue-level records by example id (ord-sum mod 100)."""
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for ex in examples:
        bucket = sum(ord(c) for c in str(ex["example_id"])) % 100
        (train if bucket < ratio * 100 else test).append(ex)
    return train, test


def replay_summary(examples: list[dict[str, Any]], config: ControllerConfig) -> dict[str, Any]:
    """Legacy replay (turn-level metrics; history counters never decay)."""
    records: list[dict[str, Any]] = []

    for ex in examples:
        turns = ex.get("turns", [])
        total_user_turns = len(turns) or 1
        prior_state: ControllerState | None = None
        prior_high_risk_turns = 0
        prior_cautious_turns = 0
        risk_memory = 0.0
        safe_streak = 0
        risky_streak = 0
        prev_calibrated = 0.0

        for turn_index, turn in enumerate(turns):
            calibrated = float(turn["calibrated_score"])
            uncertainty = float(turn["uncertainty"])
            tau_c = config.thresholds["tau_c"]
            pressure = max(0.0, calibrated - tau_c)
            effective_risk = calibrated
            if config.use_risk_memory:
                effective_risk = min(1.0, calibrated + config.risk_memory_gain * risk_memory)
            if config.use_trajectory_features:
                effective_risk = min(
                    1.0,
                    effective_risk
                    + config.trajectory_gain * risk_memory
                    + config.trajectory_trend_gain * max(0.0, calibrated - prev_calibrated)
                    - 0.05 * max(0, safe_streak - 1),
                )

            state = select_controller_state(
                calibrated=effective_risk,
                uncertainty=uncertainty,
                prior_state=prior_state,
                history_flag=prior_high_risk_turns > 0,
                safe_streak=safe_streak,
                risky_streak=risky_streak,
                config=config,
                dialogue_progress=(turn_index / max(total_user_turns - 1, 1)) if config.use_trajectory_features else 0.0,
                risk_memory=risk_memory,
                prev_risk_level=prev_calibrated,
            )
            decision = "block" if state == ControllerState.QUARANTINE else "allow"
            prompt_tokens = _prompt_tokens(state)
            label = str(turn["label"])
            records.append(
                {
                    "label": label,
                    "decision": decision,
                    "state": state.value,
                    "prompt_tokens": prompt_tokens,
                    "calibrated_score": calibrated,
                    "uncertainty": uncertainty,
                }
            )

            if state in {ControllerState.HARDENED, ControllerState.QUARANTINE}:
                prior_high_risk_turns += 1
            if state == ControllerState.CAUTIOUS:
                prior_cautious_turns += 1

            if config.use_risk_memory or config.use_trajectory_features:
                decay = config.risk_memory_decay if config.use_risk_memory else config.trajectory_decay
                risk_memory = decay * risk_memory + pressure
            else:
                risk_memory = 0.0

            if calibrated < tau_c:
                safe_streak += 1
                risky_streak = 0
            else:
                risky_streak += 1
                safe_streak = 0

            prior_state = state
            prev_calibrated = calibrated

    return summarize_records(records)


def search_trajectory_policy(
    examples: list[dict[str, Any]],
    base_config: ControllerConfig,
    *,
    train_ratio: float = 0.67,
) -> dict[str, Any]:
    """Legacy (submitted) search; see the module docstring for how it differs from the paper."""
    train_examples, test_examples = split_examples_by_id(examples, ratio=train_ratio)
    search_space = {
        "tau_c_offset": [-0.08, -0.05, -0.03, 0.0],
        "tau_h_offset": [-0.08, -0.05, -0.03, 0.0],
        "tau_q_offset": [-0.10, -0.07, -0.04, 0.0],
        "trajectory_gain": [0.20, 0.35, 0.50],
        "trajectory_trend_gain": [0.10, 0.18, 0.26],
        "trajectory_decay": [0.62, 0.72, 0.82],
        "safe_streak_to_downgrade": [1, 2],
        "risky_streak_to_escalate": [1, 2],
        "use_risk_memory": [False, True],
        "risk_memory_gain": [0.25, 0.50, 0.90],
    }

    best: dict[str, Any] | None = None
    tested = 0
    for values in product(
        search_space["tau_c_offset"],
        search_space["tau_h_offset"],
        search_space["tau_q_offset"],
        search_space["trajectory_gain"],
        search_space["trajectory_trend_gain"],
        search_space["trajectory_decay"],
        search_space["safe_streak_to_downgrade"],
        search_space["risky_streak_to_escalate"],
        search_space["use_risk_memory"],
        search_space["risk_memory_gain"],
    ):
        (
            tau_c_offset,
            tau_h_offset,
            tau_q_offset,
            trajectory_gain,
            trajectory_trend_gain,
            trajectory_decay,
            safe_streak_to_downgrade,
            risky_streak_to_escalate,
            use_risk_memory,
            risk_memory_gain,
        ) = values
        if not use_risk_memory and risk_memory_gain != search_space["risk_memory_gain"][0]:
            continue

        config = replace(
            base_config,
            num_states=4,
            allow_state_downgrade=True,
            history_decay=1,
            use_trajectory_features=True,
            use_risk_memory=use_risk_memory,
        )
        config.thresholds = {
            **base_config.thresholds,
            "tau_c": max(0.15, base_config.thresholds["tau_c"] + tau_c_offset),
            "tau_h": max(0.25, base_config.thresholds["tau_h"] + tau_h_offset),
            "tau_q": max(0.35, base_config.thresholds["tau_q"] + tau_q_offset),
        }
        config.trajectory_gain = trajectory_gain
        config.trajectory_trend_gain = trajectory_trend_gain
        config.trajectory_decay = trajectory_decay
        config.safe_streak_to_downgrade = safe_streak_to_downgrade
        config.risky_streak_to_escalate = risky_streak_to_escalate
        config.risk_memory_gain = risk_memory_gain

        summary = replay_summary(train_examples, config)
        tested += 1
        score = (
            summary["attack_block_rate"]
            + 0.20 * summary["benign_allow_rate"]
            - 0.002 * summary["mean_prompt_tokens"]
        )
        candidate = {
            "score": score,
            "config": {
                "tau_c": config.thresholds["tau_c"],
                "tau_h": config.thresholds["tau_h"],
                "tau_q": config.thresholds["tau_q"],
                "trajectory_gain": config.trajectory_gain,
                "trajectory_trend_gain": config.trajectory_trend_gain,
                "trajectory_decay": config.trajectory_decay,
                "safe_streak_to_downgrade": config.safe_streak_to_downgrade,
                "risky_streak_to_escalate": config.risky_streak_to_escalate,
                "use_risk_memory": config.use_risk_memory,
                "risk_memory_gain": config.risk_memory_gain,
            },
            "summary": summary,
        }
        if best is None or _better(candidate, best):
            best = candidate

    if best is None:
        raise RuntimeError("No trajectory policy candidates were evaluated.")

    best_config = replace(base_config)
    best_config.num_states = 4
    best_config.allow_state_downgrade = True
    best_config.history_decay = 1
    best_config.use_trajectory_features = True
    best_config.use_risk_memory = bool(best["config"]["use_risk_memory"])
    best_config.thresholds = {
        **base_config.thresholds,
        "tau_c": best["config"]["tau_c"],
        "tau_h": best["config"]["tau_h"],
        "tau_q": best["config"]["tau_q"],
    }
    best_config.trajectory_gain = best["config"]["trajectory_gain"]
    best_config.trajectory_trend_gain = best["config"]["trajectory_trend_gain"]
    best_config.trajectory_decay = best["config"]["trajectory_decay"]
    best_config.safe_streak_to_downgrade = best["config"]["safe_streak_to_downgrade"]
    best_config.risky_streak_to_escalate = best["config"]["risky_streak_to_escalate"]
    best_config.risk_memory_gain = best["config"]["risk_memory_gain"]

    test_summary = replay_summary(test_examples, best_config)
    return {
        "train_size": len(train_examples),
        "test_size": len(test_examples),
        "tested_candidates": tested,
        "best_config": best["config"],
        "train_summary": best["summary"],
        "test_summary": test_summary,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    harmful = [r for r in records if r["label"] == "harmful"]
    benign = [r for r in records if r["label"] == "benign"]
    blocked_harmful = sum(r["decision"] == "block" for r in harmful)
    allowed_benign = sum(r["decision"] == "allow" for r in benign)
    avg_tokens = sum(r["prompt_tokens"] for r in records) / max(len(records), 1)
    state_counts: dict[str, int] = {}
    for r in records:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    return {
        "num_records": len(records),
        "attack_block_rate": blocked_harmful / max(len(harmful), 1),
        "benign_allow_rate": allowed_benign / max(len(benign), 1),
        "false_positive_rate": 1.0 - (allowed_benign / max(len(benign), 1) if benign else 0.0),
        "mean_prompt_tokens": avg_tokens,
        "state_counts": state_counts,
    }


def _prompt_tokens(state: ControllerState) -> int:
    return {
        ControllerState.NORMAL: 6,
        ControllerState.CAUTIOUS: 28,
        ControllerState.HARDENED: 52,
        ControllerState.QUARANTINE: 18,
    }[state]


def _better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    if candidate["score"] != incumbent["score"]:
        return candidate["score"] > incumbent["score"]
    if candidate["summary"]["attack_block_rate"] != incumbent["summary"]["attack_block_rate"]:
        return candidate["summary"]["attack_block_rate"] > incumbent["summary"]["attack_block_rate"]
    if candidate["summary"]["benign_allow_rate"] != incumbent["summary"]["benign_allow_rate"]:
        return candidate["summary"]["benign_allow_rate"] > incumbent["summary"]["benign_allow_rate"]
    return candidate["summary"]["mean_prompt_tokens"] < incumbent["summary"]["mean_prompt_tokens"]
