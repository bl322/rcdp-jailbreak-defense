from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from .controller import ControllerConfig, ControllerState, select_controller_state


def split_examples_by_id(examples: list[dict[str, Any]], ratio: float = 0.67) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split dialogue-level records by example id."""
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for ex in examples:
        bucket = sum(ord(c) for c in str(ex["example_id"])) % 100
        (train if bucket < ratio * 100 else test).append(ex)
    return train, test


def replay_summary(examples: list[dict[str, Any]], config: ControllerConfig) -> dict[str, Any]:
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
