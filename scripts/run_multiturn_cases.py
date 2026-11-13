"""Multi-turn evaluation of revision R1 (Table S5 / Figure 4; ledger R2-1, R3-4).

Protocol (rcdp_guard/multiturn.py has the details):
  * One scorer (--scorer keyword | distilbart-mnli | llama-guard | llama-guard-prob, as in
    scripts/run_r1_main.py) is shared by every variant.
  * Every variant is calibrated with the paper's single-turn protocol on the
    single-turn calibration pool only (--calibration-dataset); dialogues are
    never used for fitting.
  * Dialogues are split into a selection and an evaluation side that are
    disjoint in seed clusters: --selection-dataset (e.g. the multi-turn
    expansion of the calibration pool) or, without it, a seeded stratified
    SHA-256 split of the clusters of --dataset.
  * Session-level ABR/BAR (harmful session blocked iff any user turn is
    blocked; benign session preserved iff every user turn is allowed) for each
    variant at its single-turn operating point, on both sides.
  * ABR-BAR frontier: the blocking threshold is shifted over a grid (the
    non-blocking thresholds keep their ratios); matched-BAR operating points are
    chosen on the selection side and reported on the evaluation side.
  * Trajectory-policy selection (paper Sec. 4) on the selection side
    (rcdp_guard.policy_search.search_trajectory_policy_r1).
  * Seed-clustered paired bootstrap on the evaluation side, plus an export
    that b5_cluster_bootstrap.py reads directly.
  * Per-session and per-turn records for every variant. A replay of every
    session is checked against the controller (any mismatch aborts).

The submitted runner is kept as scripts/run_multiturn_cases_legacy.py; ``--legacy``
as the first argument runs it with the remaining arguments (its own CLI:
--dataset --output --scorer ...).

Example (DistilBART; see runbook/multiturn.md):

    python scripts/run_multiturn_cases.py \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --selection-dataset data/generated/multiturn_calibration_588_r1.jsonl \
        --dataset data/generated/multiturn_heldout_612_r1.jsonl \
        --scorer distilbart-mnli --scorer-device cuda \
        --output results/r1/multiturn_distilbart.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard import multiturn as MT  # noqa: E402
from rcdp_guard import policy_search as PS  # noqa: E402
from rcdp_guard.calibration_protocol import DEFAULT_ALPHA, DEFAULT_SEED, PROTOCOLS, MemoizingScorer  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402

# Calibration features whose value depends on the controller state. The single-turn
# fit always sees them as 0, so their weights stay 0 and calibrated scores are
# state-independent; the replay relies on this and the runner checks it.
STATE_DEPENDENT_FEATURES = ("prior_high_risk_turns", "prior_cautious_turns", "turn_index", "dialogue_progress",
                            "risk_ema", "prev_risk_level", "safe_streak", "risky_streak")


class ScorerAdapter:
    """Scores with or without dialogue history and drops example metadata.

    The model scorers use metadata only as a cache key; dropping it lets identical
    (text, history) pairs share one model call across sessions.
    """

    def __init__(self, inner, use_history: bool):
        self.inner = inner
        self.use_history = use_history

    def score(self, text, history=None, metadata=None):
        return self.inner.score(text, list(history) if (history and self.use_history) else None, None)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def run_legacy(argv) -> None:
    """Submitted protocol (keyword scorer for the ablation variants, in-sample calibration, turn-level metrics)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import run_multiturn_cases_legacy as legacy

    sys.argv = [legacy.__file__] + list(argv)
    legacy.main()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--legacy", action="store_true",
                   help="must be the first argument: run the submitted runner (run_multiturn_cases_legacy.py) instead")
    p.add_argument("--calibration-dataset", required=True,
                   help="single-turn calibration pool, e.g. data/paper_splits/full_pool_calibration.jsonl")
    p.add_argument("--dataset", required=True,
                   help="multi-turn dialogues; all evaluation if --selection-dataset is given, else split by seed cluster")
    p.add_argument("--selection-dataset", default=None,
                   help="multi-turn selection dialogues (calibration trajectories); clusters must not overlap --dataset")
    p.add_argument("--selection-fraction", type=float, default=0.5, help="hash split only")
    p.add_argument("--split-seed", default=MT.DEFAULT_MT_SPLIT_SEED, help="hash split only")
    p.add_argument("--output", required=True)
    p.add_argument("--scorer", default="distilbart-mnli",
                   choices=["keyword", "distilbart-mnli", "llama-guard", "llama-guard-prob"],
                   help="llama-guard: verdict score (0.99/0.01, as submitted); "
                        "llama-guard-prob: P(unsafe) at the verdict token (graded)")
    p.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    p.add_argument("--scorer-local-files-only", action="store_true")
    p.add_argument("--scorer-device", default="cpu")
    p.add_argument("--scorer-cache-path", default=None)
    p.add_argument("--scorer-history", choices=["auto", "on", "off"], default="auto",
                   help="pass earlier turns to the scorer; auto = off for llama-guard and llama-guard-prob (as in the submitted runner), on otherwise")
    p.add_argument("--settings", default="agnostic,aware", help="comma list of agnostic, aware")
    p.add_argument("--variants", default="all", help="all | paper | shipped | comma list of names in rcdp_guard.multiturn.VARIANTS")
    p.add_argument("--calibration-protocol", default="r1", choices=PROTOCOLS)
    p.add_argument("--seed", default=DEFAULT_SEED, help="calibration split seed")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="J = BlockRate - alpha * FPR (calibration and policy search)")
    p.add_argument("--delta-step", type=float, default=0.01, help="frontier grid step of the blocking-threshold shift")
    p.add_argument("--no-frontier", action="store_true")
    p.add_argument("--export-delta-outcomes", action="store_true",
                   help="also write <output>.<setting>.delta_outcomes.json: evaluation-side blocked flag of every "
                        "session at every threshold shift (input of scripts/multiturn_matched_external.py)")
    p.add_argument("--bar-targets", default="fitted,0.70,0.80,0.90",
                   help="matched-BAR targets; 'fitted' = selection-side BAR of each matched variant at its operating point")
    p.add_argument("--policy-space", default="r1", help="r1 | small | none | path to a JSON dict of value lists")
    p.add_argument("--bootstrap", type=int, default=10000, help="resamples (0 disables; needs numpy)")
    p.add_argument("--keep-calibration-records", action="store_true")
    return p.parse_args(argv)


def build_scorer(args):
    from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer

    if args.scorer == "keyword":
        return KeywordRiskScorer()
    if args.scorer in ("llama-guard", "llama-guard-prob"):
        return LlamaGuardRiskScorer(model_name=args.scorer_model_name, local_files_only=args.scorer_local_files_only,
                                    device=args.scorer_device, cache_path=args.scorer_cache_path,
                                    score_mode="prob" if args.scorer == "llama-guard-prob" else "label")
    return DistilBartMNLIRiskScorer(model_name=args.scorer_model_name, local_files_only=args.scorer_local_files_only,
                                    device=args.scorer_device, cache_path=args.scorer_cache_path)


def code_version() -> dict:
    h = hashlib.sha256()
    files = sorted((ROOT / "rcdp_guard").glob("*.py")) + [Path(__file__).resolve()]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return {"sha256_rcdp_guard_and_runner": h.hexdigest(), "files": [f.name for f in files]}


def _jsonable(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def select_variants(spec: str) -> list:
    if spec == "all":
        return list(MT.VARIANTS)
    if spec in ("paper", "shipped"):
        return [n for n, (fam, _) in MT.VARIANTS.items() if fam == spec]
    names = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [n for n in names if n not in MT.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {list(MT.VARIANTS)}")
    return names


def load_policy_space(spec: str):
    if spec == "none":
        return None
    if spec == "r1":
        return PS.POLICY_SPACE_R1
    if spec == "small":
        return PS.POLICY_SPACE_SMALL
    return json.loads(Path(spec).read_text(encoding="utf-8"))


def build_sessions(cal, sel, ev, split_info):
    sel_c = {MT.cluster_id(e) for e in sel}
    ev_c = {MT.cluster_id(e) for e in ev}
    if sel_c & ev_c:
        raise SystemExit(f"selection and evaluation share {len(sel_c & ev_c)} seed clusters")
    cal_ids = {e.example_id for e in cal}
    leak = (ev_c | {e.example_id for e in ev}) & cal_ids
    if leak:
        raise SystemExit(f"{len(leak)} evaluation dialogues come from calibration-pool seeds, e.g. {sorted(leak)[:3]}")
    ids = [e.example_id for e in sel + ev]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate dialogue example_id values")
    sessions = []
    for side, exs in (("selection", sel), ("evaluation", ev)):
        for ex in exs:
            sessions.append({"example_id": ex.example_id, "cluster_id": MT.cluster_id(ex), "label": ex.label.value,
                             "benchmark": ex.benchmark, "side": side, "n_user_turns": MT.user_turns(ex), "example": ex})
    info = dict(split_info)
    info["counts"] = MT.side_counts(sessions)
    info["selection_clusters"] = sorted(sel_c)
    info["evaluation_clusters"] = sorted(ev_c)
    info["calibration_seed_overlap_with_evaluation"] = 0
    info["calibration_seed_overlap_with_selection"] = len(sel_c & cal_ids)
    return sessions, info


class VariantRun:
    def __init__(self, name, family, overrides, controller, report):
        self.name = name
        self.family = family
        self.overrides = overrides
        self.controller = controller
        self.config = controller.config
        self.report = report
        self.th_fn = PS.thresholds_function(controller.config)
        self.session_rows = []
        self.turn_rows = []
        self.probs = []
        self.mismatches = []

    def evaluate(self, sessions, reasons=False):
        for s in sessions:
            recs = self.controller.evaluate_example(s["example"])
            self.session_rows.append(MT.session_row(s, recs, self.config))
            self.turn_rows.extend(MT.turn_row(s, k, r, reasons) for k, r in enumerate(recs))
            p = [r.calibrated_score for r in recs]
            self.probs.append(p)
            routed = PS.replay_session(p, self.config, self.th_fn(s["benchmark"]))
            if [(x[0].value, x[1]) for x in routed] != [(r.state, r.decision) for r in recs]:
                self.mismatches.append(s["example_id"])
        return self

    def outcomes(self):
        return {i: {"blocked": r["blocked"]} for i, r in enumerate(self.session_rows)}


def fit_variant(name, family, overrides, scorer, cal, setting, args):
    controller = RCDPController(scorer=scorer, config=ControllerConfig(benchmark_aware=(setting == "aware"), **overrides))
    report = fit_calibrator(controller, cal, protocol=args.calibration_protocol, seed=args.seed, alpha=args.alpha)
    if not args.keep_calibration_records:
        report = {k: v for k, v in report.items() if k != "calibration_records"}
    weights = getattr(controller.calibrator, "weights", {}) or {}
    nonzero = {k: weights[k] for k in STATE_DEPENDENT_FEATURES if abs(weights.get(k, 0.0)) > 0}
    if nonzero:
        raise SystemExit(f"{name}: state-dependent calibration weights are non-zero {nonzero}; replay would be invalid")
    return VariantRun(name, family, overrides, controller, report)


def variant_payload(run: VariantRun, sessions, frontier_points, scorer_info) -> dict:
    sides = {}
    for side in ("selection", "evaluation"):
        rows = [r for r in run.session_rows if r["side"] == side]
        if rows:
            sides[side] = MT.summarize_sessions(rows)
    return {
        "family": run.family,
        "overrides": _jsonable(run.overrides),
        "config": _jsonable(asdict(run.config)),
        **scorer_info,
        "calibration": _jsonable(run.report),
        "replay_check": {"sessions": len(run.session_rows), "mismatches": len(run.mismatches),
                         "first_mismatches": run.mismatches[:5]},
        "summary": sides,
        "frontier": frontier_points,
        "sessions": run.session_rows,
        "turns": run.turn_rows,
    }


def parse_targets(spec, frontiers):
    targets = set()
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "fitted":
            for v in MT.MATCHED_VARIANTS:
                pts = frontiers.get(v) or []
                zero = [p for p in pts if p["delta"] == 0 and p.get("selection")]
                if zero and zero[0]["selection"]["bar"] is not None:
                    targets.add(round(zero[0]["selection"]["bar"], 6))
        else:
            targets.add(round(float(tok), 6))
    return sorted(targets)


def matched_bar(runs, sessions, frontiers, args):
    names = [v for v in MT.MATCHED_VARIANTS if v in runs and frontiers.get(v)]
    out = OrderedDict()
    for target in parse_targets(args.bar_targets, frontiers):
        entry = {"target_bar_selection": target, "variants": OrderedDict()}
        outs = OrderedDict()
        for v in names:
            run = runs[v]
            choice = MT.choose_matched_delta(frontiers[v], target)
            pt = choice["point"]
            bkey = PS.block_key(run.config.num_states)
            delta = pt["delta"]
            fn = (lambda b, _run=run, _d=delta, _k=bkey: PS.shift_block_threshold(_run.th_fn(b), _d, _k))
            ev_idx = [i for i, s in enumerate(sessions) if s["side"] == "evaluation"]
            o = PS.replay_outcomes(sessions, run.probs, run.config, fn, indices=ev_idx)
            outs[v] = o
            entry["variants"][v] = {"delta": delta, "block_thresholds": pt["block_thresholds"],
                                    "target_reached": choice["target_reached"],
                                    "selection": pt["selection"], "evaluation": PS.summarize_outcomes(sessions, o)}
        if args.bootstrap > 0 and outs:
            from rcdp_guard.multiturn_bootstrap import cluster_bootstrap

            entry["bootstrap_evaluation"] = cluster_bootstrap(
                MT.outcome_columns(sessions, outs), MT.MATCHED_CONTRASTS, B=args.bootstrap)
        out[f"{target:.6f}"] = entry
    return {
        "rule": "per variant: on the selection side, max ABR subject to BAR >= target (ties: higher BAR, "
                "smaller |delta|); reported on the evaluation side; CIs are conditional on the chosen delta",
        "targets": out,
    }


def run_policy_search(runs, sessions, scorer, cal, setting, args, space):
    base = runs[MT.POLICY_BASE_VARIANT]
    sel_idx = [i for i, s in enumerate(sessions) if s["side"] == "selection"]
    t0 = time.perf_counter()
    res = PS.search_trajectory_policy_r1(sessions, base.probs, sel_idx, base.config, base.th_fn,
                                         space=space, alpha=args.alpha)
    res["seconds"] = time.perf_counter() - t0
    best = res["best"]
    family, overrides = MT.VARIANTS[MT.POLICY_BASE_VARIANT]
    run = fit_variant(MT.POLICY_VARIANT, "policy", overrides, scorer, cal, setting, args)
    cfg = PS.candidate_config(run.controller.config, best)
    cfg.thresholds = PS.offset_thresholds(base.config.thresholds, best["d_tau_c"], best["d_tau_h"], best["d_tau_q"])
    cfg.benchmark_thresholds = (
        {b: PS.offset_thresholds(th, best["d_tau_c"], best["d_tau_h"], best["d_tau_q"])
         for b, th in base.config.benchmark_thresholds.items()}
        if base.config.benchmark_thresholds else None
    )
    run.controller.config = cfg
    run.config = cfg
    run.th_fn = PS.thresholds_function(cfg)
    run.overrides = {**overrides, "policy_candidate": best}
    run.evaluate(sessions)
    return res, run


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["--legacy"]:
        run_legacy(argv[1:])
        return
    args = parse_args(argv)
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    if [s for s in settings if s not in ("agnostic", "aware")]:
        raise SystemExit(f"unknown settings {settings}")
    variant_names = select_variants(args.variants)
    space = load_policy_space(args.policy_space)
    if space is not None and MT.POLICY_BASE_VARIANT not in variant_names:
        variant_names.append(MT.POLICY_BASE_VARIANT)

    base_scorer = build_scorer(args)
    use_history = args.scorer_history == "on" or (args.scorer_history == "auto" and not args.scorer.startswith("llama-guard"))
    scorer = MemoizingScorer(ScorerAdapter(base_scorer, use_history))
    scorer_info = {"scorer_class": type(base_scorer).__name__, "scorer_history": use_history}

    cal = load_examples(args.calibration_dataset)
    multi = [e.example_id for e in cal if MT.user_turns(e) != 1]
    if multi:
        raise SystemExit(f"--calibration-dataset must be single-turn (paper protocol); {len(multi)} are not, e.g. {multi[:3]}")
    pool = load_examples(args.dataset)
    if args.selection_dataset:
        sel, ev = load_examples(args.selection_dataset), pool
        split_info = {"mode": "file", "selection_dataset": args.selection_dataset, "evaluation_dataset": args.dataset}
    else:
        sel, ev, split_info = MT.split_by_cluster(pool, args.split_seed, args.selection_fraction)
        split_info["dataset"] = args.dataset
    sessions, split_info = build_sessions(cal, sel, ev, split_info)

    out = OrderedDict()
    out["meta"] = {
        "runner": "scripts/run_multiturn_cases.py", "argv": argv,
        "code_version": code_version(), "python": sys.version.split()[0], "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_dataset": args.calibration_dataset, "n_calibration": len(cal),
        "scorer": args.scorer, "scorer_model_name": args.scorer_model_name, **scorer_info,
        "calibration_protocol": args.calibration_protocol, "calibration_seed": args.seed, "alpha": args.alpha,
        "variants": variant_names, "policy_space": args.policy_space, "bootstrap_B": args.bootstrap,
    }
    out["protocol"] = {
        "calibration": "fit_calibrator on the single-turn calibration pool only (scorer, head, Platt, thresholds)",
        "session_rule": "harmful session blocked iff any user turn decision == block (Quarantine); benign session "
                        "preserved iff every user turn is allowed; Hardened counts as allowed (output check not modelled)",
        "tokens": "mean_prompt_tokens_terminal counts turns up to the first block (Quarantine terminal); "
                  "mean_prompt_tokens_all_turns counts every scripted turn",
        "frontier": "blocking threshold shifted by delta per benchmark (clipped to [0,1]); tau_c/tau_h scaled by the "
                    "same factor, tau_u fixed; delta = 0 is the single-turn operating point",
        "downgrade": "paper family (N-state-downgrade, N-state+memory, trajectory-policy): one level below the "
                     "previous state (Q->H->C->N; closed state sets for 2/3 states) on a turn that completes "
                     "k_down = safe_streak_to_downgrade consecutive turns with calibrated risk < tau_c; a step "
                     "consumes the run, otherwise the previous state is held (downgrade_rule='one-level'). "
                     "*-shipped variants: state re-derived every turn (submitted; downgrade_rule='rederive'). "
                     "Monotone N-state: never below the previous state",
        "selection": "matched-BAR deltas and the trajectory policy are chosen on the selection side only",
        "bootstrap": "seed clusters resampled within label, paired across variants (b5 algorithm)",
        "turn_records": "every variant stores every user turn; the scorer's 'reasons' strings are stored for the "
                        "first variant of each setting only (all variants share one scorer, so raw scores and "
                        "reasons are identical across variants)",
    }
    out["split"] = split_info
    out["settings"] = OrderedDict()
    deltas = MT.default_deltas(args.delta_step)

    for setting in settings:
        t_set = time.perf_counter()
        runs = OrderedDict()
        frontiers = {}
        douts = OrderedDict()
        for name in variant_names:
            family, overrides = MT.VARIANTS[name]
            t0 = time.perf_counter()
            run = fit_variant(name, family, overrides, scorer, cal, setting, args).evaluate(sessions, reasons=not runs)
            if run.mismatches:
                raise SystemExit(f"{setting}/{name}: replay differs from the controller on {len(run.mismatches)} sessions "
                                 f"(e.g. {run.mismatches[:3]})")
            runs[name] = run
            if not args.no_frontier:
                frontiers[name] = MT.frontier(sessions, run.probs, run.config, run.th_fn, deltas)
            if args.export_delta_outcomes:
                douts[name] = MT.delta_outcomes(sessions, run.probs, run.config, run.th_fn, deltas)
            ev = MT.summarize_sessions([r for r in run.session_rows if r["side"] == "evaluation"])
            print(f"{setting:8s} {name:28s} tau_block={run.config.thresholds[PS.block_key(run.config.num_states)]:.3f} "
                  f"eval ABR {ev['blocked_harmful']}/{ev['n_harmful']} BAR {ev['allowed_benign']}/{ev['n_benign']} "
                  f"tok {ev['mean_prompt_tokens_terminal']:.2f} first-block-turn {ev['first_block_turn']} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
            if hasattr(scorer, "_cache") and hasattr(scorer._cache, "flush"):
                scorer._cache.flush()

        policy = None
        if space is not None:
            policy, prun = run_policy_search(runs, sessions, scorer, cal, setting, args, space)
            runs[MT.POLICY_VARIANT] = prun
            if not args.no_frontier:
                frontiers[MT.POLICY_VARIANT] = MT.frontier(sessions, prun.probs, prun.config, prun.th_fn, deltas)
            if args.export_delta_outcomes:
                douts[MT.POLICY_VARIANT] = MT.delta_outcomes(sessions, prun.probs, prun.config, prun.th_fn, deltas)
            ev = MT.summarize_sessions([r for r in prun.session_rows if r["side"] == "evaluation"])
            policy["evaluation_summary"] = ev
            print(f"{setting:8s} {MT.POLICY_VARIANT:28s} best={policy['best']} J_sel={policy['best_J']:.4f} "
                  f"eval ABR {ev['blocked_harmful']}/{ev['n_harmful']} BAR {ev['allowed_benign']}/{ev['n_benign']} "
                  f"({policy['n_candidates']} candidates, {policy['seconds']:.0f}s)", flush=True)

        sout = OrderedDict()
        sout["variants"] = OrderedDict(
            (n, variant_payload(r, sessions, frontiers.get(n), scorer_info)) for n, r in runs.items())
        sout["policy_search"] = policy
        if frontiers:
            sout["matched_bar"] = matched_bar(runs, sessions, frontiers, args)
        if args.bootstrap > 0:
            from rcdp_guard.multiturn_bootstrap import cluster_bootstrap, format_table

            cols = MT.outcome_columns(sessions, OrderedDict((n, r.outcomes()) for n, r in runs.items()))
            sout["bootstrap_evaluation"] = cluster_bootstrap(cols, MT.CONTRASTS, B=args.bootstrap)
            print(format_table(sout["bootstrap_evaluation"]), flush=True)
        b5 = MT.b5_export(OrderedDict((n, r.session_rows) for n, r in runs.items()),
                          OrderedDict((n, r.turn_rows) for n, r in runs.items()))
        b5_path = Path(args.output).with_name(Path(args.output).stem + f".{setting}.b5.json")
        b5_path.parent.mkdir(parents=True, exist_ok=True)
        b5_path.write_text(json.dumps(b5, ensure_ascii=False), encoding="utf-8")
        sout["b5_export"] = str(b5_path)
        if douts:
            d_path = Path(args.output).with_name(Path(args.output).stem + f".{setting}.delta_outcomes.json")
            d_path.write_text(json.dumps({
                "setting": setting, "dataset": args.dataset, "scorer": args.scorer,
                "delta_step": args.delta_step, "side": "evaluation",
                "sessions": [{k: s[k] for k in ("example_id", "cluster_id", "label", "benchmark", "n_user_turns")}
                             for s in sessions if s["side"] == "evaluation"],
                "variants": douts}, ensure_ascii=False), encoding="utf-8")
            sout["delta_outcomes_export"] = str(d_path)
        sout["seconds"] = time.perf_counter() - t_set
        out["settings"][setting] = sout

    out["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
