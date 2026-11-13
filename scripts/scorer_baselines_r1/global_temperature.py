"""Table 1 baseline "Global temperature + thresholds" (revision R1, ledger X-2).

Definition implemented (recorded verbatim in the output under ``definition``):

  The paper names the row but does not define it beyond "a single global
  threshold is selected on the calibration split" (Sec. 4, method text). This
  script therefore uses: one temperature T > 0, fitted on calibration data by
  minimising the log-loss of p_T = sigmoid(z / T), where z = logit(raw scorer
  score) is the scorer logit (for the NLI scorers z = log(p_harm / p_benign) of
  the two-hypothesis entailment ratio); then one global blocking threshold tau
  on p_T, chosen on calibration data by RCDP's objective
  J = BlockRate - alpha * FPR (alpha = 1.0, grid 0.01..0.99, same tie-break).
  No benchmark identity, no features, no head, no state: a request is blocked
  iff p_T >= tau (for multi-turn input: iff any user turn reaches tau).

Data budget (same as RCDP's): the 196-example calibration pool is split with
``rcdp_guard.calibration_protocol.three_way_split`` (seed ``rcdp-r1``) into
D_head/D_platt/D_thresh = 100/48/48. T is fitted on D_head + D_platt (148, the
examples RCDP uses to fit its probability model) and tau is selected on
D_thresh (48) with ``search_threshold``. Both are configurable and recorded.

Because temperature scaling is monotone, the block set is the same as for a
threshold on the raw score; T only changes which raw cut-points the 0.01 grid
can reach and the reported probabilities (Brier/ECE).

Example:
    python scripts/scorer_baselines_r1/global_temperature.py \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --dataset data/paper_splits/heldout_204.jsonl \
        --scorer distilbart-mnli --scorer-model-name valhalla/distilbart-mnli-12-3 \
        --scorer-device cuda --output results/r1/baselines/global_temperature_distilbart.json
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from scorer_baselines_r1.common import (  # noqa: E402
    SCORER_KINDS, ScorerThresholdRouter, binary_summary, build_scorer, code_version, ensure_disjoint,
    environment, fmt_summary, load_partition, model_info, now, user_turn_scores, write_json,
)
from rcdp_guard.calibration_protocol import (  # noqa: E402
    DEFAULT_ALPHA, DEFAULT_SEED, SUBSET_NAMES, THRESHOLD_GRID, search_threshold, three_way_split,
)
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.scorers import logit, sigmoid  # noqa: E402

DEFINITION = (
    "Global temperature + thresholds: one temperature T > 0 fitted on calibration data (subsets {fit}) by "
    "minimising the log-loss of p_T = sigmoid(z / T), z = logit(raw scorer score) (logit clipped at 1e-6 as in "
    "rcdp_guard.scorers.logit); one global blocking threshold tau on p_T chosen on calibration subset(s) {thr} by "
    "J = BlockRate - alpha * FPR (alpha = {alpha}), grid 0.01..0.99, RCDP's tie-break (search_threshold with a "
    "stateless router). Block iff p_T >= tau (any user turn). Benchmark-agnostic; no features, head, Platt bias, "
    "history or state. The paper (Table 1 row name; Sec. 4 'a single global threshold is selected on the "
    "calibration split') gives no further definition. 'Thresholds' (plural) in the row name: search_threshold "
    "also rescales the non-blocking state thresholds (tau_c, tau_h; recorded under threshold_search.thresholds), "
    "but a stateless block/allow rule uses only the blocking threshold."
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Global temperature scaling + one global threshold (scorer-only baseline).")
    p.add_argument("--calibration-dataset", required=True)
    p.add_argument("--dataset", required=True, help="held-out partition (149 harmful / 55 benign)")
    p.add_argument("--external-dataset", action="append", default=[], help="extra evaluation set (never used for fitting)")
    p.add_argument("--output", required=True)
    p.add_argument("--scorer", default="distilbart-mnli", choices=SCORER_KINDS)
    p.add_argument("--scorer-model-name", default=None)
    p.add_argument("--scorer-device", default="cpu")
    p.add_argument("--scorer-dtype", default=None, help="e.g. float16 (default: as loaded)")
    p.add_argument("--scorer-local-files-only", action="store_true")
    p.add_argument("--scorer-cache-path", default=None)
    p.add_argument("--split-seed", default=DEFAULT_SEED)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--temperature-fit-subsets", default="head,platt",
                   help="comma list of head, platt, thresh, or 'all' (default: head,platt)")
    p.add_argument("--threshold-subsets", default="thresh", help="comma list or 'all' (default: thresh)")
    p.add_argument("--expect-counts", default="149,55", help="'harmful,benign' of --dataset; '' disables")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261113, help="bootstrap seed")
    return p.parse_args(argv)


def fit_inverse_temperature(z: list, y: list, lo: float = 1e-4, hi: float = 1e4) -> dict:
    """beta = 1/T minimising sum_i log(1 + exp(-s_i * beta * z_i)), s_i = +-1, beta in [lo, hi].

    The log-loss is convex in beta; its derivative sum_i (sigmoid(beta z_i) - y_i) z_i is
    increasing, so the minimiser is found by bisection on log(beta).
    """
    def grad(beta):
        return sum((sigmoid(beta * zi) - yi) * zi for zi, yi in zip(z, y))

    def nll(beta):
        tot = 0.0
        for zi, yi in zip(z, y):
            m = beta * zi if yi else -beta * zi
            tot += math.log1p(math.exp(-m)) if m > -30 else -m
        return tot

    if grad(lo) >= 0:
        beta, boundary = lo, "lower (scorer logit not positively associated with harm on the fit data)"
    elif grad(hi) <= 0:
        beta, boundary = hi, "upper (fit data separable by the logit)"
    else:
        a, b = math.log(lo), math.log(hi)
        for _ in range(200):
            mid = 0.5 * (a + b)
            if grad(math.exp(mid)) < 0:
                a = mid
            else:
                b = mid
        beta, boundary = math.exp(0.5 * (a + b)), None
    return {"beta": beta, "temperature": 1.0 / beta, "boundary": boundary,
            "nll_T1": nll(1.0) / len(z), "nll_fitted": nll(beta) / len(z), "n_fit_turns": len(z)}


def ece10(probs, labels):
    n = len(probs)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, s in enumerate(probs) if (lo <= s < hi) or (b == 9 and s == 1.0)]
        if idx:
            ece += len(idx) / n * abs(sum(probs[i] for i in idx) / len(idx) - sum(labels[i] for i in idx) / len(idx))
    return ece


def subsets_from(spec: str):
    return list(SUBSET_NAMES) if spec.strip() == "all" else [s.strip() for s in spec.split(",") if s.strip()]


def main(argv=None, scorer=None) -> dict:
    args = parse_args(argv)
    t_start = time.perf_counter()
    cal = load_examples(args.calibration_dataset)
    held = load_partition(args.dataset, args.expect_counts or None)
    ensure_disjoint(cal, held)
    externals = {}
    for path in args.external_dataset:
        exs = load_examples(path)
        ensure_disjoint(cal, exs)
        externals[Path(path).stem] = exs
    if scorer is None:
        scorer = build_scorer(args.scorer, args.scorer_model_name, args.scorer_device,
                              args.scorer_local_files_only, args.scorer_dtype, args.scorer_cache_path)

    memo = {}

    def scores(ex):
        if ex.example_id not in memo:
            memo[ex.example_id] = user_turn_scores(scorer, ex)
        return memo[ex.example_id]

    parts = dict(zip(SUBSET_NAMES, three_way_split(cal, args.split_seed)))
    fit_names, thr_names = subsets_from(args.temperature_fit_subsets), subsets_from(args.threshold_subsets)
    overlap = set(fit_names) & set(thr_names)
    fit_ex = [ex for name in fit_names for ex in parts[name]]
    thr_ex = [ex for name in thr_names for ex in parts[name]]

    z, y = [], []
    for ex in fit_ex:
        for t in scores(ex):
            z.append(logit(t["raw_score"]))
            y.append(1 if ex.label.value == "harmful" else 0)
    temp = fit_inverse_temperature(z, y)
    beta = temp["beta"]

    def turn_probs(ex):
        return [{"prob": sigmoid(beta * logit(t["raw_score"])), "raw_score": t["raw_score"]} for t in scores(ex)]

    router = ScorerThresholdRouter(turn_probs)
    search = search_threshold(router, thr_ex, alpha=args.alpha, grid=THRESHOLD_GRID)
    tau = search["selected"]["tau"]
    router.config.thresholds = dict(search["thresholds"])
    tau_eff = router.config.thresholds["tau_q"]  # the value the search evaluated (tau up to rounding)

    def record(ex, subset=None):
        ts = scores(ex)
        probs = [sigmoid(beta * logit(t["raw_score"])) for t in ts]
        k = max(range(len(probs)), key=lambda i: probs[i])
        rec = {"example_id": ex.example_id, "benchmark": ex.benchmark, "label": ex.label.value,
               "n_user_turns": len(ts), "raw_score": ts[k]["raw_score"], "scorer_logit": logit(ts[k]["raw_score"]),
               "prob_T": probs[k], "decision": "block" if probs[k] >= tau_eff else "allow"}
        if subset:
            rec["subset"] = subset
        if len(ts) > 1:
            rec["turn_raw_scores"] = [t["raw_score"] for t in ts]
        return rec

    cal_records = [record(ex, name) for name in SUBSET_NAMES for ex in parts[name]]
    held_records = [record(ex) for ex in held]
    ext_records = {name: [record(ex) for ex in exs] for name, exs in externals.items()}
    fit_probs = [sigmoid(beta * zi) for zi in z]

    summary = {"heldout": binary_summary(held_records, score_key="prob_T", B=args.bootstrap, seed=args.seed)}
    summary["heldout"]["brier_prob_T"] = sum((r["prob_T"] - (r["label"] == "harmful")) ** 2
                                             for r in held_records) / len(held_records)
    summary["heldout"]["ece10_prob_T"] = ece10([r["prob_T"] for r in held_records],
                                               [1 if r["label"] == "harmful" else 0 for r in held_records])
    for name, recs in ext_records.items():
        summary[name] = binary_summary(recs, score_key="prob_T", B=args.bootstrap, seed=args.seed)
    for name in SUBSET_NAMES:
        summary[f"calibration_{name}"] = binary_summary([r for r in cal_records if r["subset"] == name],
                                                        B=args.bootstrap, seed=args.seed)

    out = {
        "meta": {
            "runner": "scripts/scorer_baselines_r1/global_temperature.py", "argv": sys.argv[1:] if argv is None else argv,
            "code_version": code_version([__file__, Path(__file__).with_name("common.py")]),
            "started": now(), "seconds": None, "environment": environment(),
            "scorer": args.scorer, "scorer_model_name": args.scorer_model_name,
            "scorer_model": model_info(scorer), "scorer_device": args.scorer_device,
            "calibration_dataset": args.calibration_dataset, "dataset": args.dataset,
            "external_datasets": args.external_dataset,
        },
        "definition": DEFINITION.format(fit="+".join(fit_names), thr="+".join(thr_names), alpha=args.alpha),
        "protocol": {
            "split": {"function": "rcdp_guard.calibration_protocol.three_way_split", "seed": args.split_seed,
                      "sizes": {k: len(v) for k, v in parts.items()},
                      "ids": {k: [e.example_id for e in v] for k, v in parts.items()}},
            "temperature_fit_subsets": fit_names, "threshold_subsets": thr_names,
            "fit_and_threshold_subsets_overlap": sorted(overlap),
            "alpha": args.alpha, "decision_rule": "block iff p_T >= tau (any user turn)",
            "held_out_used_for": "reporting only",
        },
        "temperature": {**temp, "ece10_fit_T1": ece10([sigmoid(zi) for zi in z], y), "ece10_fit_fitted": ece10(fit_probs, y)},
        "threshold_search": search,
        "tau": tau,
        "tau_effective": tau_eff,
        "summary": summary,
        "records": {"heldout": held_records, "calibration": cal_records, **ext_records},
    }
    out["meta"]["seconds"] = time.perf_counter() - t_start
    path = write_json(args.output, out)
    print(f"T = {temp['temperature']:.4f} (fit on {'+'.join(fit_names)}, {temp['n_fit_turns']} turns"
          f"{'; boundary ' + temp['boundary'] if temp['boundary'] else ''}); tau = {tau} on {'+'.join(thr_names)} "
          f"(J = {search['selected']['J']:.4f}, tied run {search['tied_run']})")
    print(fmt_summary("held-out", summary["heldout"]))
    for name in ext_records:
        print(fmt_summary(name, summary[name]))
    print("wrote", path)
    return out


if __name__ == "__main__":
    main()
