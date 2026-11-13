"""Calibration protocol of the paper (revision R1).

Stages, all on the calibration pool only:

1. Split. The pool is partitioned into three disjoint subsets
   D_head / D_platt / D_thresh in the proportions 100 : 48 : 48 (exactly
   100/48/48 for the 196-example pool). Examples are ordered by
   (label, benchmark, SHA-256("<seed>:<example_id>")) and dealt to the subset
   with the largest quota deficit, which stratifies by label and benchmark and
   hits the subset sizes exactly.
2. Risk head. L2-regularised logistic regression (C = 0.5, i.e. lambda = 1.0;
   learning rate 0.1; 300 full-batch epochs) on D_head.
3. Platt layer. r = sigmoid(a * s~ + b_cal) fitted on the head logits of
   D_platt (Platt targets, Newton method).
4. Operating point. The blocking threshold is chosen on D_thresh by maximising
   J = BlockRate - alpha * FPR (alpha = 1.0) over the grid {0.01, ..., 0.99}.
   Ties: higher BlockRate, then the median grid point of the tied run. The
   non-blocking thresholds keep their default ratios to the blocking one
   (tau_c = 0.45/0.84 tau_q, tau_h = 0.68/0.84 tau_q) and tau_u = 0.28; they
   only route allowed requests and do not enter J. In benchmark-aware mode a
   threshold is additionally chosen per benchmark on that benchmark's harmful
   D_thresh items, with FPR measured on all D_thresh benign items (the pool's
   benign examples come from one benchmark only).

Ablation protocols reuse the same split and search:
``r1-no-platt`` (head on D_head, r = sigmoid(s~), D_platt unused) and
``raw-threshold`` (raw scorer output, no head and no Platt).
``legacy`` reproduces the submitted pipeline: unpenalised head on the whole
pool, no Platt, default thresholds, turn-1 trend boost.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from .calibration import LogisticCalibrator, PlattCalibratedHead, PlattScaler, RawScoreCalibrator
from .types import Label

PROTOCOLS = ("r1", "r1-no-platt", "raw-threshold", "legacy")
SPLIT_SIZES_196 = (100, 48, 48)
SUBSET_NAMES = ("head", "platt", "thresh")
DEFAULT_SEED = "rcdp-r1"
DEFAULT_ALPHA = 1.0
RIDGE_LAMBDA = 1.0  # C = 0.5
HEAD_LR = 0.1
HEAD_EPOCHS = 300
THRESHOLD_GRID = tuple(round(0.01 * k, 2) for k in range(1, 100))
DEFAULT_THRESHOLDS = {"tau_c": 0.45, "tau_h": 0.68, "tau_q": 0.84, "tau_u": 0.28}
TIE_BREAK = "max J; then max BlockRate; then median grid point of the tied run (lower median)"


class MemoizingScorer:
    """Caches scorer outputs so threshold search and repeated variants do not rescore."""

    def __init__(self, scorer):
        self._scorer = scorer
        self._memo: dict = {}

    def score(self, text, history=None, metadata=None):
        key = (
            text,
            tuple(history or ()),
            json.dumps(metadata or {}, sort_keys=True, default=str),
        )
        if key not in self._memo:
            self._memo[key] = self._scorer.score(text, history, metadata)
        return self._memo[key]

    def __getattr__(self, name):
        return getattr(self._scorer, name)


def split_key(example_id: str, seed: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), example_id


def subset_sizes(n: int) -> tuple[int, int, int]:
    """Largest-remainder apportionment of n in the ratio 100 : 48 : 48."""
    total = sum(SPLIT_SIZES_196)
    quotas = [n * w / total for w in SPLIT_SIZES_196]
    sizes = [int(q) for q in quotas]
    order = sorted(range(3), key=lambda j: (-(quotas[j] - sizes[j]), j))
    for j in order[: n - sum(sizes)]:
        sizes[j] += 1
    return tuple(sizes)


def three_way_split(examples, seed: str = DEFAULT_SEED):
    """Return (D_head, D_platt, D_thresh), stratified by (label, benchmark)."""
    n = len(examples)
    targets = subset_sizes(n)
    ordered = sorted(
        examples,
        key=lambda ex: (ex.label.value, str(ex.benchmark).lower(), *split_key(ex.example_id, seed)),
    )
    parts: tuple[list, list, list] = ([], [], [])
    for k, ex in enumerate(ordered):
        deficits = [targets[j] * (k + 1) / n - len(parts[j]) for j in range(3)]
        j = max(range(3), key=lambda i: (deficits[i], -i))
        parts[j].append(ex)
    assert tuple(len(p) for p in parts) == targets
    return parts


def calibration_rows(controller, examples):
    """Feature rows of the user turns, exactly as the submitted fit_calibrator built them."""
    rows, labels, owners = [], [], []
    for ex in examples:
        history: list[str] = []
        for turn in ex.turns:
            if turn.role != "user":
                history.append(turn.text)
                continue
            score_result = controller.scorer.score(turn.text, history, ex.metadata)
            features = controller.calibrate(
                raw_score=score_result.raw_score,
                text=turn.text,
                benchmark=ex.benchmark,
                prior_high_risk_turns=0,
                prior_cautious_turns=0,
            ).features
            rows.append(features)
            labels.append(1 if ex.label == Label.HARMFUL else 0)
            owners.append(ex)
            history.append(turn.text)
    return rows, labels, owners


def block_key_for(controller) -> str:
    # select_controller_state blocks at tau_h in the 2-state variant, at tau_q otherwise.
    return "tau_h" if controller.config.num_states == 2 else "tau_q"


def thresholds_for_block(tau: float, block_key: str) -> dict[str, float]:
    d = DEFAULT_THRESHOLDS
    scale = tau / d[block_key]
    return {
        "tau_c": d["tau_c"] * scale,
        "tau_h": d["tau_h"] * scale,
        "tau_q": min(1.0, d["tau_q"] * scale),
        "tau_u": d["tau_u"],
    }


def _blocked(controller, example) -> bool:
    return any(r.decision == "block" for r in controller.evaluate_example(example))


def search_threshold(
    controller,
    examples,
    *,
    alpha: float = DEFAULT_ALPHA,
    grid=THRESHOLD_GRID,
    harmful_filter: Callable | None = None,
) -> dict:
    block_key = block_key_for(controller)
    harmful = [ex for ex in examples if ex.label == Label.HARMFUL and (harmful_filter is None or harmful_filter(ex))]
    benign = [ex for ex in examples if ex.label == Label.BENIGN]
    if not harmful or not benign:
        raise ValueError("threshold search needs harmful and benign examples")
    saved = (controller.config.thresholds, controller.config.benchmark_thresholds)
    controller.config.benchmark_thresholds = None
    table = []
    try:
        for tau in grid:
            controller.config.thresholds = thresholds_for_block(tau, block_key)
            bh = sum(_blocked(controller, ex) for ex in harmful)
            bb = sum(_blocked(controller, ex) for ex in benign)
            br, fpr = bh / len(harmful), bb / len(benign)
            table.append({"tau": tau, "blocked_harmful": bh, "n_harmful": len(harmful),
                          "blocked_benign": bb, "n_benign": len(benign),
                          "block_rate": br, "fpr": fpr, "J": br - alpha * fpr})
    finally:
        controller.config.thresholds, controller.config.benchmark_thresholds = saved
    best_j = max(r["J"] for r in table)
    tied = [r for r in table if abs(r["J"] - best_j) < 1e-12]
    best_br = max(r["block_rate"] for r in tied)
    tied = [r for r in tied if abs(r["block_rate"] - best_br) < 1e-12]
    chosen = tied[(len(tied) - 1) // 2]
    return {
        "objective": "J = BlockRate - alpha * FPR",
        "alpha": alpha,
        "grid": [min(grid), max(grid), round(grid[1] - grid[0], 4), len(grid)],
        "block_key": block_key,
        "tie_break": TIE_BREAK,
        "tied_run": [tied[0]["tau"], tied[-1]["tau"]],
        "selected": chosen,
        "thresholds": thresholds_for_block(chosen["tau"], block_key),
        "table": table,
    }


def fit_protocol(controller, examples, protocol: str = "r1", *, seed: str = DEFAULT_SEED,
                 alpha: float = DEFAULT_ALPHA) -> dict:
    """Fit the calibration stages of ``protocol`` on ``examples`` and configure ``controller``.

    Returns a JSON-serialisable report (split IDs, head weights, Platt
    parameters, the full threshold-search table and per-turn calibration records).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown calibration protocol {protocol!r}; expected one of {PROTOCOLS}")
    original_scorer = controller.scorer
    if not isinstance(original_scorer, MemoizingScorer):
        controller.scorer = MemoizingScorer(original_scorer)
    try:
        report = _fit(controller, examples, protocol, seed, alpha)
    finally:
        controller.scorer = original_scorer
    controller.calibration_report = report
    return report


def _fit(controller, examples, protocol, seed, alpha):
    cfg = controller.config
    if protocol == "legacy":
        rows, labels, _ = calibration_rows(controller, examples)
        controller.calibrator = LogisticCalibrator()
        controller.calibrator.fit(rows, labels)
        cfg.legacy_first_turn_trend = True
        cfg.benchmark_thresholds = None
        return {
            "protocol": "legacy",
            "description": "submitted pipeline: unpenalised head on the whole pool, no Platt, "
                           "default thresholds, turn-1 trend boost",
            "n_calibration": len(examples),
            "head": {"lr": HEAD_LR, "epochs": HEAD_EPOCHS, "l2_lambda": 0.0,
                     "weights": dict(controller.calibrator.weights)},
            "platt": None,
            "thresholds": dict(cfg.thresholds),
            "benchmark_thresholds": None,
            "calibration_records": _records(controller, {"all": examples}),
        }

    cfg.legacy_first_turn_trend = False
    d_head, d_platt, d_thresh = three_way_split(examples, seed)
    subsets = dict(zip(SUBSET_NAMES, (d_head, d_platt, d_thresh)))

    head = None
    platt = None
    if protocol == "raw-threshold":
        controller.calibrator = RawScoreCalibrator()
    else:
        rows, labels, _ = calibration_rows(controller, d_head)
        head = LogisticCalibrator(lr=HEAD_LR, epochs=HEAD_EPOCHS, l2=RIDGE_LAMBDA)
        head.fit(rows, labels)
        if protocol == "r1":
            prow, plab, _ = calibration_rows(controller, d_platt)
            platt = PlattScaler().fit([head.decision_function(r) for r in prow], plab)
        controller.calibrator = PlattCalibratedHead(head, platt)

    search = search_threshold(controller, d_thresh, alpha=alpha)
    cfg.thresholds = dict(search["thresholds"])
    cfg.benchmark_thresholds = None
    per_benchmark = None
    if cfg.benchmark_aware:
        per_benchmark = {}
        for bench in sorted({str(ex.benchmark).lower() for ex in d_thresh if ex.label == Label.HARMFUL}):
            res = search_threshold(controller, d_thresh, alpha=alpha,
                                   harmful_filter=lambda ex, b=bench: str(ex.benchmark).lower() == b)
            per_benchmark[bench] = res
        cfg.benchmark_thresholds = {b: dict(r["thresholds"]) for b, r in per_benchmark.items()}

    return {
        "protocol": protocol,
        "seed": seed,
        "split": {
            "rule": "order by (label, benchmark, SHA-256('<seed>:<example_id>')); deal to largest quota deficit",
            "sizes": {k: len(v) for k, v in subsets.items()},
            "label_counts": {k: _label_counts(v) for k, v in subsets.items()},
            "ids": {k: [ex.example_id for ex in v] for k, v in subsets.items()},
        },
        "head": None if head is None else {
            "fitted_on": "head", "lr": HEAD_LR, "epochs": HEAD_EPOCHS,
            "l2_lambda": RIDGE_LAMBDA, "C": 1.0 / (2.0 * RIDGE_LAMBDA),
            "weights": dict(head.weights),
        },
        "platt": None if platt is None else {
            "fitted_on": "platt", "a": platt.a, "b_cal": platt.b, "n": platt.n_fit,
            "targets": list(platt.targets),
        },
        "threshold_search": search,
        "thresholds": dict(cfg.thresholds),
        "benchmark_threshold_search": per_benchmark,
        "benchmark_thresholds": cfg.benchmark_thresholds,
        "calibration_records": _records(controller, subsets),
    }


def _label_counts(examples) -> dict:
    out: dict[str, int] = {}
    for ex in examples:
        key = f"{ex.label.value}/{str(ex.benchmark).lower()}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _records(controller, subsets: dict) -> list[dict]:
    out = []
    for name, exs in subsets.items():
        for ex in exs:
            for rec in controller.evaluate_example(ex):
                out.append({
                    "subset": name,
                    "example_id": rec.example_id,
                    "benchmark": rec.benchmark,
                    "label": rec.label,
                    "raw_score": rec.raw_score,
                    "head_logit": rec.metadata.get("head_logit"),
                    "calibrated_score": rec.calibrated_score,
                    "state": rec.state,
                    "decision": rec.decision,
                    "features": rec.metadata.get("features"),
                })
    return out
