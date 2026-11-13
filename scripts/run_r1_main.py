"""Single-turn result tables of revision R1 in one command.

For one scorer, fits every requested controller variant on the calibration
pool with the paper's protocol (rcdp_guard/calibration_protocol.py) and
evaluates it on the held-out partition and on any external sets. Each variant
records its full calibration report (split IDs, head weights, Platt
parameters, threshold-search table, per-item calibration records) and
per-item evaluation records, so every table cell, the Platt ablation and the
reliability data can be recomputed from the output alone.

Example (DistilBART, both settings, all variants):

    python scripts/run_r1_main.py \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --heldout-dataset data/paper_splits/heldout_204.jsonl \
        --external-dataset data/external/mixed_harm149_xstest250.jsonl \
        --scorer distilbart-mnli --scorer-model-name valhalla/distilbart-mnli-12-3 \
        --scorer-device cuda --output results/r1/main_distilbart.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.calibration import FEATURE_GROUPS, FeatureConfig  # noqa: E402
from rcdp_guard.calibration_protocol import DEFAULT_ALPHA, DEFAULT_SEED, MemoizingScorer  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402

ALL_GROUPS = set(FEATURE_GROUPS)

# name -> (protocol, ControllerConfig overrides, feature groups)
VARIANTS = {
    "rcdp-4state": ("r1", {}, ALL_GROUPS),
    "3-state": ("r1", {"num_states": 3}, ALL_GROUPS),
    "2-state": ("r1", {"num_states": 2}, ALL_GROUPS),
    "no-platt": ("r1-no-platt", {}, ALL_GROUPS),
    "raw-threshold": ("raw-threshold", {}, ALL_GROUPS),
    "no-uncertainty-gate": ("r1", {"use_uncertainty_gate": False}, ALL_GROUPS),
    "drop-uncertainty-feature": ("r1", {}, ALL_GROUPS - {"uncertainty"}),
    "drop-length-lexical-features": ("r1", {}, ALL_GROUPS - {"length", "lexical"}),
    "legacy-submitted": ("legacy", {}, ALL_GROUPS),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calibration-dataset", required=True)
    p.add_argument("--heldout-dataset", required=True)
    p.add_argument("--external-dataset", action="append", default=[],
                   help="additional evaluation set (repeatable); never used for fitting")
    p.add_argument("--output", required=True)
    p.add_argument("--scorer", default="distilbart-mnli",
                   choices=["keyword", "distilbart-mnli", "llama-guard", "llama-guard-prob"],
                   help="llama-guard: verdict score (0.99/0.01, as submitted); "
                        "llama-guard-prob: P(unsafe) at the verdict token (graded)")
    p.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    p.add_argument("--scorer-local-files-only", action="store_true")
    p.add_argument("--scorer-device", default="cpu")
    p.add_argument("--scorer-cache-path", default=None)
    p.add_argument("--settings", default="agnostic,aware", help="comma list of agnostic, aware")
    p.add_argument("--variants", default=",".join(VARIANTS), help="comma list; default: all")
    p.add_argument("--seed", default=DEFAULT_SEED, help="split seed (the paper's value is reported in the output)")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--no-calibration-features", action="store_true",
                   help="drop per-item feature dicts from calibration records (smaller file)")
    return p.parse_args()


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


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def reliability(scores: list[float], labels: list[int], bins: int = 10) -> dict:
    rows = []
    ece = mce = 0.0
    n = len(scores)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        gap = abs(conf - acc)
        ece += len(idx) / n * gap
        mce = max(mce, gap)
        rows.append({"bin": [lo, hi], "count": len(idx), "mean_predicted": conf, "empirical_harmful": acc})
    brier = sum((s - y) ** 2 for s, y in zip(scores, labels)) / max(n, 1)
    return {"bins": rows, "ece10": ece, "mce10": mce, "brier": brier}


def evaluate(controller, examples) -> dict:
    recs = []
    for ex in examples:
        out = controller.evaluate_example(ex)
        last = out[-1]
        recs.append({
            "example_id": ex.example_id, "benchmark": ex.benchmark, "label": ex.label.value,
            "raw_score": last.raw_score, "head_logit": last.metadata.get("head_logit"),
            "calibrated_score": last.calibrated_score, "uncertainty": last.uncertainty,
            "state": last.state, "decision": last.decision, "prompt_tokens": last.prompt_tokens,
            "blocked_any_turn": any(r.decision == "block" for r in out), "n_turns": len(out),
        })
    harm = [r for r in recs if r["label"] == "harmful"]
    ben = [r for r in recs if r["label"] == "benign"]
    bh = sum(r["decision"] == "block" for r in harm)
    ab = sum(r["decision"] == "allow" for r in ben)
    states = {}
    for r in recs:
        key = f'{r["label"]}/{r["state"]}'
        states[key] = states.get(key, 0) + 1
    scores = [r["calibrated_score"] for r in recs]
    labels = [1 if r["label"] == "harmful" else 0 for r in recs]
    return {
        "n_harmful": len(harm), "blocked_harmful": bh, "abr": bh / len(harm) if harm else None,
        "n_benign": len(ben), "allowed_benign": ab, "bar": ab / len(ben) if ben else None,
        "fpr": (len(ben) - ab) / len(ben) if ben else None,
        "state_counts": dict(sorted(states.items())),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in recs) / max(len(recs), 1),
        "auc_calibrated": auc([r["calibrated_score"] for r in harm], [r["calibrated_score"] for r in ben]),
        "auc_raw": auc([r["raw_score"] for r in harm], [r["raw_score"] for r in ben]),
        "reliability": reliability(scores, labels),
        "records": recs,
    }


def main() -> None:
    args = parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS] + [s for s in settings if s not in {"agnostic", "aware"}]
    if unknown:
        raise SystemExit(f"unknown variants/settings: {unknown}")

    scorer = MemoizingScorer(build_scorer(args))  # one scorer pass shared by all variants
    cal = load_examples(args.calibration_dataset)
    evals = {"heldout": load_examples(args.heldout_dataset)}
    for path in args.external_dataset:
        evals[Path(path).stem] = load_examples(path)
    cal_ids = {ex.example_id for ex in cal}
    for name, exs in evals.items():
        overlap = cal_ids & {ex.example_id for ex in exs}
        if overlap:
            raise SystemExit(f"{name} shares {len(overlap)} example IDs with the calibration pool")

    out = {
        "meta": {
            "runner": "scripts/run_r1_main.py", "argv": sys.argv[1:], "code_version": code_version(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibration_dataset": args.calibration_dataset, "n_calibration": len(cal),
            "eval_datasets": {k: len(v) for k, v in evals.items()},
            "scorer": args.scorer, "scorer_model_name": args.scorer_model_name, "seed": args.seed, "alpha": args.alpha,
        },
        "variants": [],
    }
    for setting in settings:
        for name in variants:
            protocol, overrides, groups = VARIANTS[name]
            controller = RCDPController(
                scorer=scorer,
                feature_config=FeatureConfig(enabled_groups=set(groups)),
                config=ControllerConfig(benchmark_aware=(setting == "aware"), **overrides),
            )
            t0 = time.perf_counter()
            report = fit_calibrator(controller, cal, protocol=protocol, seed=args.seed, alpha=args.alpha)
            if args.no_calibration_features:
                for r in report.get("calibration_records", []):
                    r.pop("features", None)
            entry = {
                "variant": name, "setting": setting, "protocol": protocol,
                "config": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
                           for k, v in asdict(controller.config).items()},
                "feature_groups": sorted(groups),
                "calibration": report,
                "eval": {k: evaluate(controller, exs) for k, exs in evals.items()},
                "seconds": time.perf_counter() - t0,
            }
            out["variants"].append(entry)
            h = entry["eval"]["heldout"]
            print(f"{setting:8s} {name:30s} tau={report['thresholds'].get('tau_q', float('nan')):.3f} "
                  f"held-out ABR {h['blocked_harmful']}/{h['n_harmful']} BAR {h['allowed_benign']}/{h['n_benign']} "
                  f"AUC {h['auc_calibrated'] if h['auc_calibrated'] is not None else math.nan:.4f} "
                  f"ECE {h['reliability']['ece10']:.4f}", flush=True)
            if hasattr(scorer, "_cache") and hasattr(scorer._cache, "flush"):
                scorer._cache.flush()

    out["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
