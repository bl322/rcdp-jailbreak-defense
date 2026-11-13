"""Check that the llama-guard-prob scorer reproduces the one-shot P(unsafe) on the held-out set.

The RCDP scorer in ``--scorer llama-guard-prob`` mode and the supplementary P(unsafe) of
``scripts/scorer_baselines_r1/llama_guard_oneshot.py`` use the same chat template, greedy
decoding and verdict-token softmax, so the per-item raw scores must agree up to numerical noise.

    python scripts/lg_prob_check.py \
        --run results/r1/main_lgp_full_pool_calibration.json \
        --oneshot results/r1/cost/baselines/llama_guard_oneshot.json

Exit code 0 if every held-out item is present in both files and differs by at most --tol;
1 otherwise. The summary is printed and, with --output, written as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_scores(path: Path, variant: str, setting: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    for v in d["variants"]:
        if v["variant"] == variant and v["setting"] == setting:
            return {r["example_id"]: float(r["raw_score"]) for r in v["eval"]["heldout"]["records"]}
    raise SystemExit(f"{path}: no variant {variant!r} / setting {setting!r}")


def oneshot_scores(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    recs = d["records"]["heldout"] if isinstance(d["records"], dict) else d["records"]
    return {r["example_id"]: r.get("p_unsafe") for r in recs}


def compare(a: dict, b: dict, tol: float) -> dict:
    common = sorted(set(a) & set(b))
    diffs, flips, missing_prob = [], 0, 0
    for k in common:
        if b[k] is None:
            missing_prob += 1
            continue
        diffs.append(abs(a[k] - float(b[k])))
        flips += (a[k] >= 0.5) != (float(b[k]) >= 0.5)
    out = {
        "n_run": len(a), "n_oneshot": len(b), "n_common": len(common),
        "only_in_run": len(set(a) - set(b)), "only_in_oneshot": len(set(b) - set(a)),
        "oneshot_without_p_unsafe": missing_prob,
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
        "n_over_tol": sum(d > tol for d in diffs), "tol": tol,
        "verdict_flips_at_0.5": flips,
        "distinct_run_scores": len({round(x, 6) for x in a.values()}),
    }
    out["ok"] = (out["n_common"] == out["n_run"] == out["n_oneshot"] and missing_prob == 0
                 and out["n_over_tol"] == 0 and out["distinct_run_scores"] > 2)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run_r1_main.py output with --scorer llama-guard-prob")
    p.add_argument("--oneshot", required=True, help="llama_guard_oneshot.py output (with P(unsafe))")
    p.add_argument("--variant", default="rcdp-4state")
    p.add_argument("--setting", default="agnostic")
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    res = compare(run_scores(Path(args.run), args.variant, args.setting), oneshot_scores(Path(args.oneshot)), args.tol)
    for k, v in res.items():
        print(f"{k:28s} {v}")
    print("LG-PROB CHECK " + ("OK" if res["ok"] else "FAILED: send this output before using the run"))
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
