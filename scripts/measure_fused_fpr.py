"""Measure the *combined* view's real benign cost.

The paper previously quoted the second view's own false-positive rates
(XSTest 0.0640, held-out benign 0.2364) as if they were the effect of adding
it.  They are not: under the escalate-only ``max`` combination the blocked set
is the union of the two views' blocks, so the combined false-positive rate can
only rise.  This script measures the combination end-to-end -- same
controller, calibrator re-fitted on the combined scores -- so the paper can
quote a measured number instead of a bound.

Revision R1: per-item decisions and k/n with exact Clopper-Pearson intervals are
saved, the fitted operating point of each controller is recorded
(rcdp_hardened.calibration_summary), the results directory is created, and the
scorer device is an argument, and the log goes to <output>.log. The ``SUBMITTED`` values are the manuscript's
numbers, printed only for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rcdp_hardened as RH  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.types import Example  # noqa: E402
from stats_r1 import clopper_pearson  # noqa: E402

DATA = RH.DATA
LINES: list = []
LOG = {"path": None}  # set in main(): <output>.log


def log(msg: str) -> None:
    print(msg, flush=True)
    LINES.append(msg)
    if LOG["path"] is not None:
        LOG["path"].write_text("\n".join(LINES), encoding="utf-8")


BENIGN_SETS = {
    "heldout204_benign": DATA / "paper_splits" / "heldout_204.jsonl",
    "xstest_safe_450": DATA / "external" / "xstest_safe_450.jsonl",
    "orbench_hard_250": DATA / "external" / "orbench_hard_benign_250.jsonl",
}
HARMFUL = DATA / "paper_splits" / "heldout_harmful_149.jsonl"

# manuscript values, printed for comparison only
SUBMITTED = {
    "nli": {"heldout204_benign": 0.4545, "xstest_safe_450": 0.5720, "orbench_hard_250": 0.2720},
    "guard_alone": {
        "heldout204_benign": 0.2364,
        "xstest_safe_450": 0.0640,
        "orbench_hard_250": 0.3880,
    },
}


def single_turn_blocked(controller, ex: Example) -> bool:
    records = controller.evaluate_example(ex)
    if not records:
        return True
    return records[-1].decision != "allow"


def evaluate_set(controller, path: Path, label: str, limit: int = 0):
    exs = [e for e in load_examples(str(path)) if e.label == label]
    if limit:
        exs = exs[:limit]
    items = []
    for i, ex in enumerate(exs, start=1):
        items.append({"example_id": ex.example_id, "blocked": single_turn_blocked(controller, ex)})
        if i % 50 == 0:
            log(f"    {i}/{len(exs)}  blocked={sum(it['blocked'] for it in items)}")
    b, n = sum(it["blocked"] for it in items), len(items)
    lo, hi = clopper_pearson(b, n) if n else (None, None)
    return {"blocked": b, "n": n, "rate": b / max(n, 1), "clopper_pearson95": [lo, hi], "items": items}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Combined (max) view: measured block and false-positive rates.")
    ap.add_argument("--kinds", default="nli,ensemble")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(HERE / "results" / "fused_fpr.json"))
    args = ap.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    LOG["path"] = output.with_suffix(".log")

    out: dict = {"sets": {}, "calibration": {}}
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for kind in kinds:
        log(f"=== building controller: {kind} ===")
        controller = RH.build_controller(kind, benchmark_aware=False, device=args.device)
        out["calibration"][kind] = RH.calibration_summary(controller)
        log(f"[=] calibrator fitted for {kind}: thresholds {controller.config.thresholds}")

        res: dict = {}
        log(f"--- harmful ({kind}) ---")
        res["harmful_149"] = evaluate_set(controller, HARMFUL, "harmful", args.limit)
        r = res["harmful_149"]
        log(f"[=] harmful_149 blocked {r['blocked']}/{r['n']} = {r['rate']:.4f}")
        for name, path in BENIGN_SETS.items():
            log(f"--- {name} ({kind}) ---")
            res[name] = evaluate_set(controller, path, "benign", args.limit)
            r = res[name]
            log(f"[=] {name} fpr {r['blocked']}/{r['n']} = {r['rate']:.4f}")
        out["sets"][kind] = res

    output.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    log("")
    log("=== summary (k/n, rate) ===")
    log(f"{'set':22s} " + " ".join(f"{k:>20s}" for k in kinds) + f" {'submitted guard':>16s}")
    for name in ["harmful_149", *BENIGN_SETS]:
        cells = []
        for k in kinds:
            r = out["sets"][k][name]
            cells.append(f"{r['blocked']:>4d}/{r['n']:<4d} {r['rate']:8.4f}")
        g = SUBMITTED["guard_alone"].get(name)
        log(f"{name:22s} " + " ".join(f"{c:>20s}" for c in cells) + f" {(g if g is not None else float('nan')):16.4f}")
    log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
