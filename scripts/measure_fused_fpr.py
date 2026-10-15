"""Measure the *combined* view's real benign cost.

The paper previously quoted the second view's own false-positive rates
(XSTest 0.0640, held-out benign 0.2364) as if they were the effect of adding
it.  They are not: under the escalate-only ``max`` combination the blocked set
is the union of the two views' blocks, so the combined false-positive rate can
only rise.  This script measures the combination end-to-end -- same
controller, calibrator re-fitted on the combined scores -- so the paper can
quote a measured number instead of a bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rcdp_hardened import DATA, build_controller  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.types import Example  # noqa: E402

LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LINES.append(msg)
    (HERE / "_fused_fpr.txt").write_text("\n".join(LINES), encoding="utf-8")


BENIGN_SETS = {
    "heldout204_benign": DATA / "paper_splits" / "heldout_204.jsonl",
    "xstest_safe_450": DATA / "external" / "xstest_safe_450.jsonl",
    "orbench_hard_250": DATA / "external" / "orbench_hard_benign_250.jsonl",
}
HARMFUL = DATA / "paper_splits" / "heldout_harmful_149.jsonl"

# component rates already measured, used as a self-check
KNOWN = {
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


def evaluate_set(controller, path: Path, label: str, limit: int | None = None):
    exs = [e for e in load_examples(str(path)) if e.label == label]
    if limit:
        exs = exs[:limit]
    blocked = 0
    for i, ex in enumerate(exs, start=1):
        if single_turn_blocked(controller, ex):
            blocked += 1
        if i % 50 == 0:
            log(f"    {i}/{len(exs)}  blocked={blocked}")
    return blocked, len(exs)


def main() -> None:
    out: dict = {"sets": {}}

    for kind in ("nli", "ensemble"):
        log(f"=== building controller: {kind} ===")
        controller = build_controller(kind, benchmark_aware=False)
        log(f"[=] calibrator fitted for {kind}")

        res: dict = {}
        log(f"--- harmful ({kind}) ---")
        b, n = evaluate_set(controller, HARMFUL, "harmful")
        res["harmful_149"] = {"blocked": b, "n": n, "rate": b / max(n, 1)}
        log(f"[=] harmful_149 blocked {b}/{n} = {b/max(n,1):.4f}")

        for name, path in BENIGN_SETS.items():
            log(f"--- {name} ({kind}) ---")
            b, n = evaluate_set(controller, path, "benign")
            res[name] = {"blocked": b, "n": n, "fpr": b / max(n, 1)}
            log(f"[=] {name} fpr {b}/{n} = {b/max(n,1):.4f}")
        out["sets"][kind] = res

    (HERE / "results" / "fused_fpr.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    log("")
    log("=== summary ===")
    log(f"{'set':22s} {'NLI':>8s} {'guard':>8s} {'fused':>8s}")
    for name in ["harmful_149", *BENIGN_SETS]:
        nli = out["sets"]["nli"][name]
        fus = out["sets"]["ensemble"][name]
        key = "rate" if name == "harmful_149" else "fpr"
        g = KNOWN["guard_alone"].get(name)
        log(
            f"{name:22s} {nli[key]:8.4f} "
            f"{(g if g is not None else float('nan')):8.4f} {fus[key]:8.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())
