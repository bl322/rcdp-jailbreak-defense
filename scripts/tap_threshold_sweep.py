"""Test whether the published operating point is a plain-threshold stage.

Table ``tab:main_results`` names the row "RCDP risk estimation + Platt scaling
(benchmark-agnostic)" with ABR 0.7047 / BAR 0.7091, i.e. 105/149 blocked.  The
full finite-state controller does not reproduce that number under any of the 48
flag combinations tested by ``tap_config_sweep.py``.

This script tests the alternative reading: the row reports the *scoring +
calibration* stage on its own, i.e. block if the Platt-scaled probability
clears a single threshold, with the state machine not yet applied.  If one
threshold reproduces both published numbers, then 105 is reproducible and the
adaptive table's denominator is sound.
"""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402
from rcdp_guard.scorers import DistilBartMNLIRiskScorer  # noqa: E402

HELD204 = EXPERIMENTS / "data" / "paper_splits" / "heldout_204.jsonl"
CALIBRATION = EXPERIMENTS / "data" / "paper_splits" / "full_pool_calibration.jsonl"

TARGET_AGNOSTIC = {"abr": 0.7047, "bar": 0.7091}
TARGET_AWARE = {"abr": 0.8926, "bar": 0.7273}


def collect(benchmark_aware: bool) -> tuple[list[float], list[float]]:
    scorer = DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3",
        local_files_only=True,
        device="cpu",
    )
    controller = RCDPController(
        scorer=scorer,
        feature_config=FeatureConfig.all(),
        config=ControllerConfig(benchmark_aware=benchmark_aware),
    )
    fit_calibrator(controller, load_examples(CALIBRATION))

    harmful: list[float] = []
    benign: list[float] = []
    for ex in load_examples(HELD204):
        records = controller.evaluate_example(ex)
        if not records:
            continue
        last = records[-1]
        if last.label == "harmful":
            harmful.append(last.calibrated_score)
        else:
            benign.append(last.calibrated_score)
    return harmful, benign


def sweep(harmful: list[float], benign: list[float], target: dict) -> dict:
    best = None
    for step in range(1, 200):
        t = step / 200.0
        n_h = len(harmful)
        n_b = len(benign)
        blocked = sum(1 for s in harmful if s >= t)
        allowed = sum(1 for s in benign if s < t)
        abr = blocked / max(n_h, 1)
        bar = allowed / max(n_b, 1)
        err = abs(abr - target["abr"]) + abs(bar - target["bar"])
        if best is None or err < best["err"]:
            best = {
                "threshold": round(t, 4),
                "abr": round(abr, 4),
                "bar": round(bar, 4),
                "blocked": blocked,
                "n_harmful": n_h,
                "n_benign": n_b,
                "err": round(err, 4),
            }
    return best


def main() -> None:
    out = {}
    for ba in (False, True):
        key = "benchmark_aware=%s" % ba
        harmful, benign = collect(ba)
        target = TARGET_AGNOSTIC if not ba else TARGET_AWARE
        best = sweep(harmful, benign, target)
        out[key] = best
        print(
            f"[{key}] target abr={target['abr']} bar={target['bar']} -> "
            f"threshold={best['threshold']} abr={best['abr']} bar={best['bar']} "
            f"blocked={best['blocked']}/{best['n_harmful']} err={best['err']}",
            flush=True,
        )
        if best["blocked"] == 105:
            print(f"    ** reproduces the manuscript's 105 blocked **", flush=True)

    Path(__file__).resolve().parent.joinpath("tap_threshold_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
