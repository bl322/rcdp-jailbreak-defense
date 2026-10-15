"""Offline protocol check for the adaptive-attack table.

Purpose
-------
The CMC manuscript (Table ``tab:adaptive``) reports that 105 of the 149
held-out harmful prompts are blocked before adaptation, using a
"benchmark-agnostic DistilBART" configuration.  The archived result files in
``experiments/results`` report ``originally_blocked`` = 135 (distilbart) and
143 (llama-guard).  This script re-runs the *real* RCDP controller over the
*real* held-out split and reports the blocked count under several
configurations, so that the denominator used in the paper is identified from
measurement rather than assumption.

It needs no network access and no attacker model: it only exercises the
defence side (scorer + calibrator + controller).

Usage
-----
    <miniconda python> tap_protocol_check.py
"""

from __future__ import annotations

import os

import json
import sys
import time
from pathlib import Path

EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402
from rcdp_guard.scorers import DistilBartMNLIRiskScorer  # noqa: E402

HELDOUT = EXPERIMENTS / "data" / "paper_splits" / "heldout_harmful_149.jsonl"
CALIBRATION = EXPERIMENTS / "data" / "paper_splits" / "full_pool_calibration.jsonl"


def build_controller(*, benchmark_aware: bool) -> RCDPController:
    scorer = DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3",
        local_files_only=True,
        device="cpu",
    )
    config = ControllerConfig(benchmark_aware=benchmark_aware)
    controller = RCDPController(
        scorer=scorer,
        feature_config=FeatureConfig.all(),
        config=config,
    )
    fit_calibrator(controller, load_examples(CALIBRATION))
    return controller


def scan(controller: RCDPController) -> dict:
    examples = load_examples(HELDOUT)
    blocked = 0
    harmful = 0
    per_example = []
    for ex in examples:
        records = controller.evaluate_example(ex)
        if not records:
            continue
        last = records[-1]
        if last.label != "harmful":
            continue
        harmful += 1
        is_blocked = last.decision == "block"
        blocked += is_blocked
        per_example.append(
            {
                "example_id": ex.example_id,
                "blocked": is_blocked,
                "state": last.state,
                "raw_score": round(last.raw_score, 4),
                "calibrated_score": round(last.calibrated_score, 4),
            }
        )
    return {
        "harmful_examples": harmful,
        "originally_blocked": blocked,
        "pass_through": harmful - blocked,
        "block_rate": blocked / max(harmful, 1),
        "per_example": per_example,
    }


def main() -> None:
    started = time.time()
    report = {
        "heldout": str(HELDOUT),
        "calibration": str(CALIBRATION),
        "scorer": "distilbart-mnli (valhalla/distilbart-mnli-12-3, local files, cpu)",
        "variants": {},
    }

    for benchmark_aware in (True, False):
        key = "benchmark_aware=%s" % benchmark_aware
        print(f"[...] fitting + scanning {key}", flush=True)
        controller = build_controller(benchmark_aware=benchmark_aware)
        result = scan(controller)
        summary = {k: v for k, v in result.items() if k != "per_example"}
        report["variants"][key] = summary
        print(f"[=] {key}: {json.dumps(summary)}", flush=True)
        report["variants"][key]["per_example"] = result["per_example"]

    report["elapsed_sec"] = round(time.time() - started, 1)
    out = Path(__file__).resolve().parent / "tap_protocol_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
