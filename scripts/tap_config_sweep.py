"""Find which controller configuration reproduces the manuscript's operating point.

The manuscript reports, on the 204-example held-out partition:

    benchmark-agnostic : ABR 0.7047, BAR 0.7091, FPR 0.2909
    benchmark-aware    : ABR 0.8926, BAR 0.7273, FPR 0.2727   (diagnostic)

Table ``tab:adaptive`` then uses 105/149 blocked as its denominator.  A plain
run of the current code gives 135/149 (benchmark-aware) and 117/149
(benchmark-agnostic), i.e. neither published point.  This sweep walks the
controller flags to identify the configuration that actually produced the
published numbers, so the adaptive-attack table is anchored to a reproducible
configuration instead of an assumption.
"""

from __future__ import annotations

import os

import itertools
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

TARGETS = {
    "benchmark_agnostic": {"abr": 0.7047, "bar": 0.7091},
    "benchmark_aware": {"abr": 0.8926, "bar": 0.7273},
}


def evaluate(controller: RCDPController, examples) -> dict:
    harmful_total = harmful_blocked = 0
    benign_total = benign_allowed = 0
    for ex in examples:
        records = controller.evaluate_example(ex)
        if not records:
            continue
        last = records[-1]
        if last.label == "harmful":
            harmful_total += 1
            harmful_blocked += last.decision == "block"
        else:
            benign_total += 1
            benign_allowed += last.decision == "allow"
    return {
        "n_harmful": harmful_total,
        "n_benign": benign_total,
        "blocked": harmful_blocked,
        "abr": round(harmful_blocked / max(harmful_total, 1), 4),
        "bar": round(benign_allowed / max(benign_total, 1), 4),
        "fpr": round(1 - benign_allowed / max(benign_total, 1), 4),
    }


def main() -> None:
    examples = load_examples(HELD204)
    calib = load_examples(CALIBRATION)
    print(f"heldout204={len(examples)}  calibration={len(calib)}", flush=True)

    # One scorer reused across every configuration: the NLI score does not depend
    # on the controller flags, so this turns an N-fold rescan into a single scan.
    shared_scorer = DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3",
        local_files_only=True,
        device="cpu",
    )
    print("[...] warming the shared scorer over the calibration split", flush=True)
    warm = RCDPController(scorer=shared_scorer, feature_config=FeatureConfig.all())
    fit_calibrator(warm, calib)
    print("[=] scorer warm", flush=True)

    results = []
    grid = list(
        itertools.product(
            [False, True],  # benchmark_aware
            [True, False],  # use_trajectory_features
            [True, False],  # use_uncertainty_gate
            [True, False],  # use_history
            [4, 3, 2],      # num_states
        )
    )

    for ba, traj, unc, hist, nstates in grid:
        controller = RCDPController(
            scorer=shared_scorer,
            feature_config=FeatureConfig.all(),
            config=ControllerConfig(
                benchmark_aware=ba,
                use_trajectory_features=traj,
                use_uncertainty_gate=unc,
                use_history=hist,
                num_states=nstates,
            ),
        )
        fit_calibrator(controller, calib)
        metrics = evaluate(controller, examples)
        row = {
            "benchmark_aware": ba,
            "use_trajectory_features": traj,
            "use_uncertainty_gate": unc,
            "use_history": hist,
            "num_states": nstates,
            **metrics,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    # Closest matches to the published operating points.
    for name, target in TARGETS.items():
        want_ba = name == "benchmark_aware"
        pool = [r for r in results if r["benchmark_aware"] == want_ba]
        best = min(
            pool,
            key=lambda r: abs(r["abr"] - target["abr"]) + abs(r["bar"] - target["bar"]),
        )
        print(
            f"\n[best {name}] target abr={target['abr']} bar={target['bar']}\n"
            f"  -> {json.dumps(best)}\n"
            f"  delta_abr={abs(best['abr']-target['abr']):.4f} "
            f"delta_bar={abs(best['bar']-target['bar']):.4f}",
            flush=True,
        )

    out = Path(__file__).resolve().parent / "tap_config_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[+] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
