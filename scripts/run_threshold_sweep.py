from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.evaluation import evaluate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["tau_c", "tau_h", "tau_q", "attack_block_rate", "benign_allow_rate", "false_positive_rate"],
        )
        writer.writeheader()
        for tau_c in (0.35, 0.45, 0.55):
            for tau_h in (0.60, 0.68, 0.75):
                for tau_q in (0.80, 0.84, 0.90):
                    controller = RCDPController(
                        config=ControllerConfig(
                            thresholds={"tau_c": tau_c, "tau_h": tau_h, "tau_q": tau_q, "tau_u": 0.28}
                        )
                    )
                    summary = evaluate_dataset(args.dataset, controller)
                    writer.writerow(
                        {
                            "tau_c": tau_c,
                            "tau_h": tau_h,
                            "tau_q": tau_q,
                            "attack_block_rate": summary["attack_block_rate"],
                            "benign_allow_rate": summary["benign_allow_rate"],
                            "false_positive_rate": summary["false_positive_rate"],
                        }
                    )


if __name__ == "__main__":
    main()
