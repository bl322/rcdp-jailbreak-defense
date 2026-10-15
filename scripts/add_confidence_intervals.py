from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.evaluation import bootstrap_binomial_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach deterministic 95% bootstrap intervals to an existing evaluation JSON."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    harmful = [row for row in records if row.get("label") == "harmful"]
    benign = [row for row in records if row.get("label") == "benign"]
    if not records:
        raise ValueError("The input JSON does not contain per-example records.")

    payload["confidence_intervals"] = {
        "method": "nonparametric percentile bootstrap",
        "confidence_level": 0.95,
        "attack_block_rate": bootstrap_binomial_ci(
            [row.get("controller_decision", row.get("decision")) == "block" for row in harmful],
            samples=args.samples,
            seed=20260903,
        ),
        "benign_allow_rate": bootstrap_binomial_ci(
            [row.get("controller_decision", row.get("decision")) == "allow" for row in benign],
            samples=args.samples,
            seed=20260904,
        ),
        "false_positive_rate": bootstrap_binomial_ci(
            [row.get("controller_decision", row.get("decision")) == "block" for row in benign],
            samples=args.samples,
            seed=20260905,
        ),
    }
    payload["confidence_intervals"]["note"] = (
        "Intervals are computed over example-level records; they quantify sampling "
        "uncertainty within this fixed evaluation set and do not replace external validation."
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["confidence_intervals"], indent=2))


if __name__ == "__main__":
    main()
