from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aggregate ablation diagnostics from saved per-example records."
    )
    parser.add_argument("--input", required=True, help="Saved ablation JSON file")
    parser.add_argument("--output", required=True, help="CSV output path")
    return parser.parse_args()


def counts(records: list[dict], label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        if record["label"] == label:
            state = record["state"]
            result[state] = result.get(state, 0) + 1
    return result


def mean_tokens(records: list[dict], label: str) -> float:
    selected = [r["prompt_tokens"] for r in records if r["label"] == label]
    return sum(selected) / max(len(selected), 1)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for variant in payload["variants"]:
        summary = variant["summary"]
        records = variant["records"]
        rows.append(
            {
                "setting": variant["setting"],
                "num_records": summary["num_records"],
                "num_harmful": sum(r["label"] == "harmful" for r in records),
                "num_benign": sum(r["label"] == "benign" for r in records),
                "blocked_harmful": sum(
                    r["label"] == "harmful" and r["decision"] == "block"
                    for r in records
                ),
                "allowed_benign": sum(
                    r["label"] == "benign" and r["decision"] == "allow"
                    for r in records
                ),
                "attack_block_rate": summary["attack_block_rate"],
                "benign_allow_rate": summary["benign_allow_rate"],
                "false_positive_rate": summary["false_positive_rate"],
                "mean_prompt_tokens": summary["mean_prompt_tokens"],
                "mean_prompt_tokens_harmful": mean_tokens(records, "harmful"),
                "mean_prompt_tokens_benign": mean_tokens(records, "benign"),
                "mean_calibrated_harmful": summary["mean_calibrated_harmful"],
                "mean_calibrated_benign": summary["mean_calibrated_benign"],
                "calibrated_score_gap": summary["calibrated_score_gap"],
                "brier_score": summary["brier_score"],
                "ranking_auc": summary["ranking_auc"],
                "state_counts": str(summary["state_counts"]),
                "state_counts_harmful": str(counts(records, "harmful")),
                "state_counts_benign": str(counts(records, "benign")),
                "scorer": payload.get("scorer", ""),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
