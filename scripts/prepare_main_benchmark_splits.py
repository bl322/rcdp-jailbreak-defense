from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper's 400-example pool and deterministic calibration/held-out JSONL splits."
    )
    parser.add_argument("--benign-csv", required=True)
    parser.add_argument("--harmful-csv", required=True)
    parser.add_argument("--harmbench-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def stable_key(example_id: str) -> tuple[int, str]:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16), example_id


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jb_payload(row: dict[str, str], *, label: str, prefix: str) -> dict:
    idx = str(row["Index"]).strip()
    return {
        "example_id": f"{prefix}-{idx}",
        "benchmark": "JailbreakBench",
        "label": label,
        "turns": [{"role": "user", "text": row["Goal"].strip()}],
        "metadata": {
            "target": row.get("Target", "").strip(),
            "behavior": row.get("Behavior", "").strip(),
            "category": row.get("Category", "").strip(),
            "source": row.get("Source", "").strip(),
        },
    }


def hb_payload(row: dict[str, str]) -> dict:
    behavior_id = row["BehaviorID"].strip()
    return {
        "example_id": f"hb-{behavior_id}",
        "benchmark": "HarmBench",
        "label": "harmful",
        "turns": [{"role": "user", "text": row["Behavior"].strip()}],
        "metadata": {
            "functional_category": row.get("FunctionalCategory", "").strip(),
            "semantic_category": row.get("SemanticCategory", "").strip(),
            "tags": row.get("Tags", "").strip(),
            "context": row.get("ContextString", "").strip(),
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    benign_rows = load_csv(Path(args.benign_csv))[:100]
    harmful_rows = load_csv(Path(args.harmful_csv))[:100]
    harmbench_rows = load_csv(Path(args.harmbench_csv))[:200]

    benign = [jb_payload(row, label="benign", prefix="jb-b") for row in benign_rows]
    jb_harmful = [jb_payload(row, label="harmful", prefix="jb-h") for row in harmful_rows]
    hb_harmful = [hb_payload(row) for row in harmbench_rows]

    harmful = jb_harmful + hb_harmful
    harmful_sorted = sorted(harmful, key=lambda row: stable_key(row["example_id"]))
    benign_sorted = sorted(benign, key=lambda row: stable_key(row["example_id"]))

    heldout_harmful = harmful_sorted[:149]
    calibration_harmful = harmful_sorted[149:]
    heldout_benign = benign_sorted[:55]
    calibration_benign = benign_sorted[55:]

    full_pool = benign + harmful
    calibration = calibration_harmful + calibration_benign
    heldout = heldout_harmful + heldout_benign

    write_jsonl(output_dir / "full_pool_400.jsonl", full_pool)
    write_jsonl(output_dir / "full_pool_calibration.jsonl", calibration)
    write_jsonl(output_dir / "heldout_204.jsonl", heldout)
    write_jsonl(output_dir / "heldout_harmful_149.jsonl", heldout_harmful)

    summary = {
        "full_pool": len(full_pool),
        "calibration": len(calibration),
        "heldout": len(heldout),
        "heldout_harmful": len(heldout_harmful),
        "heldout_benign": len(heldout_benign),
        "calibration_harmful": len(calibration_harmful),
        "calibration_benign": len(calibration_benign),
        "selection": {
            "JailbreakBench_benign": 100,
            "JailbreakBench_harmful": 100,
            "HarmBench_harmful": 200,
        },
        "split_rule": "stable SHA-256 ordering within label groups",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
