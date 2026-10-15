from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT_COLUMNS = ("prompt", "text", "sentence", "query", "input")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the official XSTest prompt file into the RCDP JSONL format."
    )
    parser.add_argument("--source", required=True, help="Official XSTest CSV or JSONL file.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--type-column", default="type")
    parser.add_argument(
        "--benign-values",
        default="safe,benign",
        help="Comma-separated values in --type-column retained as benign examples.",
    )
    return parser.parse_args()


def read_rows(source: Path) -> list[dict[str, str]]:
    if source.suffix.lower() == ".jsonl":
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_prompt_column(rows: list[dict[str, str]], requested: str) -> str:
    if rows and requested in rows[0]:
        return requested
    if rows:
        for candidate in PROMPT_COLUMNS:
            if candidate in rows[0]:
                return candidate
    raise ValueError(
        f"Could not find prompt column '{requested}'. "
        f"Available columns: {sorted(rows[0]) if rows else []}"
    )


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    rows = read_rows(source)
    prompt_column = resolve_prompt_column(rows, args.prompt_column)
    benign_values = {value.strip().lower() for value in args.benign_values.split(",")}

    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row_type = str(row.get(args.type_column, "")).strip().lower()
            if row_type not in benign_values:
                continue
            prompt = str(row.get(prompt_column, "")).strip()
            if not prompt:
                continue

            payload = {
                "example_id": f"xstest-safe-{index}",
                "benchmark": "xstest",
                "label": "benign",
                "turns": [{"role": "user", "text": prompt}],
                "metadata": {
                    "source_row": index,
                    "xstest_type": row_type,
                    "focus": row.get("focus", ""),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1

    if kept == 0:
        raise ValueError(
            "No benign examples were retained. Check --type-column and --benign-values "
            "against the downloaded XSTest release."
        )
    print(f"Wrote {kept} benign XSTest examples to {output}")


if __name__ == "__main__":
    main()
