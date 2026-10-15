from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True, choices=["harmful", "benign"])
    parser.add_argument("--benchmark", default="rpdp")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    with output.open("w", encoding="utf-8") as fh:
        for idx, line in enumerate(lines[1:], start=1):
            if not line.strip():
                continue
            cells = [c.strip().strip('"') for c in line.split(",") if c.strip()]
            turns = [{"role": "user", "text": cell} for cell in cells[:5] if cell]
            if not turns:
                continue
            payload = {
                "example_id": f"rpdp-{idx}",
                "benchmark": args.benchmark,
                "label": args.label,
                "turns": turns,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
