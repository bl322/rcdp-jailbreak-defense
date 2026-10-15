from __future__ import annotations

import json
from pathlib import Path

from .types import Example, Label, Turn


def _parse_label(value: str) -> Label:
    norm = value.strip().lower()
    if norm not in {"harmful", "benign"}:
        raise ValueError(f"Unsupported label: {value}")
    return Label(norm)


def load_examples(path: str | Path) -> list[Example]:
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Only JSONL datasets are supported by the rebuilt scaffold.")

    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            turns = [Turn(role=t["role"], text=t["text"]) for t in payload["turns"]]
            examples.append(
                Example(
                    example_id=str(payload.get("example_id", f"ex-{line_no}")),
                    benchmark=payload.get("benchmark", "unknown"),
                    label=_parse_label(payload["label"]),
                    turns=turns,
                    metadata=payload.get("metadata", {}),
                )
            )
    return examples
