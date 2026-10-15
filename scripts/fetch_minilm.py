"""Fetch all-MiniLM-L6-v2 through the reachable mirror into a local folder."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
BASE = "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
]

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            log(f"cached  {rel} ({dest.stat().st_size} bytes)")
            continue
        url = BASE + rel
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001
            log(f"FAIL    {rel}: {type(exc).__name__} {exc}")
            continue
        if not data or data[:1] in {b"<", b"{"} is False and len(data) < 64:
            log(f"SUSPECT {rel}: {len(data)} bytes, head={data[:40]!r}")
        dest.write_bytes(data)
        log(f"ok      {rel} ({len(data)} bytes)")

    # sanity: config.json must parse
    cfg = OUT / "config.json"
    if cfg.exists():
        try:
            parsed = json.loads(cfg.read_text(encoding="utf-8"))
            log(f"config.json parses OK: arch={parsed.get('architectures')}")
        except Exception as exc:  # noqa: BLE001
            log(f"config.json BROKEN: {exc}; head={cfg.read_text('utf-8', 'ignore')[:120]!r}")

    (Path(__file__).resolve().parent / "_fetch.log").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
