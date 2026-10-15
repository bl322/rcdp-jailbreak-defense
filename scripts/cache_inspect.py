"""How much of the corpus is already scored in the existing LlamaGuard caches?"""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from rcdp_guard.datasets import load_examples  # noqa: E402

LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LINES.append(msg)


CACHE_DIR = EXPERIMENTS / "cache"
DATA = EXPERIMENTS / "data"

caches = {}
for p in CACHE_DIR.glob("llamaguard_*.json"):
    try:
        caches[p.name] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"{p.name}: unreadable ({exc})")

for name, entries in caches.items():
    log(f"{name}: {len(entries)} entries")
    if entries:
        k = next(iter(entries))
        log(f"   sample key: {k[:220]}")
        log(f"   sample val: {entries[k]}")

# Build a text -> score index from every cache, ignoring metadata differences.
index: dict[str, float] = {}
for entries in caches.values():
    for k, v in entries.items():
        try:
            payload = json.loads(k)
        except Exception:  # noqa: BLE001
            continue
        text = payload.get("text")
        if isinstance(text, str) and "raw_score" in v:
            index[text] = float(v["raw_score"])
log(f"\ntext->score index size: {len(index)}")

SETS = {
    "harmful_149": DATA / "paper_splits" / "heldout_harmful_149.jsonl",
    "heldout204_benign": DATA / "paper_splits" / "heldout_204.jsonl",
    "xstest_safe_450": DATA / "external" / "xstest_safe_450.jsonl",
    "orbench_hard_250": DATA / "external" / "orbench_hard_benign_250.jsonl",
    "benign_external_500": DATA / "external" / "benign_external_500.jsonl",
}
for name, path in SETS.items():
    exs = load_examples(str(path))
    if name == "heldout204_benign":
        exs = [e for e in exs if e.label == "benign"]
    hit = sum(1 for e in exs if e.turns[-1].text in index)
    log(f"{name}: {hit}/{len(exs)} already cached")

tap = json.loads((HERE / "results" / "tap_heldout149.json").read_text(encoding="utf-8"))
cands = [c["prompt"] for r in tap["runs"] for c in r["attempted_candidates"]]
hit = sum(1 for t in cands if t in index)
log(f"tap_candidates: {hit}/{len(cands)} already cached")

(HERE / "_cacheinspect.txt").write_text("\n".join(LINES), encoding="utf-8")
