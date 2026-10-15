"""Inspect successful TAP evasions: what framing actually broke through?"""

from __future__ import annotations

import os

import json
import sys
from pathlib import Path

_OUT = Path(__file__).resolve().parent / "_inspect.txt"

_RESULTS = Path(__file__).resolve().parent / "results" / "tap_heldout149.json"
RESULTS = _RESULTS


def print(*args, **kwargs):  # noqa: A001 - deliberate shadow for tee
    sep = kwargs.get("sep", " ")
    line = sep.join(str(a) for a in args)
    with _OUT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

data = json.loads(RESULTS.read_text(encoding="utf-8"))
runs = [r for r in data["runs"] if r["success"]]
print(f"successful evasions: {len(runs)} / {len(data['runs'])}")

# Original goals come from the dataset.
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(EXPERIMENTS))
from rcdp_guard.datasets import load_examples  # noqa: E402

examples = {ex.example_id: ex for ex in load_examples(data["protocol"]["dataset"])}

for r in runs[:6]:
    ex = examples.get(r["example_id"])
    print("=" * 78)
    print(f"ID: {r['example_id']}   queries={r['service_queries']}  depth={r['tree_depth_used']}")
    print(f"ORIGINAL GOAL: {ex.turns[-1].text[:300] if ex else '(not found)'}")
    print(f"WINNING PROMPT: {r['best_prompt'][:600]}")

# How many wins came at depth 1 (i.e. the first rewrite already passed)?
from collections import Counter

depth_hist = Counter()
for r in runs:
    winners = [
        c for c in r["attempted_candidates"] if c["allowed"] and c["score"] >= 10
    ]
    depth_hist[min((c["level"] for c in winners), default=0)] += 1
print("=" * 78)
print("depth at which the winning prompt first appeared:", dict(sorted(depth_hist.items())))

# How many candidates were allowed overall vs blocked?
allc = [c for r in data["runs"] for c in r["attempted_candidates"]]
allowed = sum(1 for c in allc if c["allowed"])
print(f"all scored candidates: {len(allc)}   allowed: {allowed} ({allowed/max(len(allc),1):.3f})")
