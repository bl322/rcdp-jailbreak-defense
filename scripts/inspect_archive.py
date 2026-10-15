import os
"""Inspect the archived adaptive experiment to find where the 105 denominator came from."""
import json
from pathlib import Path

EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
arch = EXPERIMENTS / "results" / "adaptive_heldout149_distilbart_api.json"
d = json.loads(arch.read_text(encoding="utf-8"))

def summarize_block(name, block):
    if not isinstance(block, dict):
        print(f"  {name}: <{type(block).__name__}>")
        return
    keys = list(block.keys())
    info = {}
    for k in ("protocol", "config", "controller", "pre_attack", "blocked_before",
             "blocked_before_adaptation", "n_blocked", "n_harmful", "n_attacked",
             "evaded", "evasion_rate", "attack_success", "per_example", "results"):
        if k in block:
            v = block[k]
            if isinstance(v, list):
                info[k] = f"list[{len(v)}]"
            elif isinstance(v, dict):
                info[k] = f"dict[{len(v)}] keys={list(v.keys())[:8]}"
            else:
                info[k] = v
    print(f"  [{name}] keys={keys}")
    for k, v in info.items():
        print(f"      {k} = {v}")
    # if per_example present, count blocked
    pe = block.get("per_example")
    if isinstance(pe, list):
        nb = sum(1 for r in pe if r.get("blocked"))
        print(f"      per_example n={len(pe)} blocked={nb}")
    # nested blocks
    for k, v in block.items():
        if isinstance(v, dict) and k not in info:
            # print small dicts
            if len(v) <= 12:
                print(f"      sub[{k}] = {v}")

print("=== protocol ===")
print(json.dumps(d.get("protocol", {}), ensure_ascii=False, indent=2)[:1500])
for name in ("rewrite_heuristic", "multiturn_state_manipulation_heuristic",
             "rewrite_api", "multiturn_state_manipulation_api"):
    print(f"\n=== {name} ===")
    summarize_block(name, d.get(name))
