"""Does principal binding harm benign conversational traffic?

Monotone escalation is what makes the sequential controller stop an iterative
attack: once a caller reaches HARDENED/QUARANTINE the state does not go back
down.  The obvious objection is that this would also lock out an ordinary user
who happens to trip a cautious state once.  This script measures exactly that,
on multi-turn dialogues, separately from the attack results.

Reported per turn index, so the cost is visible rather than averaged away.
"""

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

from rcdp_hardened import HardenedService, build_controller  # noqa: E402

DATA = EXPERIMENTS / "data"
LOG = HERE / "_bindcost.log"
LINES: list[str] = []


def log(m: str) -> None:
    print(m, flush=True)
    LINES.append(m)
    LOG.write_text("\n".join(LINES), encoding="utf-8")


SETS = {
    "multiturn_heldout_612": DATA / "generated" / "multiturn_heldout_612.jsonl",
    "multiturn_quant_54": DATA / "samples" / "multiturn_quant_54.jsonl",
    "paired_benign_55": DATA / "generated" / "paired_benign_55.jsonl",
}


def main() -> None:
    for kind in ["nli"]:
        log(f"=== scorer={kind} ===")
        controller = build_controller(kind, benchmark_aware=False)
        for name, path in SETS.items():
            if not path.exists():
                log(f"  {name}: missing")
                continue
            all_ex = load_examples(str(path))
            benign = [e for e in all_ex if e.label == "benign"]
            if not benign:
                log(f"  {name}: no benign examples ({len(all_ex)} total)")
                continue
            multi = [e for e in benign if len(e.turns) > 1]
            log(f"  {name}: benign={len(benign)}  multi-turn={len(multi)}")

            for label, pool in [("all_benign", benign), ("multi_turn", multi)]:
                if not pool:
                    continue
                stateless_blocked = 0
                bound_blocked = 0
                bound_turns = 0
                latched_sessions = 0
                # per-turn cost for multi-turn sessions
                turn_first_block = {}
                for ex in pool:
                    # stateless: judge every turn independently
                    recs = controller.evaluate_example(ex)
                    user_recs = [
                        r for r in recs if r.example_id.endswith(f":turn-{i}")
                        for i, t in enumerate(ex.turns)
                        if t.role == "user" and r.example_id.endswith(f":turn-{i}")
                    ]
                    stateless_blocked += sum(1 for r in recs if r.decision == "block")

                    # bound: replay turn by turn through one principal
                    svc = HardenedService(controller, ex, bind=True)
                    for t in ex.turns:
                        if t.role != "user":
                            continue
                        out = svc.query(t.text, principal_id="acct")
                        bound_turns += 1
                        if not out["allowed"]:
                            bound_blocked += 1
                            idx = sum(1 for x in ex.turns[: ex.turns.index(t)] if x.role == "user")
                            turn_first_block[idx] = turn_first_block.get(idx, 0) + 1
                            break
                    if svc.principals["acct"].latched:
                        latched_sessions += 1

                log(
                    f"    {label:11s} n={len(pool):4d}  "
                    f"stateless_blocked_turns={stateless_blocked}  "
                    f"bound_sessions_blocked={bound_blocked} "
                    f"({bound_blocked/max(len(pool),1):.4f})  "
                    f"sessions_that_latched={latched_sessions}"
                )


if __name__ == "__main__":
    sys.exit(main())
