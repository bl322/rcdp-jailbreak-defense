"""Seed-clustered bootstrap on a scripts/run_multiturn_cases.py output (no model needed).

The runner already stores this analysis (settings.<s>.bootstrap_evaluation);
use this script to recompute it with other variants, contrasts, side or B.
b5_cluster_bootstrap.py from the audit reads the runner's
<output>.<setting>.b5.json export directly and gives identical session-level
numbers for its six variants.

    python scripts/multiturn_cluster_bootstrap.py results/r1/multiturn_distilbart.json out.json \
        --setting agnostic --variants 4-state-downgrade,4-state+memory \
        --contrast "4-state+memory:4-state-downgrade"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard import multiturn as MT  # noqa: E402
from rcdp_guard.multiturn_bootstrap import B_DEFAULT, SEED_DEFAULT, cluster_bootstrap, format_table  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--setting", default="agnostic")
    p.add_argument("--side", default="evaluation", choices=["evaluation", "selection"])
    p.add_argument("--variants", default="all")
    p.add_argument("--contrast", action="append", default=[], help="'a:b' for a - b (repeatable); default: runner contrasts")
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--rng-seed", type=int, default=SEED_DEFAULT)
    args = p.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    variants = payload["settings"][args.setting]["variants"]
    names = list(variants) if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    first = variants[names[0]]
    sessions = [{k: r[k] for k in ("example_id", "cluster_id", "label", "side")} for r in first["sessions"]]
    outs = OrderedDict()
    for v in names:
        rows = variants[v]["sessions"]
        if [r["example_id"] for r in rows] != [s["example_id"] for s in sessions]:
            raise SystemExit(f"{v}: session order differs")
        outs[v] = {i: {"blocked": r["blocked"]} for i, r in enumerate(rows)}
    contrasts = [tuple(c.split(":", 1)) for c in args.contrast] or MT.CONTRASTS
    res = cluster_bootstrap(MT.outcome_columns(sessions, outs, side=args.side), contrasts, B=args.B, seed=args.rng_seed)
    res["input"] = args.input
    res["setting"] = args.setting
    res["side"] = args.side
    Path(args.output).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(format_table(res))


if __name__ == "__main__":
    main()
