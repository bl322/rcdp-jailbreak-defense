"""Attach exact and bootstrap 95% intervals to an evaluation JSON with per-item records.

Revision R1 fix (ledger X-15): the submitted version imported
``bootstrap_binomial_ci`` from ``rcdp_guard.evaluation``, which does not
exist, so the script could not run. The intervals are now computed by the
standard-library helpers in ``scripts/stats_r1.py`` (no scipy, no numpy):

* exact Clopper-Pearson interval for each k/n;
* seeded percentile bootstrap (10,000 resamples by default; seeds 20260903,
  20260904 and 20260905 as in the submitted script);
* the percentile-bootstrap endpoints in the limit B -> infinity, which for a
  proportion follow exactly from Binomial(n, k/n). A finite-B run can land one
  count away when a CDF value is close to 0.025; the output flags this.

Records are read from ``--records-key`` (a dotted path; default ``records``; a
step into a list is an index or ``key=value[&key=value]``, e.g.
``variants.variant=rcdp-4state&setting=agnostic.eval.heldout.records`` for
scripts/run_r1_main.py output, ``records.heldout`` for the scorer-only baselines).
The decision field is ``controller_decision`` when present (end-to-end runs),
else ``decision``. Records of multi-turn examples (``<id>:turn-<k>``) are
collapsed to one outcome per example (blocked if any turn is blocked).

Example:
    python scripts/add_confidence_intervals.py \
        --input results/e2e_qwen3next_independent_llamaguard_heldout204.json \
        --output results/r1/ci/e2e_qwen3next_ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stats_r1 import proportion_summary, resolve  # noqa: E402

SEEDS = {"attack_block_rate": 20260903, "benign_allow_rate": 20260904, "false_positive_rate": 20260905}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach Clopper-Pearson and percentile-bootstrap 95% intervals to an evaluation JSON."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-key", default="records",
                        help="dotted path to the per-item record list (list indices allowed)")
    parser.add_argument("--decision-field", default="auto",
                        help="auto (controller_decision, else decision) or an explicit field name")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--conf", type=float, default=0.95)
    return parser.parse_args(argv)


def per_example_outcomes(records, decision_field: str):
    """Return [(example_id, label, blocked)], collapsing per-turn records to examples."""
    out: dict = {}
    order = []
    for row in records:
        field = decision_field
        if field == "auto":
            field = "controller_decision" if "controller_decision" in row else "decision"
        decision = row[field]
        if decision not in ("block", "allow"):
            raise ValueError(f"unexpected decision {decision!r} in field {field!r}")
        base = str(row.get("example_id", len(order))).split(":turn-")[0]
        if base not in out:
            out[base] = [row["label"], False]
            order.append(base)
        elif out[base][0] != row["label"]:
            raise ValueError(f"example {base} has records with different labels")
        out[base][1] = out[base][1] or decision == "block"
    return [(eid, out[eid][0], out[eid][1]) for eid in order], len(records) != len(order)


def main(argv=None) -> dict:
    args = parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = resolve(payload, args.records_key)
    if not records:
        raise ValueError(f"No per-item records at '{args.records_key}' in {args.input}.")
    items, collapsed = per_example_outcomes(records, args.decision_field)
    harmful = [blocked for _, label, blocked in items if label == "harmful"]
    benign = [blocked for _, label, blocked in items if label == "benign"]

    def summary(outcomes, name):
        res = proportion_summary(outcomes, B=args.samples, seed=SEEDS[name], conf=args.conf)
        if res["n"] and res["bootstrap_percentile"] is not None:
            res["monte_carlo_matches_limit"] = all(
                abs(a - b) < 1e-12 for a, b in zip(res["bootstrap_percentile"], res["bootstrap_percentile_limit"]))
        return res

    ci = {
        "method": "exact Clopper-Pearson; percentile bootstrap over example-level outcomes "
                  "(seeded Monte Carlo and its B -> infinity limit)",
        "confidence_level": args.conf,
        "records_key": args.records_key,
        "decision_field": args.decision_field,
        "per_turn_records_collapsed": collapsed,
        "attack_block_rate": summary(harmful, "attack_block_rate"),
        "benign_allow_rate": summary([not b for b in benign], "benign_allow_rate"),
        "false_positive_rate": summary(benign, "false_positive_rate"),
        "note": ("Intervals are computed over example-level records; they quantify sampling uncertainty "
                 "within this fixed evaluation set and do not replace external validation."),
    }
    payload["confidence_intervals"] = ci
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("attack_block_rate", "benign_allow_rate", "false_positive_rate"):
        r = ci[name]
        if not r["n"]:
            print(f"{name:20s} n=0")
            continue
        cp, bs, lim = r["clopper_pearson"], r["bootstrap_percentile"], r["bootstrap_percentile_limit"]
        boot = (f"boot(B={r['bootstrap_B']}, seed={r['bootstrap_seed']}) [{bs[0]:.4f}, {bs[1]:.4f}]  "
                + ("" if r["monte_carlo_matches_limit"] else "(MC != limit)  ")) if bs else ""
        print(f"{name:20s} {r['k']}/{r['n']} = {r['estimate']:.4f}  CP [{cp[0]:.4f}, {cp[1]:.4f}]  "
              f"{boot}limit [{lim[0]:.4f}, {lim[1]:.4f}]")
    print("wrote", output)
    return ci


if __name__ == "__main__":
    main()
