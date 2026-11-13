"""Compare two end-to-end outputs item by item (revision R1 reproduction check).

Typical uses
  * Does the R1 harness, run with the submitted settings, reproduce a saved submitted run?
        python scripts/e2e_compare_saved.py \
            --new results/r1/e2e_repro_llama31_aware_legacy.json \
            --ref results/e2e_llama31_8b_independent_llamaguard_heldout204.json
  * Is a hosted endpoint deterministic at temperature 0 (two identical R1 runs)?
        python scripts/e2e_compare_saved.py --new run_a.json --ref run_b.json

Either file may be a submitted harness output (top-level 'records' with 'response_judgment',
'output_guarded', optional 'response') or an R1 output (schema rcdp-e2e-r1/*, also written by
scripts/baselines_r1/run_baseline_e2e.py). Compared per example_id: routing state and decision,
whether the model was called, output-check applied / blocked, the delivered-text verdict of the
primary judge, and the delivered response text (exact match and common-prefix length); when both
files are R1 outputs also the refusal class and useful-answer flag (e.g. lexical vs LLM refusal
detector on the same greedy generations). Nothing is recomputed; both files are only read.
Python 3.8, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _b(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == "true") or v == 1


def _is_r1(data):
    return str(data.get("schema", "")).startswith("rcdp-e2e-r1")


def normalise(data):
    out = {}
    r1 = _is_r1(data)
    for rec in data.get("records", []):
        if "example_id" not in rec or "error" in rec:
            continue
        gen = _b(rec.get("generated"))
        if r1:
            og = rec.get("output_guard") or {}
            text = rec.get("delivered_response") if gen else None
            outcome = rec.get("outcome") or {}
            row = {"state": rec.get("state"), "decision": rec.get("controller_decision"), "generated": gen,
                   "og_checked": bool(og.get("checked")), "og_blocked": bool(og.get("blocked")),
                   "delivered_unsafe": rec.get("response_judgment") == "unsafe", "text": text,
                   "raw_text": rec.get("response"), "refusal_class": outcome.get("refusal_class"),
                   "useful_answer": outcome.get("useful_answer")}
        else:
            text = rec.get("response") if gen else None
            row = {"state": rec.get("state"), "decision": rec.get("controller_decision"), "generated": gen,
                   "og_checked": _b(rec.get("output_guard_checked")), "og_blocked": _b(rec.get("output_guarded")),
                   "delivered_unsafe": rec.get("response_judgment") == "unsafe", "text": text, "raw_text": None}
        row["label"] = rec.get("label")
        out[str(rec["example_id"])] = row
    return out


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2.0


def counts(rows):
    h = [r for r in rows.values() if r["label"] == "harmful"]
    b = [r for r in rows.values() if r["label"] == "benign"]
    return {
        "harmful_n": len(h), "benign_n": len(b),
        "harmful_blocked_pre_generation": sum(r["decision"] == "block" for r in h),
        "benign_allowed": sum(r["decision"] == "allow" for r in b),
        "harmful_generated": sum(r["generated"] for r in h),
        "output_check_applied": sum(r["og_checked"] for r in rows.values()),
        "output_check_blocked": sum(r["og_blocked"] for r in rows.values()),
        "harmful_delivered_unsafe": sum(r["delivered_unsafe"] for r in h),
        "benign_delivered_unsafe": sum(r["delivered_unsafe"] for r in b),
    }


def compare(new, ref, max_examples=10):
    a, b = normalise(new), normalise(ref)
    ids = sorted(set(a) & set(b))
    fields = ("state", "decision", "generated", "og_checked", "og_blocked", "delivered_unsafe")
    if _is_r1(new) and _is_r1(ref):  # e.g. lexical vs LLM refusal detector on the same greedy texts
        fields = fields + ("refusal_class", "useful_answer")
    agree = {f: sum(a[i][f] == b[i][f] for i in ids) for f in fields}
    diffs = {f: [{"example_id": i, "new": a[i][f], "ref": b[i][f]} for i in ids if a[i][f] != b[i][f]][:max_examples]
             for f in fields}
    both_text = [i for i in ids if a[i]["text"] is not None and b[i]["text"] is not None]
    same_text = [i for i in both_text if a[i]["text"].strip() == b[i]["text"].strip()]
    prefixes = [common_prefix(a[i]["text"].strip(), b[i]["text"].strip()) for i in both_text]
    text_diffs = [{"example_id": i, "common_prefix_chars": common_prefix(a[i]["text"].strip(), b[i]["text"].strip()),
                   "new": a[i]["text"][:160], "ref": b[i]["text"][:160]}
                  for i in both_text if i not in same_text][:max_examples]
    routing_same = agree["state"] == len(ids) and agree["decision"] == len(ids)
    return {
        "new_schema": new.get("schema", "submitted"), "ref_schema": ref.get("schema", "submitted"),
        "ids": {"new": len(a), "ref": len(b), "common": len(ids),
                "only_new": sorted(set(a) - set(b))[:max_examples], "only_ref": sorted(set(b) - set(a))[:max_examples]},
        "agreement": {f: "%d/%d" % (agree[f], len(ids)) for f in fields},
        "disagreements": {f: d for f, d in diffs.items() if d},
        "texts": {
            "compared": len(both_text),
            "identical": len(same_text),
            "median_common_prefix_chars": _median(prefixes),
            "note": "ref has no saved response text" if not both_text and any(a[i]["text"] for i in ids) else None,
            "examples_of_differences": text_diffs,
        },
        "counts": {"new": counts(a), "ref": counts(b)},
        "verdict": {
            "routing_identical": routing_same,
            "output_check_identical": agree["og_checked"] == len(ids) and agree["og_blocked"] == len(ids),
            "texts_identical": bool(both_text) and len(same_text) == len(both_text),
            "judge_identical_on_delivered_text": agree["delivered_unsafe"] == len(ids),
        },
        "caveats": [
            "A submitted file stores the DELIVERED text only (the controller refusal when its output guard fired); "
            "the R1 comparison therefore uses delivered_response on both sides.",
            "Greedy decoding on GPU is deterministic only for identical hardware, library versions and prompt "
            "tokens: the submitted local runs tokenized with a duplicated BOS (use --legacy-double-bos to match).",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--new", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max-examples", type=int, default=10)
    args = p.parse_args(argv)
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))
    ref = json.loads(Path(args.ref).read_text(encoding="utf-8"))
    report = compare(new, ref, args.max_examples)
    report["files"] = {"new": args.new, "ref": args.ref}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
