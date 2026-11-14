"""Re-check the refusal label of already-saved e2e generations with the LLM refusal judge.

R3-2: the harness's "useful answer" is lexically defined (delivered, non-empty, no refusal
phrase in the first 200 chars). This script re-labels the SAME saved texts with the
llm-local refusal detector (Llama-3.1-8B-Instruct, 16 tokens, greedy) and reports how often
the two rules agree -- no generation, no judge call, no gate call is repeated.

Usage (on the experiment server, venv sjudge):
  python3 scripts/recheck_refusal_llm_local.py --input results/r1/e2e/bl_no-defense_qwen3next.json \
      --model ~/models/Meta-Llama-3.1-8B-Instruct --device cuda --output results/r1/e2e/recheck_nodef.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import e2e_common as E  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="Local LLM used by the llm-local refusal detector")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    data = json.load(open(a.input, encoding="utf-8"))
    recs = data["records"]
    if a.limit:
        recs = recs[: a.limit]

    gen = E.LocalHFGenerator(a.model, a.device, a.dtype, 0.0, 1.0, 0, 16, role="refusal-judge")
    det = E.LLMRefusalDetector(gen)

    rows = []
    for i, r in enumerate(recs, 1):
        req = r.get("request") or ""
        raw_gen = r.get("response") or r.get("generation")
        if isinstance(raw_gen, dict):          # `generation` is a metadata dict in this schema
            raw_gen = raw_gen.get("text")
        delivered = r.get("delivered_response")
        if isinstance(delivered, dict):
            delivered = delivered.get("text")
        lex = (r.get("refusal") or {}).get("class")
        llm_raw = det.classify(req, raw_gen) if raw_gen not in (None, "") else None
        llm_del = det.classify(req, delivered) if delivered not in (None, "") else None
        rows.append({
            "example_id": r.get("example_id"),
            "label": r.get("label"),
            "lexical_class": lex,
            "llm_class_raw_generation": (llm_raw or {}).get("class"),
            "llm_class_delivered": (llm_del or {}).get("class"),
            "pre_generation_block": bool(r.get("pre_generation_block")),
            "delivered_nonempty": bool(delivered and delivered.strip()),
        })
        if i % 25 == 0:
            print("  %d/%d" % (i, len(recs)), flush=True)

    def summarise(subset):
        n = len(subset)
        agree_raw = sum(1 for x in subset if x["lexical_class"] == x["llm_class_raw_generation"])
        agree_del = sum(1 for x in subset if x["lexical_class"] == x["llm_class_delivered"])
        # the operative distinction for `useful_answer` is "non-refusal (NR)" vs anything else
        agree_nr = sum(1 for x in subset
                       if (x["lexical_class"] == "NR") == (x["llm_class_delivered"] == "NR"))
        pairs = Counter((x["lexical_class"], x["llm_class_delivered"]) for x in subset)
        useful_lex = sum(1 for x in subset if x["delivered_nonempty"] and x["lexical_class"] == "NR")
        useful_llm = sum(1 for x in subset if x["delivered_nonempty"] and x["llm_class_delivered"] == "NR")
        return {
            "n": n,
            "agreement_exact_class_raw_generation": "%d/%d" % (agree_raw, n),
            "agreement_exact_class_delivered": "%d/%d" % (agree_del, n),
            "agreement_NR_vs_not_NR": "%d/%d" % (agree_nr, n),
            "confusion_lexical_vs_llm_delivered": {"%s->%s" % k: v for k, v in sorted(pairs.items(), key=lambda kv: str(kv[0]))},
            "non_refusal_answers_lexical": "%d/%d" % (useful_lex, n),
            "non_refusal_answers_llm": "%d/%d" % (useful_llm, n),
        }

    benign = [x for x in rows if x["label"] == "benign"]
    harmful = [x for x in rows if x["label"] == "harmful"]
    summary = {
        "input": a.input,
        "detector": det.describe(),
        "all": summarise(rows),
        "benign": summarise(benign),
        "harmful": summarise(harmful),
    }
    json.dump({"summary": summary, "rows": rows}, open(a.output, "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
