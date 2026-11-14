"""Offline second-judge re-scoring on saved generations (revision R1, R3-3 / editor item 8).

Reads a saved end-to-end JSON (no regeneration), re-judges every *delivered* generation
with a different judge model, and writes a sidecar JSON with

  * per-record judge_secondary_cross {verdict, raw}          (same prompt as the harness)
  * asr_secondary / asr_both_judges / benign_flagged_secondary recomputed over the same denominators
  * the ids where the two judges disagree, marked per harmful / benign

The harness prompt and label rule are imported from scripts/e2e_common.py so the re-scoring
is the same measurement, not a second definition. Only records that were generated and not
blocked by the output check are judged: for every other record the harness's own
harmful_completion_secondary is False by construction (delivered_gen is False).

Usage:
  python3 scripts/rescore_second_judge.py --input results/r1/e2e/loc_llama31_no-defense.json \
      --judge-model ~/models/Mistral-7B-Instruct-v0.3 --device cuda \
      --output results/r1/e2e/sj_mistral_on_llama31_no-defense.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import e2e_common as E  # noqa: E402


def delivered(r):
    return bool(r.get("generated")) and not bool((r.get("output_guard") or {}).get("blocked"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge-model", default=None, help="local judge model directory")
    ap.add_argument("--judge-api-model", default=None, help="API judge model name (OpenAI-compatible endpoint)")
    ap.add_argument("--judge-api-base-url", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--primary-name", default=None, help="label for the existing judge_primary column")
    args = ap.parse_args(argv)

    blob = json.load(open(args.input, encoding="utf-8"))
    recs = blob["records"]
    if args.limit:
        recs = recs[: args.limit]

    if args.judge_api_model:
        key = os.environ.get("OPENAI_API_KEY")
        if not key or not args.judge_api_base_url:
            raise SystemExit("--judge-api-model needs --judge-api-base-url and OPENAI_API_KEY")
        gen = E.OpenAIChatGenerator(args.judge_api_base_url, args.judge_api_model, key, 0.0, 1.0,
                                    args.max_new_tokens, {"enable_thinking": False}, False, 120.0, 3,
                                    role="second-judge-cross")
    else:
        if not args.judge_model:
            raise SystemExit("either --judge-model or --judge-api-model is required")
        gen = E.LocalHFGenerator(args.judge_model, args.device, args.dtype, 0.0, 1.0, 0,
                                 args.max_new_tokens, role="second-judge-cross")
    judge = E.LLMResponseJudge(gen, role="secondary")

    t0 = time.time()
    out = []
    for i, r in enumerate(recs, 1):
        j1 = (r.get("judge_primary") or {}).get("verdict")
        row = {"example_id": r["example_id"], "label": r["label"],
               "delivered": delivered(r), "primary": j1, "secondary": None, "raw": None}
        if row["delivered"]:
            res = judge.judge(r.get("request") or "", r.get("response") or "")
            row["secondary"] = res["verdict"]
            row["raw"] = res["raw"]
        out.append(row)
        if i % 25 == 0 or i == len(recs):
            print("  %d/%d  %.1fs" % (i, len(recs), time.time() - t0), flush=True)

    def tally(rows):
        n = len(rows)
        d = [x for x in rows if x["delivered"]]
        dis = [x for x in d if x["primary"] != x["secondary"]]
        unsafe_s = [x for x in d if x["secondary"] == "unsafe"]
        unknown_s = [x for x in d if x["secondary"] == "unknown"]
        return {
            "n": n,
            "delivered": len(d),
            "secondary_unsafe": len(unsafe_s),
            "secondary_unknown": len(unknown_s),
            "primary_unsafe": sum(1 for x in d if x["primary"] == "unsafe"),
            "both_unsafe": sum(1 for x in d if x["primary"] == "unsafe" and x["secondary"] == "unsafe"),
            "disagreements": len(dis),
            "disagreement_ids": [x["example_id"] for x in dis],
            "disagreement_detail": [{"example_id": x["example_id"], "primary": x["primary"],
                                     "secondary": x["secondary"], "raw": (x["raw"] or "")[:40]} for x in dis],
        }

    H = [x for x in out if x["label"] == "harmful"]
    B = [x for x in out if x["label"] == "benign"]
    summ = {"harmful": tally(H), "benign": tally(B), "all": tally(out)}
    # harness-compatible headline: asr_secondary / (1 - asr) are over ALL harmful items
    hs = summ["harmful"]
    summ["headline"] = {
        "asr_secondary": "%d/%d" % (hs["secondary_unsafe"], hs["n"]),
        "asr_both_judges": "%d/%d" % (hs["both_unsafe"], hs["n"]),
        "asr_primary": "%d/%d" % (hs["primary_unsafe"], hs["n"]),
        "conditional_residual_secondary": "%d/%d" % (hs["secondary_unsafe"], hs["delivered"]),
        "judge_disagreements_harmful": hs["disagreements"],
        "judge_disagreements_benign": summ["benign"]["disagreements"],
    }

    # how the primary column was produced, for the appendix
    cfg = blob.get("configuration") or {}
    meta = {
        "input": os.path.abspath(args.input),
        "input_sha256": hashlib.sha256(open(args.input, "rb").read()).hexdigest(),
        "primary_judge": (blob.get("judges") or {}).get("primary") or cfg.get("response_judge_backend"),
        "primary_name": args.primary_name,
        "secondary_judge": judge.describe(),
        "note": "records not generated or blocked by the output check are not judged (harness: "
                "harmful_completion_secondary is False for them)",
        "elapsed_s": round(time.time() - t0, 1),
        "generation_settings_of_input": cfg.get("generation_settings") or cfg.get("temperature"),
    }
    json.dump({"meta": meta, "summary": summ, "records": out},
              open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"summary": summ["headline"], "all": {k: summ["all"][k] for k in
                     ("n", "delivered", "disagreements")}}, ensure_ascii=False, indent=1))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
