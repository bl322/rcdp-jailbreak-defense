# -*- coding: utf-8 -*-
"""Supplementary Table S12: primary and second judge on the delivered answers to the 149 harmful requests, every method
and endpoint (T=0 / 128 tokens) and the local endpoints at T=0.7 / 512 tokens.

Hosted model: second judge Llama-3.1-8B-Instruct, recorded in the generation runs (judge_secondary).
Local models: second judge qwen3-next-80b-a3b-instruct re-scoring the saved answers (verify_second_judge.py, which checks
provenance and must be run first). Judge-disputed: delivered answers that exactly one judge labels unsafe.

    python -X utf8 table_s12_judges.py [--root <project root>]
"""
import argparse
import json
from pathlib import Path

import collect_e2e as C

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    local = json.loads((HERE.parent / "build" / "second_judge_local.json").read_text(encoding="utf-8"))
    rows, tex = [], []
    tex.append(r"\multicolumn{7}{l}{\emph{" + C.ENDPOINT_NAMES["qwen"] + r", second judge \texttt{Llama-3.1-8B-Instruct}}} \\")
    for m in C.METHODS:
        _, recs = C.load(root, "qwen", m)
        n = p = s = both = po = so = unk = 0
        for r in recs.values():
            if r["label"] != "harmful" or not r["generated"] or r["outcome"]["blocked_output_check"]:
                continue
            n += 1
            pu = r["judge_primary"]["verdict"] == "unsafe"
            sv = (r.get("judge_secondary") or {}).get("verdict")
            su = sv == "unsafe"
            unk += sv not in ("safe", "unsafe")
            assert pu == r["outcome"]["harmful_completion_primary"]
            p += pu; s += su; both += pu and su; po += pu and not su; so += su and not pu
        rows.append(("qwen", "T0/128", m, n, p, s, both, po, so, unk))
        tex.append(f"{C.TEXNAME[m]} & {n} & {p} & {s} & {both} & {po} & {so} \\\\")
    for ep in ("llama31", "mistral"):
        for dec, ms in (("T0/128", C.METHODS), ("T0.7/512", ("no-defense", "llamaguard-oneshot", "rcdp_dist", "rcdp_lgprob"))):
            label = "" if dec == "T0/128" else r", temperature 0.7, 512 tokens"
            tex.append(r"\multicolumn{7}{l}{\emph{" + C.ENDPOINT_NAMES[ep] + label +
                       r", second judge \texttt{qwen3-next-80b-a3b-instruct}}} \\")
            for m in ms:
                c = local[f"{dec}|{ep}|{m}|qwen3-next"]["counts"]["harmful"]
                rows.append((ep, dec, m, c["delivered"], c["primary_unsafe"], c["secondary_unsafe"], c["both_unsafe"],
                             c["primary_only"], c["secondary_only"], c["secondary_unknown"]))
                tex.append(f"{C.TEXNAME[m]} & {c['delivered']} & {c['primary_unsafe']} & {c['secondary_unsafe']} & "
                           f"{c['both_unsafe']} & {c['primary_only']} & {c['secondary_only']} \\\\")
    b = HERE.parent / "build"
    (b / "tableS12_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (b / "tableS12.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
