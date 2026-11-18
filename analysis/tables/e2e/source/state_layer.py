# -*- coding: utf-8 -*-
"""What the RCDP states add beyond the gate, end to end (R2 Weakness 1).

Gate-only counterfactual: a request the gate allows is sent without any instruction and without the output check. The
undefended run sends exactly that input (no system prompt, same user turn, greedy decoding), so for the requests a gate
allows, the gate-only outcome is the undefended outcome on the same request. On the two locally served models greedy
decoding reproduces the text exactly (probe: 30/30), so the counterfactual is exact there; on the hosted model it is an
approximation (probe: 2/30 identical texts, 30/30 identical refusal decisions, 28/30 identical judge verdicts).

Validation: the one-shot Llama-Guard baseline is itself gate-only, so its actual run must match the counterfactual
built from the undefended run.

    python -I -X utf8 state_layer.py [--root <project root>]
"""
import argparse
import json
from pathlib import Path

import collect_e2e as C

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    a = ap.parse_args()
    root = Path(a.root)
    F = {(ep, m): {i: C.flags(r) for i, r in C.load(root, ep, m)[1].items()} for ep in C.ENDPOINTS for m in C.METHODS}
    ids = sorted(F["qwen", "no-defense"])
    H = [i for i in ids if F["qwen", "no-defense"][i]["label"] == "harmful"]
    B = [i for i in ids if F["qwen", "no-defense"][i]["label"] == "benign"]
    out, L = {}, ["# State layer: full method vs gate-only counterfactual (undefended outcome on the allowed requests)", ""]
    for ep in C.ENDPOINTS:
        nd = F[ep, "no-defense"]
        for m in ("llamaguard-oneshot", "smoothllm-llamaguard", "rcdp_dist", "rcdp_deb", "rcdp_lgprob"):
            f = F[ep, m]
            ha = [i for i in H if not f[i]["pre"]]
            ba = [i for i in B if not f[i]["pre"]]
            # harmful: judged-harmful delivered answers; benign: non-refusal answers
            h_full = sum(f[i]["asr"] for i in ha); h_gate = sum(nd[i]["asr"] for i in ha)
            h_a = sum(f[i]["asr"] and not nd[i]["asr"] for i in ha); h_b = sum(nd[i]["asr"] and not f[i]["asr"] for i in ha)
            b_full = sum(f[i]["nonrefusal"] for i in ba); b_gate = sum(nd[i]["nonrefusal"] for i in ba)
            b_a = sum(f[i]["nonrefusal"] and not nd[i]["nonrefusal"] for i in ba)
            b_b = sum(nd[i]["nonrefusal"] and not f[i]["nonrefusal"] for i in ba)
            # refusals of harmful requests that reach the model
            r_full = sum(f[i]["refused"] or f[i]["ogc"] for i in ha); r_gate = sum(nd[i]["refused"] for i in ha)
            ogc = sum(f[i]["ogc"] for i in ha)
            rec = {"harmful_reaching": len(ha), "judged_harmful_full": h_full, "judged_harmful_gate_only": h_gate,
                   "judged_harmful_discordant": [h_a, h_b], "p_harmful": C.mcnemar(h_a, h_b),
                   "harmful_refused_or_replaced_full": r_full, "harmful_refused_gate_only": r_gate, "output_check_harmful": ogc,
                   "benign_reaching": len(ba), "nonrefusal_full": b_full, "nonrefusal_gate_only": b_gate,
                   "nonrefusal_discordant": [b_a, b_b], "p_benign": C.mcnemar(b_a, b_b)}
            if m.startswith("rcdp"):
                by = {}
                for i in ha:
                    s = f[i]["state"]; t = by.setdefault(s, [0, 0, 0])
                    t[0] += 1; t[1] += f[i]["asr"]; t[2] += nd[i]["asr"]
                rec["harmful_by_state_[n,full,gate_only]"] = by
            out[f"{m} @ {ep}"] = rec
            L.append(f"- {ep:8s} {m:22s} harmful reaching {len(ha):3d}: judged harmful full {h_full:2d} vs gate-only {h_gate:2d} "
                     f"(disc {h_a}/{h_b}, p={C.mcnemar(h_a, h_b):.3g}); refused/replaced {r_full} vs {r_gate} (ogc {ogc}) | "
                     f"benign reaching {len(ba)}: non-refusal full {b_full} vs gate-only {b_gate} (disc {b_a}/{b_b}, "
                     f"p={C.mcnemar(b_a, b_b):.3g})" + (f" | by state {rec['harmful_by_state_[n,full,gate_only]']}" if m.startswith("rcdp") else ""))
        L.append("")
    # Supplementary Table S8 rows: static instruction (no gate) and the three RCDP configurations, per model
    short = {"static-patch": "Static instruction", "rcdp_dist": "RCDP, DistilBART", "rcdp_deb": "RCDP, DeBERTa",
             "rcdp_lgprob": "RCDP, Llama-Guard"}
    fmt_p = C.fmt_p
    tex = []
    for ep in C.ENDPOINTS:
        nd = F[ep, "no-defense"]
        raw = {m: C.load(root, ep, m)[1] for m in ("rcdp_dist", "rcdp_deb", "rcdp_lgprob")}
        tex.append(r"\multicolumn{10}{l}{\emph{" + C.ENDPOINT_NAMES[ep] + r"}} \\")
        for m in ("static-patch", "rcdp_dist", "rcdp_deb", "rcdp_lgprob"):
            f = F[ep, m]
            ha = [i for i in H if not f[i]["pre"]]
            ba = [i for i in B if not f[i]["pre"]]
            g = sum(nd[i]["asr"] for i in ha)
            full = sum(f[i]["asr"] for i in ha)
            if m == "static-patch":
                instr, chk = str(full), "--"
            else:
                instr = str(sum(raw[m][i]["outcome"]["harmful_completion_primary_before_output_check"] for i in ha))
                chk = str(full)
            h_a = sum(f[i]["asr"] and not nd[i]["asr"] for i in ha); h_b = sum(nd[i]["asr"] and not f[i]["asr"] for i in ha)
            bg = sum(nd[i]["nonrefusal"] for i in ba); bf = sum(f[i]["nonrefusal"] for i in ba)
            b_a = sum(f[i]["nonrefusal"] and not nd[i]["nonrefusal"] for i in ba)
            b_b = sum(nd[i]["nonrefusal"] and not f[i]["nonrefusal"] for i in ba)
            tex.append(f"{short[m]} & {len(ha)} & {g} & {instr} & {chk} & {fmt_p(C.mcnemar(h_a, h_b))} & "
                       f"{len(ba)} & {bg} & {bf} & {fmt_p(C.mcnemar(b_a, b_b))} " + "\\\\")
    b = HERE.parent / "build"
    (b / "tableS8_state_layer_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (b / "state_layer.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "state_layer.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
