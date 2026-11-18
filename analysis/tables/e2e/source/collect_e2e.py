# -*- coding: utf-8 -*-
"""End-to-end results (Table 3) from the R1 e2e result files, recounted from the per-example records.

Inputs (read-only): _r1_out/e2e_qwen/extracted/e2e_qwen/*.json (qwen baselines) and
_r1_out/e2e_rcdp/extracted/results/r1/e2e/*.json (RCDP on qwen, all configurations on the two local endpoints,
llm-local refusal re-checks). Writes build/e2e_summary.md and build/e2e.json.

    python -I -X utf8 collect_e2e.py [--root <project root>]
"""
import argparse
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q = "_r1_out/e2e_qwen/extracted/e2e_qwen/"
R = "_r1_out/e2e_rcdp/extracted/results/r1/e2e/"
ENDPOINTS = ("qwen", "llama31", "mistral")
ENDPOINT_NAMES = {"qwen": "qwen3-next-80b-a3b-instruct (hosted)", "llama31": "Llama-3.1-8B-Instruct (local)",
                  "mistral": "Mistral-7B-Instruct-v0.3 (local)"}
METHODS = ("no-defense", "static-patch", "llamaguard-oneshot", "smoothllm-llamaguard", "rcdp_dist", "rcdp_deb", "rcdp_lgprob")
METHOD_NAMES = {"no-defense": "No defence", "static-patch": "Static safety instruction",
                "llamaguard-oneshot": "Llama-Guard-3-1B, one-shot verdict", "smoothllm-llamaguard": "SmoothLLM-style + Llama Guard",
                "rcdp_dist": "RCDP, DistilBART-MNLI", "rcdp_deb": "RCDP, DeBERTa-v3-large", "rcdp_lgprob": "RCDP, Llama-Guard-3-1B, P(unsafe)"}


TEXNAME = {"no-defense": "No defence", "static-patch": "Static safety instruction",
           "llamaguard-oneshot": "Llama-Guard, one-shot verdict", "smoothllm-llamaguard": "SmoothLLM-style + Llama Guard",
           "rcdp_dist": "RCDP, DistilBART-MNLI", "rcdp_deb": "RCDP, DeBERTa-v3-large",
           "rcdp_lgprob": r"RCDP, Llama-Guard, $P(\texttt{unsafe})$"}


def path(ep, m):
    if ep == "qwen":
        if m.startswith("rcdp_"):
            return R + {"rcdp_dist": "rcdp_qwen_dist_agnostic", "rcdp_deb": "rcdp_qwen_deberta_agnostic",
                        "rcdp_lgprob": "rcdp_qwen_lgprob_agnostic"}[m] + ".json"
        return Q + f"bl_{m}_qwen3next.json"
    return R + f"loc_{ep}_{m}.json"


from decimal import Decimal, ROUND_HALF_UP

def fmt_p(p):
    """Two significant digits, half-up (as in the rest of the manuscript)."""
    if p >= 0.995:
        return "1.0"
    d = Decimal(repr(p))
    ex = d.adjusted()
    q = d.scaleb(-ex).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if q >= 10:
        q, ex = q / 10, ex + 1
    if ex >= -3:
        return str(q.scaleb(ex).quantize(Decimal(1).scaleb(ex - 1)))
    return f"${q}\\times10^{{{ex}}}$"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def clopper_pearson(k, n, alpha=0.05):
    def betainc_inv(p, a, b):  # bisection on the regularized incomplete beta via the binomial tail
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            # P(X >= a | n = a + b - 1, p = mid) = I_mid(a, b)
            nn = a + b - 1
            tail = sum(comb(nn, j) * mid ** j * (1 - mid) ** (nn - j) for j in range(a, nn + 1))
            lo, hi = (mid, hi) if tail < p else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else betainc_inv(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else betainc_inv(1 - alpha / 2, k + 1, n - k)
    return lower, upper


def load(root, ep, m):
    d = json.loads((root / path(ep, m)).read_text(encoding="utf-8"))
    recs = {r["example_id"]: r for r in d["records"]}
    assert len(recs) == 204 and not any("error" in r for r in recs.values()), (ep, m)
    return d, recs


def flags(r):
    o = r["outcome"]
    return {"label": r["label"], "gate_block": r["controller_decision"] == "block" if "controller_decision" in r else False,
            "pre": o["blocked_pre_generation"], "ogc": o["blocked_output_check"], "refused": o["refused"],
            "abr_e2e": o["blocked_e2e"], "asr": o["harmful_completion_primary"], "nonrefusal": o["useful_answer"],
            "state": r.get("state"), "generated": r["generated"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    a = ap.parse_args()
    root = Path(a.root)
    F, META = {}, {}
    for ep in ENDPOINTS:
        for m in METHODS:
            d, recs = load(root, ep, m)
            F[ep, m] = {i: flags(r) for i, r in recs.items()}
            META[ep, m] = {"generation": d.get("generation_settings") or d.get("settings", {}).get("generation"),
                           "judge2": bool((d.get("judges") or {}).get("secondary"))}
    ids = sorted(F["qwen", "no-defense"])
    H = [i for i in ids if F["qwen", "no-defense"][i]["label"] == "harmful"]
    B = [i for i in ids if F["qwen", "no-defense"][i]["label"] == "benign"]
    assert len(H) == 149 and len(B) == 55
    out = {"rows": {}, "paired": {}, "state_effect": {}, "recheck": {}}
    L = ["# End-to-end results (R1), recounted from records", ""]
    # gate decisions must not depend on the endpoint
    for m in METHODS:
        g = {ep: tuple(F[ep, m][i]["pre"] for i in ids) for ep in ENDPOINTS}
        assert len(set(g.values())) == 1, ("gate differs across endpoints", m)
    L.append("Gate (pre-generation) decisions are identical across the three endpoints for every method (asserted).")
    L.append("")
    L.append("| Method | gate H blocked /149 | gate B allowed /55 | " + " | ".join(
        f"{ep}: ASR /149 · ABR_e2e /149 · non-refusal /55" for ep in ENDPOINTS) + " |")
    L.append("|---|---|---|" + "---|" * len(ENDPOINTS))
    for m in METHODS:
        f0 = F["qwen", m]
        gh = sum(f0[i]["pre"] for i in H)
        gb = sum(not f0[i]["pre"] for i in B)
        row = {"gate_harmful_blocked": gh, "gate_benign_allowed": gb}
        cells = []
        for ep in ENDPOINTS:
            f = F[ep, m]
            asr = sum(f[i]["asr"] for i in H)
            abr = sum(f[i]["abr_e2e"] for i in H)
            nr = sum(f[i]["nonrefusal"] for i in B)
            lo, hi = clopper_pearson(asr, 149)
            row[ep] = {"asr": asr, "asr_ci95": [lo, hi], "abr_e2e": abr, "nonrefusal": nr,
                       "harmful_refused": sum(f[i]["refused"] for i in H),
                       "harmful_refused_and_judged_unsafe": sum(f[i]["refused"] and f[i]["asr"] for i in H),
                       "harmful_neither_refused_nor_unsafe": sum(not f[i]["abr_e2e"] and not f[i]["asr"] for i in H),
                       "benign_refused_after_gate": sum(f[i]["refused"] for i in B),
                       "benign_output_check_blocked": sum(f[i]["ogc"] for i in B),
                       "harmful_output_check_blocked": sum(f[i]["ogc"] for i in H)}
            cells.append(f"{asr} [{100 * lo:.1f}, {100 * hi:.1f}] · {abr} · {nr}")
        out["rows"][m] = row
        L.append(f"| {METHOD_NAMES[m]} | {gh} | {gb} | " + " | ".join(cells) + " |")
    tex = []
    has_ogc = {"rcdp_dist", "rcdp_deb", "rcdp_lgprob"}
    for ep in ENDPOINTS:
        tex.append(r"\multicolumn{8}{l}{\emph{(" + "abc"[ENDPOINTS.index(ep)] + ") " + ENDPOINT_NAMES[ep] + r"}} \\")
        for m in METHODS:
            r, c = out["rows"][m], out["rows"][m][ep]
            ogc = str(c["harmful_output_check_blocked"]) if m in has_ogc else "--"
            pre = str(r["gate_harmful_blocked"]) if m not in ("no-defense", "static-patch") else "--"
            alw = str(r["gate_benign_allowed"]) if m not in ("no-defense", "static-patch") else "--"
            tex.append(f"{TEXNAME[m]} & {pre} & {ogc} & {c['harmful_refused']} & {c['abr_e2e']} & {c['asr']} & "
                       f"{alw} & {c['nonrefusal']} " + "\\\\")
        if ep != ENDPOINTS[-1]:
            tex.append(r"\midrule")
    (HERE.parent / "build").mkdir(exist_ok=True)
    (HERE.parent / "build" / "table3_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    L.append("")
    L.append("Overlaps (harmful): " + "; ".join(f"{ep}/{m}: refused&unsafe {out['rows'][m][ep]['harmful_refused_and_judged_unsafe']}, "
                                               f"neither {out['rows'][m][ep]['harmful_neither_refused_nor_unsafe']}"
                                               for ep in ENDPOINTS for m in ("no-defense", "static-patch")))
    L.append("")
    L.append("## Paired exact McNemar (per endpoint): ASR on 149 harmful, non-refusal on 55 benign")
    L.append("Format: only-A / only-B, p.  A = row method, B = comparison method.")
    comps = [("rcdp_lgprob", "llamaguard-oneshot"), ("rcdp_dist", "llamaguard-oneshot"), ("rcdp_deb", "llamaguard-oneshot"),
             ("smoothllm-llamaguard", "llamaguard-oneshot"), ("static-patch", "no-defense"), ("llamaguard-oneshot", "no-defense"),
             ("rcdp_lgprob", "no-defense"), ("rcdp_dist", "no-defense"), ("rcdp_lgprob", "static-patch"), ("llamaguard-oneshot", "static-patch")]
    for x, y in comps:
        parts = []
        for ep in ENDPOINTS:
            fx, fy = F[ep, x], F[ep, y]
            a1 = sum(fx[i]["asr"] and not fy[i]["asr"] for i in H); b1 = sum(fy[i]["asr"] and not fx[i]["asr"] for i in H)
            a2 = sum(fx[i]["nonrefusal"] and not fy[i]["nonrefusal"] for i in B); b2 = sum(fy[i]["nonrefusal"] and not fx[i]["nonrefusal"] for i in B)
            out["paired"][f"{x} vs {y} @ {ep}"] = {"asr": [a1, b1, mcnemar(a1, b1)], "nonrefusal": [a2, b2, mcnemar(a2, b2)]}
            parts.append(f"{ep}: ASR {a1}/{b1} p={mcnemar(a1, b1):.3g}; non-refusal {a2}/{b2} p={mcnemar(a2, b2):.3g}")
        L.append(f"- {x} vs {y}: " + " | ".join(parts))
    L.append("")
    L.append("## State-instruction effect on benign requests that reach the model")
    L.append("For benign requests a method lets through, refusals under that method vs under no defence on the same requests.")
    for ep in ENDPOINTS:
        for m in ("static-patch", "llamaguard-oneshot", "smoothllm-llamaguard", "rcdp_dist", "rcdp_deb", "rcdp_lgprob"):
            f, f0 = F[ep, m], F[ep, "no-defense"]
            thru = [i for i in B if not f[i]["pre"]]
            ref_m = sum(f[i]["refused"] for i in thru)
            ref_0 = sum(f0[i]["refused"] for i in thru)
            a1 = sum(f[i]["refused"] and not f0[i]["refused"] for i in thru); b1 = sum(f0[i]["refused"] and not f[i]["refused"] for i in thru)
            by_state = {}
            for i in thru:
                s = f[i]["state"] or "-"
                t = by_state.setdefault(s, [0, 0, 0])
                t[0] += 1; t[1] += f[i]["refused"]; t[2] += f0[i]["refused"]
            out["state_effect"][f"{m} @ {ep}"] = {"reach_model": len(thru), "refused": ref_m, "refused_no_defense_same_items": ref_0,
                                                  "only_method": a1, "only_no_defense": b1, "p": mcnemar(a1, b1), "by_state": by_state}
            L.append(f"- {ep} {m}: reach {len(thru)}, refused {ref_m} vs {ref_0} without defence (only-method {a1} / only-none {b1}, "
                     f"p={mcnemar(a1, b1):.3g}); by state [n, refused, refused w/o defence]: {by_state}")
    L.append("")
    L.append("## llm-local refusal re-check (qwen; saved generations, no regeneration)")
    for m, fn in (("no-defense", "recheck_bl_no-defense_qwen3next"), ("static-patch", "recheck_bl_static-patch_qwen3next"),
                  ("llamaguard-oneshot", "recheck_bl_llamaguard-oneshot_qwen3next"),
                  ("smoothllm-llamaguard", "recheck_bl_smoothllm-llamaguard_qwen3next"),
                  ("rcdp_dist", "recheck_rcdp_qwen_dist_agnostic"), ("rcdp_deb", "recheck_rcdp_qwen_deberta_agnostic"),
                  ("rcdp_lgprob", "recheck_rcdp_qwen_lgprob_agnostic")):
        d = json.loads((root / R / f"{fn}.json").read_text(encoding="utf-8"))
        rows = {r["example_id"]: r for r in d["rows"]}
        _, recs = load(root, "qwen", m)
        def nr(i, key):
            o = recs[i]
            return bool(o["generated"] and not o["output_guard"]["blocked"] and (o.get("response") or "").strip()) and rows[i][key] == "NR"
        lex = sum(nr(i, "lexical_class") for i in B); llm = sum(nr(i, "llm_class_delivered") for i in B)
        agree = sum(nr(i, "lexical_class") == nr(i, "llm_class_delivered") for i in B)
        assert lex == out["rows"][m]["qwen"]["nonrefusal"], (m, lex)
        out["recheck"][m] = {"lexical": lex, "llm_local": llm, "agree": agree}
        L.append(f"- {m}: non-refusal lexical {lex}/55, llm-local {llm}/55, item agreement {agree}/55")
    s7 = [f"{TEXNAME[m]} & {v['lexical']} & {v['llm_local']} & {v['agree']} " + "\\\\" for m, v in out["recheck"].items()]
    (HERE.parent / "build" / "tableS7_rows.tex").write_text("\n".join(s7) + "\n", encoding="utf-8")

    L.append("")
    L.append("Second judge configured: " + ", ".join(f"{ep}/{m}" for (ep, m), v in META.items() if v["judge2"]))
    b = HERE.parent / "build"
    b.mkdir(exist_ok=True)
    (b / "e2e_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "e2e.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
