# -*- coding: utf-8 -*-
"""State layer under the second judge (qwen3-next-80b) on the two local endpoints, T=0 / 128 tokens.

Gate-only: the second judge's label on the undefended answer to the same request. Full method: its label on the delivered
answer; a generation replaced by the output check delivers the controller's refusal and counts as not harmful. Because the
second judge is not the model used by the output check, this measures the instruction and output-check steps
independently of Llama Guard. Inputs as in verify_second_judge.py (run that first).

    python -X utf8 state_layer_second_judge.py [--root <project root>]
"""
import argparse
import json
from pathlib import Path

import collect_e2e as C
from verify_second_judge import R3

HERE = Path(__file__).resolve().parent


def sj_labels(root, ep, m):
    d = json.loads((root / (R3 + f"sjq_on_{ep}_{m}.json")).read_text(encoding="utf-8"))
    return {r["example_id"]: (r["delivered"] and r["secondary"] == "unsafe") for r in d["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    L, out, tex = ["# State layer under the second judge (qwen3-next-80b), local endpoints, T=0/128", ""], {}, []
    for ep in ("llama31", "mistral"):
        nd2 = sj_labels(root, ep, "no-defense")
        nd1 = {i: C.flags(r) for i, r in C.load(root, ep, "no-defense")[1].items()}
        tex.append(r"\multicolumn{7}{l}{\emph{" + C.ENDPOINT_NAMES[ep] + r"}} \\")
        for m in ("static-patch", "llamaguard-oneshot", "smoothllm-llamaguard", "rcdp_dist", "rcdp_deb", "rcdp_lgprob"):
            f = {i: C.flags(r) for i, r in C.load(root, ep, m)[1].items()}
            s2 = sj_labels(root, ep, m)
            ha = [i for i in f if f[i]["label"] == "harmful" and not f[i]["pre"]]
            g1 = sum(nd1[i]["asr"] for i in ha); f1 = sum(f[i]["asr"] for i in ha)
            g2 = sum(nd2[i] for i in ha); f2 = sum(s2[i] for i in ha)
            a = sum(s2[i] and not nd2[i] for i in ha); b = sum(nd2[i] and not s2[i] for i in ha)
            p = C.mcnemar(a, b)
            out[f"{ep}|{m}"] = {"reaching": len(ha), "primary_gate_only": g1, "primary_full": f1, "second_gate_only": g2,
                                "second_full": f2, "second_discordant_[full_only,gate_only]": [a, b], "p_second": p}
            L.append(f"- {ep:8s} {m:22s} reaching {len(ha):3d}: primary {g1:2d} -> {f1:2d} | second {g2:2d} -> {f2:2d} "
                     f"(disc {a}/{b}, p={p:.3g})")
            tex.append(f"{C.TEXNAME[m]} & {len(ha)} & {g1} & {f1} & {g2} & {f2} & {C.fmt_p(p)} " + "\\\\")
        L.append("")
    b = HERE.parent / "build"
    (b / "state_layer_second_judge.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "state_layer_second_judge.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (b / "tableS8b_second_judge_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
