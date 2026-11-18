# -*- coding: utf-8 -*-
"""End-to-end runs under other generation conditions and the 346-example calibration pool, recounted from records.

  T07   local models, temperature 0.7, 512 new tokens      r3: loc_{ep}_T07_512_{m}.json
  Q512  hosted model, temperature 0, 512 new tokens        r3: s9_qwen512_{m}.json
  P346  346-example calibration pool, T=0 / 128 tokens     r3: p346_qwen_{m}.json, r4: p346l_{ep}_{m}.json

Checks: 204 records without errors; stored outcomes re-derived as in collect_e2e; gate decisions equal to the T=0 runs of
the same method (T07, Q512) or identical across the three endpoints (P346); recorded generation settings.

    python -X utf8 collect_conditions_e2e.py [--root <project root>]
"""
import argparse
import json
from pathlib import Path

import collect_e2e as C
from verify_second_judge import R3, R4

HERE = Path(__file__).resolve().parent
M4 = ("no-defense", "llamaguard-oneshot", "rcdp_dist", "rcdp_lgprob")


def load(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    recs = {r["example_id"]: r for r in d["records"]}
    assert len(recs) == 204 and not any("error" in r for r in recs.values()), p.name
    return d, {i: C.flags(r) for i, r in recs.items()}, recs


def counts(f):
    H = [i for i in f if f[i]["label"] == "harmful"]; B = [i for i in f if f[i]["label"] == "benign"]
    return {"pre": sum(f[i]["pre"] for i in H), "ogc": sum(f[i]["ogc"] for i in H),
            "refused": sum(f[i]["refused"] and not f[i]["ogc"] for i in H), "abr_e2e": sum(f[i]["abr_e2e"] for i in H),
            "asr": sum(f[i]["asr"] for i in H), "allowed": sum(not f[i]["pre"] for i in B),
            "nonrefusal": sum(f[i]["nonrefusal"] for i in B), "ogc_benign": sum(f[i]["ogc"] for i in B)}


def gen_settings(d):
    g = d.get("generation_settings") or {}
    return {k: g.get(k) for k in ("temperature", "max_new_tokens", "top_p", "do_sample", "seed") if k in g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out, L = {}, ["# End-to-end runs under other conditions", "",
                  "| condition | endpoint | method | pre | ogc | refused | ABR_e2e | judged harmful | allowed | non-refusal | gen settings |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
    jobs = [("T07", ep, m, root / (R3 + f"loc_{ep}_T07_512_{m}.json")) for ep in ("llama31", "mistral") for m in M4]
    jobs += [("Q512", "qwen", m, root / (R3 + f"s9_qwen512_{m}.json")) for m in M4]
    jobs += [("P346", "qwen", m, root / (R3 + f"p346_qwen_{m}.json")) for m in M4]
    jobs += [("P346", ep, m, root / (R4 + f"p346l_{ep}_{m}.json")) for ep in ("llama31", "mistral") for m in M4]
    gates = {}
    for cond, ep, m, p in jobs:
        d, f, _ = load(p)
        c = counts(f)
        gs = gen_settings(d)
        if cond in ("T07", "Q512"):
            t0 = {i: C.flags(r) for i, r in C.load(root, ep, m)[1].items()}
            diff = [i for i in f if f[i]["pre"] != t0[i]["pre"]]
            assert not diff, (cond, ep, m, diff[:5])
            c["t0_judged_harmful"] = sum(t0[i]["asr"] for i in t0 if t0[i]["label"] == "harmful")
            c["t0_nonrefusal"] = sum(t0[i]["nonrefusal"] for i in t0 if t0[i]["label"] == "benign")
        else:
            gates.setdefault(m, []).append((ep, {i: f[i]["pre"] for i in f}))
        out[f"{cond}|{ep}|{m}"] = {"counts": c, "generation": gs, "file": p.name}
        L.append(f"| {cond} | {ep} | {m} | {c['pre']} | {c['ogc']} | {c['refused']} | {c['abr_e2e']} | {c['asr']} | "
                 f"{c['allowed']} | {c['nonrefusal']} | {gs} |")
    for m, lst in gates.items():
        ref = lst[0][1]
        for ep, g in lst[1:]:
            assert g == ref, ("P346 gate differs across endpoints", m, ep)
    L += ["", "Gate decisions: T07/Q512 identical to the T=0 runs of the same method; P346 identical across the three endpoints."]
    b = HERE.parent / "build"
    (b / "conditions_e2e.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "conditions_e2e.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))
    for k, v in out.items():
        if "t0_judged_harmful" in v["counts"]:
            print(f"  {k}: judged harmful T0 {v['counts']['t0_judged_harmful']} -> {v['counts']['asr']}; "
                  f"non-refusal T0 {v['counts']['t0_nonrefusal']} -> {v['counts']['nonrefusal']}")


if __name__ == "__main__":
    main()
