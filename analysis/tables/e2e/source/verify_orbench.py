# -*- coding: utf-8 -*-
"""OR-Bench-hard: 500-prompt subset on the hosted model (r4 package), recounted from records.

Checks: the subset is the first 500 prompts of OR-Bench-hard-1k ordered by sha256('orbench-r4:<example_id>') (recomputed
from the released 1,319-row file) and matches the manifest and the file hash recorded in every run; the gate decisions of
the generation runs (orb5_*) equal those of the gate-only runs (orbg5_*); stored outcomes re-derived. Reports gate
allowance, refusals and non-refusal answers, paired McNemar tests, and the state layer (gate-only counterfactual =
undefended outcome on the same allowed requests).

    python -I -X utf8 verify_orbench.py [--root <project root>]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = "_r1_out/r4/extracted/r4_pack/"
METHODS = ("no-defense", "llamaguard-oneshot", "rcdp_dist", "rcdp_lgprob")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    sys.path.insert(0, str(HERE))
    import collect_e2e as C
    p = root / P
    full = [json.loads(l) for l in (p / "orbench_hard_1k.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    man = json.loads((p / "orbench_hard_sub500.manifest.json").read_text(encoding="utf-8"))
    sub_raw = (p / "orbench_hard_sub500.jsonl").read_bytes()
    sub = [json.loads(l) for l in sub_raw.decode("utf-8").splitlines() if l.strip()]
    order = sorted((r["example_id"] for r in full), key=lambda i: hashlib.sha256(f"orbench-r4:{i}".encode()).hexdigest())[:500]
    assert len(full) == 1319 and len({r["example_id"] for r in full}) == 1319
    assert set(order) == set(man["ids"]) == {r["example_id"] for r in sub}, "subset rule"
    assert hashlib.sha256(sub_raw).hexdigest() == man["sha256_subset_file"]
    byid = {r["example_id"]: r for r in full}
    assert all(r["turns"] == byid[r["example_id"]]["turns"] for r in sub), "subset text differs from source"
    L = [f"- subset: rule reproduced from the 1,319-row file; ids match manifest; file sha256 {man['sha256_subset_file'][:12]}...", ""]
    F, G = {}, {}
    for m in METHODS:
        d = json.loads((p / f"e2e/orb5_{m}.json").read_text(encoding="utf-8"))
        assert d["datasets"]["evaluation"]["sha256"] == man["sha256_subset_file"]
        recs = {r["example_id"]: r for r in d["records"]}
        assert len(recs) == 500 and set(recs) == set(order) and not any("error" in r for r in recs.values())
        F[m] = {i: C.flags(r) for i, r in recs.items()}
        G[m] = recs
        if m != "no-defense":
            g = {r["example_id"]: r for r in json.loads((p / f"e2e/orbg5_{m}.json").read_text(encoding="utf-8"))["records"]}
            diff = [i for i in recs if recs[i]["outcome"]["blocked_pre_generation"] != g[i]["outcome"]["blocked_pre_generation"]]
            assert not diff, (m, "gate differs from gate-only run", diff[:3])
        assert d["generation_settings"]["model"] == "qwen3-next-80b-a3b-instruct"
    out = {}
    nd = F["no-defense"]
    for m in METHODS:
        f = F[m]
        allowed = [i for i in f if not f[i]["pre"]]
        st = {}
        for i in allowed:
            s = G[m][i].get("state") or "none"
            t = st.setdefault(s, [0, 0, 0])
            t[0] += 1; t[1] += f[i]["nonrefusal"]; t[2] += nd[i]["nonrefusal"]
        rec = {"allowed": len(allowed), "ogc": sum(f[i]["ogc"] for i in f), "refused": sum(f[i]["refused"] and not f[i]["ogc"] for i in f),
               "nonrefusal": sum(f[i]["nonrefusal"] for i in f),
               "gate_only_nonrefusal": sum(nd[i]["nonrefusal"] for i in allowed),
               "by_state_[n,nonrefusal_full,nonrefusal_gate_only]": st}
        a = sum(f[i]["nonrefusal"] and not nd[i]["nonrefusal"] for i in allowed)
        b = sum(nd[i]["nonrefusal"] and not f[i]["nonrefusal"] for i in allowed)
        rec["state_layer_disc_[full_only,gate_only]"] = [a, b]; rec["p_state_layer"] = C.mcnemar(a, b)
        out[m] = rec
        L.append(f"- {m:20s} allowed {len(allowed)}/500, output check {rec['ogc']}, refused {rec['refused']}, non-refusal "
                 f"{rec['nonrefusal']}/500 | gate only {rec['gate_only_nonrefusal']} (disc {a}/{b}, p={C.fmt_p(rec['p_state_layer'])}) | by state {st}")
    L.append("")
    for x, y in (("rcdp_lgprob", "llamaguard-oneshot"), ("rcdp_dist", "llamaguard-oneshot"), ("rcdp_lgprob", "rcdp_dist")):
        for key, nm in (("pre", "allowed"), ("nonrefusal", "non-refusal")):
            if key == "pre":
                a = sum(not F[x][i]["pre"] and F[y][i]["pre"] for i in order); b = sum(F[x][i]["pre"] and not F[y][i]["pre"] for i in order)
            else:
                a = sum(F[x][i][key] and not F[y][i][key] for i in order); b = sum(F[y][i][key] and not F[x][i][key] for i in order)
            out[f"{x} vs {y} {nm}"] = {"disc": [a, b], "p": C.mcnemar(a, b)}
            L.append(f"- {x} vs {y}, {nm}: {a}/{b} discordant, p={C.fmt_p(C.mcnemar(a, b))}")
    b = HERE.parent / "build"
    (b / "orbench.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "orbench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
