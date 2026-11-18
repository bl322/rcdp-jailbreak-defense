# -*- coding: utf-8 -*-
"""Second-judge re-scoring of the saved local-endpoint generations: provenance check and recount.

Inputs (read-only):
  r3 package  _r1_out/e2e_r3/extracted/results/r1/e2e/
      sjq_on_{ep}_{m}.json              qwen3-next-80b judge on the T=0 / 128-token runs
      sj_{judge}_on_{ep}_{m}.json       cross pairing (Mistral-7B judges Llama outputs, Llama-3.1-8B judges Mistral outputs)
      loc_{ep}_T07_512_{m}.json         T=0.7 / 512-token generation runs
  r4 package  _r1_out/r4/extracted/r4_pack/e2e/sjq_on_{ep}_T07_512_{m}.json   qwen judge on the T=0.7 runs
  T=0 generations: _r1_out/e2e_rcdp/extracted/results/r1/e2e/loc_{ep}_{m}.json

Checks: the judge file's input_sha256 equals the local generation file; 'delivered' and the primary label of every record
equal the generation run; the stored summary equals the recount from the records.

    python -X utf8 verify_second_judge.py [--root <project root>]
"""
import argparse
import hashlib
import json
from pathlib import Path

import collect_e2e as C

# R1 release (only change to the script as run): on the redacted result package, the SHA-256 of the original file,
# as recorded in _r1_out/ORIGINAL_SHA256.json by analysis/setup_layout.py; otherwise the file's own SHA-256.
def file_sha256(p):
    p = Path(p).resolve()
    for d in p.parents:
        idx = d / "ORIGINAL_SHA256.json"
        if d.name == "_r1_out" and idx.is_file():
            got = json.loads(idx.read_text(encoding="utf-8")).get(p.relative_to(d).as_posix())
            if got:
                return got
    return hashlib.sha256(p.read_bytes()).hexdigest()

HERE = Path(__file__).resolve().parent
R3 = "_r1_out/e2e_r3/extracted/results/r1/e2e/"
R4 = "_r1_out/r4/extracted/r4_pack/e2e/"
EPS = ("llama31", "mistral")
M_T0 = ("no-defense", "static-patch", "llamaguard-oneshot", "smoothllm-llamaguard", "rcdp_dist", "rcdp_deb", "rcdp_lgprob")
M_T07 = ("no-defense", "llamaguard-oneshot", "rcdp_dist", "rcdp_lgprob")
CROSS = {"llama31": "mistral", "mistral": "llama31"}


def sha(p):
    return file_sha256(p)


def gen_path(root, ep, m, t07):
    return root / (R3 + f"loc_{ep}_T07_512_{m}.json" if t07 else C.R + f"loc_{ep}_{m}.json")


def check(root, jf, gf):
    j = json.loads(jf.read_text(encoding="utf-8"))
    assert j["meta"]["input_sha256"] == sha(gf), (jf.name, "input hash")
    g = {r["example_id"]: r for r in json.loads(gf.read_text(encoding="utf-8"))["records"]}
    assert len(g) == 204 and len(j["records"]) == 204, jf.name
    cnt = {lab: {"n": 0, "delivered": 0, "primary_unsafe": 0, "secondary_unsafe": 0, "both_unsafe": 0,
                 "secondary_unknown": 0, "disagreements": 0, "primary_only": 0, "secondary_only": 0} for lab in ("harmful", "benign")}
    ids = {"harmful": {"primary_only": [], "secondary_only": [], "both": []}, "benign": {"primary_only": [], "secondary_only": [], "both": []}}
    for r in j["records"]:
        e = g[r["example_id"]]
        o = e["outcome"]
        assert r["label"] == e["label"]
        dlv = bool(e["generated"] and not o["blocked_output_check"])
        assert r["delivered"] == dlv, (jf.name, r["example_id"], "delivered")
        c = cnt[r["label"]]
        c["n"] += 1
        if not dlv:
            continue
        c["delivered"] += 1
        p_uns = e["judge_primary"]["verdict"] == "unsafe"
        assert (r["primary"] == "unsafe") == p_uns, (jf.name, r["example_id"], "primary")
        if r["label"] == "harmful":
            assert p_uns == o["harmful_completion_primary"], (jf.name, r["example_id"], "asr")
        s = r["secondary"]
        s_uns = s == "unsafe"
        if s not in ("safe", "unsafe"):
            c["secondary_unknown"] += 1
        c["primary_unsafe"] += p_uns
        c["secondary_unsafe"] += s_uns
        c["both_unsafe"] += p_uns and s_uns
        # stored 'disagreements' compares label strings, so an unparsable second-judge reply ('unknown') counts as one
        c["disagreements"] += r["secondary"] != r["primary"]
        if p_uns != s_uns:
            c["primary_only" if p_uns else "secondary_only"] += 1
            ids[r["label"]]["primary_only" if p_uns else "secondary_only"].append(r["example_id"])
        elif p_uns:
            ids[r["label"]]["both"].append(r["example_id"])
    for lab in ("harmful", "benign"):
        st = j["summary"][lab]
        for k in ("n", "delivered", "primary_unsafe", "secondary_unsafe", "both_unsafe", "secondary_unknown", "disagreements"):
            if k in st:
                assert st[k] == cnt[lab][k], (jf.name, lab, k, st[k], cnt[lab][k])
    return cnt, ids, j["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out, L = {}, ["# Second judge on the local-endpoint generations (recounted from records)", "",
                  "| decode | endpoint | method | judge | H delivered | primary | second | both | primary only | second only | unknown | B delivered | B disagree |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    judges_seen = {}
    jobs = []
    for ep in EPS:
        for m in M_T0:
            jobs.append(("T0/128", ep, m, "qwen3-next", root / (R3 + f"sjq_on_{ep}_{m}.json"), gen_path(root, ep, m, False)))
            jobs.append(("T0/128", ep, m, CROSS[ep], root / (R3 + f"sj_{CROSS[ep]}_on_{ep}_{m}.json"), gen_path(root, ep, m, False)))
        for m in M_T07:
            jobs.append(("T0.7/512", ep, m, "qwen3-next", root / (R4 + f"sjq_on_{ep}_T07_512_{m}.json"), gen_path(root, ep, m, True)))
    for dec, ep, m, jd, jf, gf in jobs:
        cnt, ids, meta = check(root, jf, gf)
        sj = meta.get("secondary_judge", {})
        judges_seen.setdefault(jd, set()).add(json.dumps({k: sj.get(k) for k in ("backend", "model", "decoding", "max_new_tokens")
                                                          if k in sj}, sort_keys=True))
        h, b = cnt["harmful"], cnt["benign"]
        out[f"{dec}|{ep}|{m}|{jd}"] = {"counts": cnt, "ids": ids}
        L.append(f"| {dec} | {ep} | {m} | {jd} | {h['delivered']} | {h['primary_unsafe']} | {h['secondary_unsafe']} | "
                 f"{h['both_unsafe']} | {h['primary_only']} | {h['secondary_only']} | {h['secondary_unknown']}/{b['secondary_unknown']} | "
                 f"{b['delivered']} | {b['disagreements']} |")
    tot = {}
    for k, v in out.items():
        dec, _, _, jd = k.split("|")
        t = tot.setdefault((dec, jd), [0, 0, 0])
        t[0] += 1; t[1] += v["counts"]["harmful"]["disagreements"]; t[2] += v["counts"]["benign"]["disagreements"]
    L += ["", "Totals (runs, harmful disagreements, benign disagreements): " +
          "; ".join(f"{d} {j}: {t}" for (d, j), t in sorted(tot.items())), "", "Secondary judge settings seen:"]
    for jd, s in judges_seen.items():
        L.append(f"- {jd}: " + " || ".join(sorted(s)))
    b = HERE.parent / "build"
    (b / "second_judge_local.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "second_judge_local.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
