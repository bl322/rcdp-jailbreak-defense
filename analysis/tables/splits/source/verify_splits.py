# -*- coding: utf-8 -*-
"""Supplementary Table S1: the four subsets of the single-turn protocol, recounted from the result files and
recomputed with the released split code.

  * held-out partition (204) and calibration pool (196): the record ids of every e2e result file, and the
    deterministic SHA-256 rule of the appendix (within each label, sort by int(SHA256(id)[:16], 16), id;
    the first 149 harmful and 55 benign are held out)
  * D_head / D_platt / D_thresh: calibration_report.split.ids of every e2e result file, and
    rcdp_guard.calibration_protocol.three_way_split (seed rcdp-r1) run on the calibration ids

Writes the released id file (code/rcdp-r1/data/paper_splits/split_ids_r1.json) and build/splits_r1.{md,json}
and build/table_s1_rows.tex. Imports project code, so it is not run with -I; data paths are arguments.

    python -X utf8 verify_splits.py <project root>
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
CODE = ROOT / "ccfa-workfiles/code/rcdp-r1"
sys.path.insert(0, str(CODE))
from rcdp_guard.calibration_protocol import DEFAULT_SEED, three_way_split  # noqa: E402

E2E = sorted((ROOT / "_r1_out/e2e_rcdp/extracted/results/r1/e2e").glob("*.json"))
E2E += sorted((ROOT / "_r1_out/e2e_r3").glob("loc_*_rcdp_*.json"))
EXPECT_DATA = {"evaluation": "33df979c376683888fd26aa4a83a71d57da6f0fb49df1c2588562002d566dd9f",
               "calibration": "85af40b255833c88ade1a1edb4e9b0c6a6c910e5fc56a6faf6675dadfbf68241"}


def digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def main():
    assert DEFAULT_SEED == "rcdp-r1"
    held, subsets, meta, used = None, None, {}, []
    for f in E2E:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("datasets", {}).get("evaluation", {}).get("n") != 204:
            continue  # recheck files (no records) and the 30-prompt probes
        ds = d["datasets"]
        assert ds["evaluation"]["sha256"] == EXPECT_DATA["evaluation"], f.name
        assert ds["calibration"]["sha256"] == EXPECT_DATA["calibration"], f.name
        ids = {r["example_id"]: (r["label"], r["benchmark"]) for r in d["records"]}
        assert len(ids) == 204
        held = held or ids
        assert ids == held, f.name
        cr = d.get("calibration_report")
        if not cr:
            continue  # baselines without a calibrated controller
        assert cr["protocol"] == "r1" and cr["seed"] == "rcdp-r1", f.name
        sp = {k: list(v) for k, v in cr["split"]["ids"].items()}
        subsets = subsets or sp
        assert {k: sorted(v) for k, v in sp.items()} == {k: sorted(v) for k, v in subsets.items()}, f.name
        rec = {}
        for r in cr["calibration_records"]:
            rec[r["example_id"].split(":turn-")[0]] = (r["subset"], r["label"], r["benchmark"])
        assert len(rec) == 196
        assert all(rec[i][0] == k for k, v in sp.items() for i in v), f.name
        for i, (_, lab, bm) in rec.items():
            meta.setdefault(i, (lab, bm))
            assert meta[i] == (lab, bm)
        used.append(f.name)
    assert len(used) >= 9, used
    assert sorted(len(v) for v in subsets.values()) == [48, 48, 100]
    cal = set(meta)
    assert cal.isdisjoint(held) and len(cal) == 196

    # recompute the three-way split with the released code
    exs = [SimpleNamespace(example_id=i, label=SimpleNamespace(value=lab), benchmark=bm) for i, (lab, bm) in meta.items()]
    parts = three_way_split(exs, seed="rcdp-r1")
    for name, part in zip(("head", "platt", "thresh"), parts):
        assert sorted(e.example_id for e in part) == sorted(subsets[name]), name

    # recompute the held-out partition with the appendix rule
    allx = {**{i: lb for i, lb in held.items()}, **meta}
    assert len(allx) == 400
    rule_held = set()
    for lab, k in (("harmful", 149), ("benign", 55)):
        group = sorted((i for i, (l_, _) in allx.items() if l_ == lab),
                       key=lambda i: (int(hashlib.sha256(i.encode("utf-8")).hexdigest()[:16], 16), i))
        rule_held |= set(group[:k])
    assert rule_held == set(held), (len(rule_held ^ set(held)))

    comp = {}
    order = [("head", subsets["head"]), ("platt", subsets["platt"]), ("thresh", subsets["thresh"]), ("heldout", list(held))]
    lookup = {**held, **meta}
    for name, ids in order:
        c = {"n": len(ids)}
        for i in ids:
            lab, bm = lookup[i]
            key = f"{lab}/{bm}"
            c[key] = c.get(key, 0) + 1
        c["sha256"] = digest(ids)
        comp[name] = c
    exp = {"head": (26, 51, 23), "platt": (12, 25, 11), "thresh": (13, 24, 11)}
    for k, (a, b, c) in exp.items():
        assert (comp[k].get("harmful/JailbreakBench"), comp[k].get("harmful/HarmBench"), comp[k].get("benign/JailbreakBench")) == (a, b, c), comp[k]
    tot = {key: sum(comp[s].get(key, 0) for s in comp) for key in ("harmful/JailbreakBench", "harmful/HarmBench", "benign/JailbreakBench")}
    assert tot == {"harmful/JailbreakBench": 100, "harmful/HarmBench": 200, "benign/JailbreakBench": 100}, tot

    release = {"description": "Example identifiers of the four subsets of the single-turn protocol (revision R1). "
                              "Held-out partition: SHA-256 rule of the appendix. D_head / D_platt / D_thresh: "
                              "rcdp_guard.calibration_protocol.three_way_split, seed rcdp-r1. Each list is sorted; "
                              "sha256 is the digest of the newline-joined sorted list.",
               "datasets": {"data/paper_splits/heldout_204.jsonl": EXPECT_DATA["evaluation"],
                            "data/paper_splits/full_pool_calibration.jsonl": EXPECT_DATA["calibration"]},
               "subsets": {name: {"n": len(ids), "sha256": digest(ids), "ids": sorted(ids)} for name, ids in order}}
    out = CODE / "data/paper_splits/split_ids_r1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(release, indent=1) + "\n", encoding="utf-8")
    fsha = hashlib.sha256(out.read_bytes()).hexdigest()

    label = {"head": r"$\mathcal{D}_{\mathrm{head}}$ (risk head)", "platt": r"$\mathcal{D}_{\mathrm{platt}}$ (Platt map)",
             "thresh": r"$\mathcal{D}_{\mathrm{thresh}}$ (thresholds)", "heldout": "Held-out partition (evaluation)"}
    rows = []
    for name, _ in order:
        c = comp[name]
        rows.append(f"{label[name]} & {c['n']} & {c.get('harmful/JailbreakBench', 0)} & {c.get('harmful/HarmBench', 0)} & "
                    f"{c.get('benign/JailbreakBench', 0)} & \\texttt{{{c['sha256'][:16]}}} \\\\")
    (HERE.parent / "build").mkdir(exist_ok=True)
    (HERE.parent / "build" / "table_s1_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (HERE.parent / "build" / "splits_r1.json").write_text(json.dumps({"composition": comp, "files_checked": used,
                                                                        "release_file_sha256": fsha}, indent=1), encoding="utf-8")
    md = ["# Split identifiers (recounted)", "", f"- result files checked: {len(used)} ({', '.join(used)})",
          "- three_way_split(seed rcdp-r1) on the calibration ids reproduces D_head/D_platt/D_thresh: yes",
          "- the SHA-256 held-out rule on the 400 ids reproduces the held-out partition: yes",
          f"- release file sha256: {fsha}"]
    md += [f"- {k}: {v}" for k, v in comp.items()]
    (HERE.parent / "build" / "splits_r1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print("\n".join(rows))


if __name__ == "__main__":
    main()
