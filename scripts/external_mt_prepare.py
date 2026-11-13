"""Build the independent multi-turn evaluation sets (ledger R3-4) from the raw public files.

Inputs (in --raw-dir; sha256 pinned below, checked unless --allow-unpinned):
  Attack_600.json                    SafeMTData (ActorAttack, Ren et al. 2024), MIT
  MS-XSB_prompts.csv                 MS-XSB (arXiv:2510.08158), MIT
  mtbench101.jsonl                   MT-Bench-101 (ACL 2024), Apache-2.0
  harmbench_behaviors_text_all.csv   HarmBench behaviour table, MIT (maps Attack_600 queries to BehaviorIDs)
  MHJ_harmbench_behaviors.csv        optional; MHJ (Li et al. 2024, ScaleAI/mhj, gated: obtained by the authors under
                                     the dataset terms and never redistributed). Built only if the file is present.

Rules fixed before any result was seen (authors' decision 2026-11-14):
  * Attack_600: its 200 queries are the 200 HarmBench standard behaviours, i.e. our 200 HarmBench seeds.
    Only rows whose behaviour is a held-out seed are kept (100 behaviours x 3 actors = 300 dialogues).
    cluster_id = "hb-<BehaviorID>", the seed's own id, so run_multiturn_cases.py refuses the file if a
    calibration seed slips in. Every user turn is kept; there are no assistant turns.
  * MS-XSB: one dialogue per scenario; the scenario description is the first user turn, followed by
    prompt1..prompt20 (empty prompts skipped). Label benign.
  * MT-Bench-101: every dialogue; history {user, bot} -> user / assistant turns; the final assistant
    turn is dropped (the controller routes user turns). Label benign.
  * No subsampling and no filtering on content.
  * MHJ (fixed 2026-11-14, before any MHJ result): question_id is the spreadsheet row of the HarmBench table
    (header = row 1), i.e. behaviour = data row question_id - 2. Checked on the pinned files without reading
    any score: mean keyword coverage of the behaviour text by the conversation is 0.580 at offset -2 against
    0.051-0.125 at offsets -5..-1, and the labelled row ranks first among the 400 behaviours for 377 of the
    537 conversations; the script repeats the offset check and stops unless -2 is the clear maximum.
    Conversations on calibration behaviours are dropped; all others are kept (held-out standard behaviours
    and contextual behaviours, which are in neither seed half), whatever their number of turns.
    cluster_id = "hb-<BehaviorID>". The system prompt is dropped; the file holds no model replies, so every
    turn is a user turn. These attacks were written against other defended models (column Source).

Outputs (in --out-dir): attack600_heldout.jsonl, msxsb.jsonl, mtbench101.jsonl, benign.jsonl,
mixed_attack600.jsonl (attack + benign), and if MHJ is present mhj_kept.jsonl and mixed_mhj.jsonl (MHJ + the
same benign dialogues); manifest.json records counts and sha256 of inputs and outputs.

    python scripts/external_mt_prepare.py --raw-dir data/external_mt/raw \
        --heldout data/paper_splits/heldout_204.jsonl --calibration data/paper_splits/full_pool_calibration.jsonl \
        --out-dir data/external_mt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PINNED = {
    "Attack_600.json": "58222f797df19e4e008575f8f74ad5865be914bc07952288b02c1e934b564159",
    "MS-XSB_prompts.csv": "145dca9bef23fd6ad1c1eba943120bc25528e6a5c8689c1fd66142b135a7eca0",
    "mtbench101.jsonl": "1a1e433acf7e2ef013b5001b60fd8e48bc3e4202460727e25b2ea8575aae7d80",
    "harmbench_behaviors_text_all.csv": "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
}
SOURCES = {
    "Attack_600.json": "https://huggingface.co/datasets/SafeMTData/SafeMTData (SafeMTData/Attack_600.json), MIT",
    "MS-XSB_prompts.csv": "https://huggingface.co/datasets/ScaDSAI/XSB_and_MS-XSB (MS-XSB_prompts.csv), MIT",
    "mtbench101.jsonl": "https://github.com/mtbench101/mt-bench-101 (data/subjective/mtbench101.jsonl), Apache-2.0",
    "harmbench_behaviors_text_all.csv": "https://github.com/centerforaisafety/HarmBench "
                                        "(data/behavior_datasets/harmbench_behaviors_text_all.csv), MIT",
}
OPTIONAL = {"MHJ_harmbench_behaviors.csv": "97e0dc879b66c7b664deb2a9c2e5bd78b647b84e16a9a43ba95d0117e3b35e46"}
SOURCES["MHJ_harmbench_behaviors.csv"] = ("https://huggingface.co/datasets/ScaleAI/mhj (harmbench_behaviors.csv, "
                                          "gated; not redistributed)")
MHJ_OFFSET = -2
_STOP = set("about after again against could would should their there these those which while where with within "
            "without write detailed instructions give generate describe explain create provide using someone other "
            "being".split())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def seed_ids(path: Path) -> set:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["example_id"]))
    return ids


def record(eid, benchmark, label, turns, cluster, source, source_id, **extra):
    n_user = sum(t["role"] == "user" for t in turns)
    md = {"cluster_id": cluster, "source": source, "source_id": str(source_id), "n_user_turns": n_user}
    md.update(extra)
    return {"example_id": eid, "benchmark": benchmark, "label": label, "turns": turns, "metadata": md}


def build_attack600(raw: Path, held_hb: set, cal_hb: set, allow_unmapped: bool):
    table = list(csv.DictReader(open(raw / "harmbench_behaviors_text_all.csv", encoding="utf-8")))
    text2id = {norm(r["Behavior"]): r["BehaviorID"] for r in table}
    rows = json.loads((raw / "Attack_600.json").read_text(encoding="utf-8"))
    out, stats = [], Counter()
    for r in rows:
        bid = text2id.get(norm(r["plain_query"]))
        if bid is None:
            stats["unmapped"] += 1
            continue
        if bid in cal_hb:
            stats["excluded_calibration_behaviour"] += 1
            continue
        if bid not in held_hb:
            stats["excluded_not_a_seed"] += 1
            continue
        turns = [{"role": "user", "text": str(q).strip()} for q in r["multi_turn_queries"] if str(q).strip()]
        out.append(record(f"ActorAttack-{r['id']}", "ActorAttack", "harmful", turns, f"hb-{bid}",
                          "SafeMTData/Attack_600", r["id"], behavior_id=bid, query_id=r["query_id"],
                          category=r.get("category")))
        stats["kept"] += 1
    if stats["unmapped"] and not allow_unmapped:
        raise SystemExit(f"Attack_600: {stats['unmapped']} plain_query values not found in the HarmBench table")
    return out, dict(stats)


def build_msxsb(raw: Path):
    rows = list(csv.DictReader(open(raw / "MS-XSB_prompts.csv", encoding="utf-8-sig")))
    pcols = sorted((c for c in rows[0] if re.fullmatch(r"prompt\d+", c)), key=lambda c: int(c[6:]))
    out = []
    for r in rows:
        texts = [r["scenario"]] + [r[c] for c in pcols]
        turns = [{"role": "user", "text": str(t).strip()} for t in texts if str(t or "").strip()]
        eid = f"MS-XSB-{r['id']}"
        out.append(record(eid, "MS-XSB", "benign", turns, eid, "ScaDSAI/XSB_and_MS-XSB:MS-XSB_prompts.csv", r["id"],
                          first_turn="scenario"))
    return out


def build_mtbench101(raw: Path):
    out = []
    for line in (raw / "mtbench101.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        turns = []
        for h in d["history"]:
            for role, key in (("user", "user"), ("assistant", "bot")):
                text = str(h.get(key) or "").strip()
                if text:
                    turns.append({"role": role, "text": text})
        while turns and turns[-1]["role"] != "user":
            turns.pop()
        eid = f"MT-Bench-101-{d['id']}"
        out.append(record(eid, "MT-Bench-101", "benign", turns, eid, "mtbench101/mt-bench-101", d["id"],
                          task=d.get("task")))
    return out


def _kw(text) -> set:
    return {w for w in re.findall(r"[a-z]{5,}", str(text).lower()) if w not in _STOP}


def mhj_offset_check(conv, table):
    """Mean keyword coverage of the behaviour text by each conversation for every row offset in -5..0 that
    keeps all question_ids inside the table. conv = [(question_id, keyword set)]."""
    bw = [_kw(r["Behavior"]) for r in table]
    qs = [q for q, _ in conv]
    res = {}
    for off in range(-5, 1):
        if min(qs) + off < 0 or max(qs) + off >= len(table):
            continue
        vals = [len(bw[q + off] & w) / len(bw[q + off]) for q, w in conv if bw[q + off]]
        res[off] = sum(vals) / len(vals) if vals else 0.0
    return res


def build_mhj(path: Path, raw: Path, held_hb: set, cal_hb: set):
    csv.field_size_limit(10 ** 9)
    table = list(csv.DictReader(open(raw / "harmbench_behaviors_text_all.csv", encoding="utf-8")))
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    mcols = sorted((c for c in rows[0] if re.fullmatch(r"message_\d+", c)), key=lambda c: int(c[8:]))
    parsed = []
    for k, r in enumerate(rows):
        msgs = [json.loads(r[c]) for c in mcols if str(r[c] or "").strip()]
        users = [str(m.get("body") or "").strip() for m in msgs if str(m.get("role")).lower() == "user"]
        other = {str(m.get("role")).lower() for m in msgs} - {"user", "system"}
        if other:
            raise SystemExit(f"MHJ row {k}: unexpected roles {sorted(other)}")
        parsed.append((k, r, [u for u in users if u]))
    q_ids = [int(r["question_id"]) for _, r, _ in parsed]
    if min(q_ids) + MHJ_OFFSET < 0 or max(q_ids) + MHJ_OFFSET >= len(table):
        raise SystemExit("MHJ: question_id outside the HarmBench table under the row mapping question_id - 2")
    check = mhj_offset_check([(q, _kw(" ".join(u) + " " + r.get("submission_message", "")))
                              for q, (_, r, u) in zip(q_ids, parsed)], table)
    others = [v for off, v in check.items() if off != MHJ_OFFSET]
    if MHJ_OFFSET not in check or (others and check[MHJ_OFFSET] < 2 * max(others)):
        raise SystemExit(f"MHJ: row-mapping check failed, coverage by offset {check}")
    out, stats = [], Counter()
    for q, (k, r, users) in zip(q_ids, parsed):
        b = table[q + MHJ_OFFSET]
        bid, cat = b["BehaviorID"], b["FunctionalCategory"]
        if bid in cal_hb:
            stats["excluded_calibration_behaviour"] += 1
            continue
        if not users:
            stats["excluded_no_user_turn"] += 1
            continue
        group = "heldout" if bid in held_hb else "neither"
        stats[f"kept_{group}_{cat}"] += 1
        out.append(record(f"MHJ-{k}", "MHJ", "harmful", [{"role": "user", "text": u} for u in users], f"hb-{bid}",
                          "ScaleAI/mhj:harmbench_behaviors.csv", k, behavior_id=bid, question_id=q,
                          functional_category=cat, seed_group=group, tactic=r.get("tactic"),
                          target_source=r.get("Source"), temperature=r.get("temperature")))
    return out, dict(stats), {str(k): round(v, 4) for k, v in sorted(check.items())}


def write(path: Path, rows) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sha256(path)


def summary(rows):
    return {"dialogues": len(rows), "clusters": len({r["metadata"]["cluster_id"] for r in rows}),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "user_turns_total": sum(r["metadata"]["n_user_turns"] for r in rows),
            "user_turns_hist": dict(sorted(Counter(r["metadata"]["n_user_turns"] for r in rows).items()))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", default="data/external_mt/raw")
    p.add_argument("--heldout", default="data/paper_splits/heldout_204.jsonl")
    p.add_argument("--calibration", default="data/paper_splits/full_pool_calibration.jsonl")
    p.add_argument("--out-dir", default="data/external_mt")
    p.add_argument("--allow-unpinned", action="store_true", help="accept raw files whose sha256 differs (tests only)")
    p.add_argument("--allow-unmapped", action="store_true", help="drop Attack_600 rows not in the HarmBench table")
    p.add_argument("--expect-attack", type=int, default=300, help="expected held-out Attack_600 dialogues (0 = no check)")
    p.add_argument("--expect-mhj", type=int, default=384, help="expected kept MHJ dialogues if MHJ is present (0 = no check)")
    args = p.parse_args(argv)
    raw, out_dir = Path(args.raw_dir), Path(args.out_dir)

    hashes = {}
    for name, want in list(PINNED.items()) + list(OPTIONAL.items()):
        f = raw / name
        if not f.exists():
            if name in OPTIONAL:
                print(f"{name} not in {raw}: MHJ sets not built")
                continue
            raise SystemExit(f"missing raw file: {f}")
        hashes[name] = sha256(f)
        if hashes[name] != want and not args.allow_unpinned:
            raise SystemExit(f"{name}: sha256 {hashes[name]} differs from the pinned {want}")

    held, cal = seed_ids(Path(args.heldout)), seed_ids(Path(args.calibration))
    held_hb = {i[3:] for i in held if i.startswith("hb-")}
    cal_hb = {i[3:] for i in cal if i.startswith("hb-")}
    if held_hb & cal_hb:
        raise SystemExit("held-out and calibration seed files share HarmBench behaviours")

    attack, astats = build_attack600(raw, held_hb, cal_hb, args.allow_unmapped)
    if args.expect_attack and len(attack) != args.expect_attack:
        raise SystemExit(f"Attack_600: {len(attack)} held-out dialogues, expected {args.expect_attack} ({astats})")
    msxsb, mtb = build_msxsb(raw), build_mtbench101(raw)
    benign = msxsb + mtb
    mhj, mstats, mcheck = None, None, None
    if "MHJ_harmbench_behaviors.csv" in hashes:
        mhj, mstats, mcheck = build_mhj(raw / "MHJ_harmbench_behaviors.csv", raw, held_hb, cal_hb)
        if args.expect_mhj and len(mhj) != args.expect_mhj:
            raise SystemExit(f"MHJ: {len(mhj)} kept dialogues, expected {args.expect_mhj} ({mstats})")
    ids = [r["example_id"] for r in attack + benign + (mhj or [])]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate example_id values")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "attack600_heldout.jsonl": attack, "msxsb.jsonl": msxsb, "mtbench101.jsonl": mtb,
        "benign.jsonl": benign, "mixed_attack600.jsonl": attack + benign,
    }
    if mhj is not None:
        outputs.update({"mhj_kept.jsonl": mhj, "mixed_mhj.jsonl": mhj + benign})
    all_pins = dict(PINNED, **OPTIONAL)
    manifest = {
        "script": "scripts/external_mt_prepare.py",
        "inputs": {n: {"sha256": h, "pinned": h == all_pins[n], "source": SOURCES[n]} for n, h in hashes.items()},
        "seeds": {"heldout_harmbench": len(held_hb), "calibration_harmbench": len(cal_hb)},
        "attack600_filter": astats,
        "mhj": "absent" if mhj is None else {"filter": mstats, "row_mapping": "question_id - 2",
                                              "keyword_coverage_by_offset": mcheck},
        "rules": __doc__.split("Rules fixed before any result was seen", 1)[1].split("Outputs (in", 1)[0].strip(),
        "outputs": {},
    }
    for name, rows in outputs.items():
        manifest["outputs"][name] = {"sha256": write(out_dir / name, rows), **summary(rows)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, info in manifest["outputs"].items():
        print(f"{name:26s} {info['dialogues']:5d} dialogues  {info['clusters']:5d} clusters  "
              f"{info['user_turns_total']:6d} user turns  {info['by_label']}")
    print("Attack_600 filter:", astats)
    if mhj is not None:
        print("MHJ filter:", mstats, "| keyword coverage by row offset:", mcheck)
    return 0


if __name__ == "__main__":
    sys.exit(main())
