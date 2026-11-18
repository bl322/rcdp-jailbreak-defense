"""Build every data file used by the experiments from the public sources, and check it against the files the
experiments read.

    python scripts/prepare_data.py --download          # fetch the public sources into data/raw, then build
    python scripts/prepare_data.py                     # build from files already in data/raw

Run from the repository root. Steps (each existing script is run unchanged):

  1. JailbreakBench + HarmBench  -> data/paper_splits/        scripts/prepare_main_benchmark_splits.py
     (400-example pool, 196-example calibration pool, 204-example held-out partition, its 149 harmful prompts)
  2. XSTest safe prompts         -> data/external/xstest_safe.jsonl                    scripts/prepare_xstest.py
  3. XSTest split 150 / 100: ordered by SHA-256 of the example id, the first 150 join the calibration pool
     (data/paper_splits/calibration_v2_external_benign.jsonl, 346 examples) and the other 100 form the test half
  4. mixed single-turn sets: the 149 held-out harmful prompts followed by the 250 (or the 100) XSTest prompts
  5. synthetic multi-turn sessions (588 calibration, 612 held-out) -> data/generated/
                                                                    scripts/generate_multiturn_quant_dataset.py
  6. external dialogues (ActorAttack / SafeMTData, MS-XSB, MT-Bench-101; MHJ if present) -> data/external_mt/
                                                                    scripts/external_mt_prepare.py
  7. OR-Bench-hard-1k (1,319 prompts) and its 500-prompt subset: ordered by SHA-256 of "orbench-r4:<id>", first 500

Then the SHA-256 of every output is compared with EXPECTED. "recorded" hashes come from the result files or the
server records of the runs; "rebuilt" files have no recorded hash (their example ids and order were checked against
the result records when this script was written). The script exits with status 1 if a recorded hash differs.

MHJ (ScaleAI/mhj) is gated and is not downloaded: request access, put harmbench_behaviors.csv into data/raw as
MHJ_harmbench_behaviors.csv, and run the script again. Without it the MHJ files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# public sources: file name in data/raw -> (URL, SHA-256 of the copy used)
SOURCES = {
    "benign-behaviors.csv": ("https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/"
                             "benign-behaviors.csv",
                             "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468"),
    "harmful-behaviors.csv": ("https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/"
                              "harmful-behaviors.csv",
                              "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1"),
    "harmbench_behaviors_text_all.csv": ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
                                         "behavior_datasets/harmbench_behaviors_text_all.csv",
                                         "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"),
    "xstest_prompts.csv": ("https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
                           "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"),
    "or-bench-hard-1k.csv": ("https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv",
                             "a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d"),
    "Attack_600.json": ("https://huggingface.co/datasets/SafeMTData/SafeMTData/resolve/main/SafeMTData/Attack_600.json",
                        "58222f797df19e4e008575f8f74ad5865be914bc07952288b02c1e934b564159"),
    "MS-XSB_prompts.csv": ("https://huggingface.co/datasets/ScaDSAI/XSB_and_MS-XSB/resolve/main/MS-XSB_prompts.csv",
                           "145dca9bef23fd6ad1c1eba943120bc25528e6a5c8689c1fd66142b135a7eca0"),
    "mtbench101.jsonl": ("https://raw.githubusercontent.com/mtbench101/mt-bench-101/main/data/subjective/"
                         "mtbench101.jsonl",
                         "1a1e433acf7e2ef013b5001b60fd8e48bc3e4202460727e25b2ea8575aae7d80"),
}

# output (relative to the repository root) -> (SHA-256, "recorded" | "rebuilt")
EXPECTED = {
    "data/paper_splits/full_pool_400.jsonl":
        ("741c192f483b8733733b419e3e2a408ae1a954e10655af94b29754e33e8c7684", "rebuilt"),
    "data/paper_splits/full_pool_calibration.jsonl":
        ("85af40b255833c88ade1a1edb4e9b0c6a6c910e5fc56a6faf6675dadfbf68241", "recorded"),
    "data/paper_splits/heldout_204.jsonl":
        ("33df979c376683888fd26aa4a83a71d57da6f0fb49df1c2588562002d566dd9f", "recorded"),
    "data/paper_splits/heldout_harmful_149.jsonl":
        ("6b88b5dcccc9664e81e1b92108f5d4022e52abf4b8062a577534bf500071721b", "rebuilt"),
    "data/external/xstest_safe.jsonl":
        ("5d9c04a549e61e4e5f6a81ecc74b30e772e15863c5f6099c408b02b528a35146", "recorded"),
    "data/external/xstest_safe_calib150.jsonl":
        ("c50fccadc3e065bb1b6176db76c14705d1ec07622b603a3061ddb08b891236fd", "recorded"),
    "data/external/xstest_safe_test100.jsonl":
        ("9350d7fa7ff17ef35e31e9a742753625cafbfa56f6f83f7c40d99bfb9ec698a9", "recorded"),
    "data/paper_splits/calibration_v2_external_benign.jsonl":
        ("7a8a9af7de906b080c31cb83ddea568c88dd1af9f45b821ff97d4859b5e9b734", "recorded"),
    "data/external/mixed_harm149_xstest250.jsonl":
        ("0dd10529af32542fd72e872853811fef63db0bb0af97ca4230a9259915bdd10f", "rebuilt"),
    "data/paper_splits/mixed_harm149_xstest_test100.jsonl":
        ("1444cd7d5a4f07086f52b760f4051316b70c307a0fcae37900e1b36a9c533cfd", "rebuilt"),
    "data/generated/multiturn_calibration_588_r1.jsonl":
        ("6b4711c8d25a3e3c0a01cc863c4ae1ec0967872ce6af4ae193e477f5c857629e", "rebuilt"),
    "data/generated/multiturn_heldout_612_r1.jsonl":
        ("38bc6f5a126c590ddc055e59df94851b14d7816e778b1b33f9a2f2cc83438fb5", "recorded"),
    "data/external_mt/attack600_heldout.jsonl":
        ("fa69b5c85081455eb0de36f3c1cd111be942747aa9cf42e8f49469935e72286f", "recorded"),
    "data/external_mt/msxsb.jsonl":
        ("5f24477a3f20c6855759c6c16c18705d9094dbde931e9916751677412c5d90ac", "recorded"),
    "data/external_mt/mtbench101.jsonl":
        ("6129a76fd681ea0d6ebda7bebb2b6ca7781b22818adfeb6d443436dcf4b85e00", "recorded"),
    "data/external_mt/benign.jsonl":
        ("5ea39b3bfc5f6c23b7fdb6a74152170189b94d5f50a0fece9e118bccc3c03a01", "recorded"),
    "data/external_mt/mixed_attack600.jsonl":
        ("412503a29600d0c3a4895b73d423c33ea6cf8866f15582812b2179efda24dfd7", "recorded"),
    "data/external_mt/mhj_kept.jsonl":
        ("4578b18e02d5332d9760adce1bcc0855d5f7dfc96f543a883295e814fc89b6a6", "recorded"),
    "data/external_mt/mixed_mhj.jsonl":
        ("eaa21070588f1e4d4af129c97b0736b7a69f36d4183cf688ae332820b3835e94", "recorded"),
    "data/external/orbench_hard_1k.jsonl":
        ("41d3cc917adca12ab18f620685a4eaa5c16fead5fe71c9684594a2457375bc5a", "recorded"),
    "data/external/orbench_hard_sub500.jsonl":
        ("cb06ab550bf2270efbd81ceee24d6a3fe53bc69e5816a8d1bf31a413095cbf43", "recorded"),
}
OPTIONAL = {"data/external_mt/mhj_kept.jsonl", "data/external_mt/mixed_mhj.jsonl"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str) -> None:
    print("+ python", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))


def to_lf(path: Path) -> None:
    """The generator writes in text mode; on Windows that gives CRLF. The experiments read LF files."""
    b = path.read_bytes()
    if b"\r\n" in b:
        path.write_bytes(b.replace(b"\r\n", b"\n"))


def download(raw: Path, allow_unpinned: bool) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, _) in SOURCES.items():
        if (raw / name).exists():
            continue
        print("download", url, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "rcdp-prepare-data"})
        with urllib.request.urlopen(req, timeout=120) as r:
            (raw / name).write_bytes(r.read())
    check_raw(raw, allow_unpinned)


def check_raw(raw: Path, allow_unpinned: bool) -> None:
    missing = [n for n in SOURCES if not (raw / n).exists()]
    if missing:
        raise SystemExit(f"missing in {raw}: {missing} (run with --download)")
    bad = [n for n, (_, h) in SOURCES.items() if sha256(raw / n) != h]
    if bad and not allow_unpinned:
        raise SystemExit(f"these source files differ from the copies used ({bad}); the upstream file may have "
                         "changed. Use --allow-unpinned to build anyway; the check below then shows the effect.")


def build(raw: Path) -> None:
    ps, ext, gen = ROOT / "data/paper_splits", ROOT / "data/external", ROOT / "data/generated"

    def rel(p: Path) -> str:
        try:
            return p.relative_to(ROOT).as_posix()
        except ValueError:
            return str(p)

    # 1. single-turn pool and splits
    run("scripts/prepare_main_benchmark_splits.py", "--benign-csv", rel(raw / "benign-behaviors.csv"),
        "--harmful-csv", rel(raw / "harmful-behaviors.csv"), "--harmbench-csv",
        rel(raw / "harmbench_behaviors_text_all.csv"), "--output-dir", "data/paper_splits")
    # 2. XSTest safe prompts (the label column of the official file holds safe / unsafe)
    run("scripts/prepare_xstest.py", "--source", rel(raw / "xstest_prompts.csv"),
        "--output", "data/external/xstest_safe.jsonl", "--type-column", "label")
    # 3. XSTest 150 / 100 and the 346-example calibration pool
    xs = read_jsonl(ext / "xstest_safe.jsonl")
    xs = sorted(xs, key=lambda r: hashlib.sha256(r["example_id"].encode("utf-8")).hexdigest())
    write_jsonl(ext / "xstest_safe_calib150.jsonl", xs[:150])
    write_jsonl(ext / "xstest_safe_test100.jsonl", xs[150:])
    write_jsonl(ps / "calibration_v2_external_benign.jsonl", read_jsonl(ps / "full_pool_calibration.jsonl") + xs[:150])
    # 4. mixed single-turn sets
    harm = read_jsonl(ps / "heldout_harmful_149.jsonl")
    write_jsonl(ext / "mixed_harm149_xstest250.jsonl", harm + read_jsonl(ext / "xstest_safe.jsonl"))
    write_jsonl(ps / "mixed_harm149_xstest_test100.jsonl", harm + xs[150:])
    # 5. synthetic multi-turn sessions
    gen.mkdir(parents=True, exist_ok=True)
    for seed, out in (("full_pool_calibration.jsonl", "multiturn_calibration_588_r1.jsonl"),
                      ("heldout_204.jsonl", "multiturn_heldout_612_r1.jsonl")):
        run("scripts/generate_multiturn_quant_dataset.py", "--seed", f"data/paper_splits/{seed}",
            "--output", f"data/generated/{out}", "--variants", "3")
        to_lf(gen / out)
    # 6. external dialogues
    args = ["scripts/external_mt_prepare.py", "--raw-dir", rel(raw), "--out-dir", "data/external_mt"]
    run(*args)
    # 7. OR-Bench-hard-1k and the 500-prompt subset
    with (raw / "or-bench-hard-1k.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ob = [{"example_id": "orbench-hard-%04d" % i, "benchmark": "or-bench-hard-1k", "label": "benign",
           "turns": [{"role": "user", "text": r["prompt"]}],
           "metadata": {"source": "bench-llm/or-bench", "split": "hard-1k", "category": r.get("category", ""),
                        "source_row": i}} for i, r in enumerate(rows)]
    write_jsonl(ext / "orbench_hard_1k.jsonl", ob)
    sub = sorted(ob, key=lambda r: hashlib.sha256(("orbench-r4:" + r["example_id"]).encode("utf-8")).hexdigest())
    write_jsonl(ext / "orbench_hard_sub500.jsonl", sub[:500])


def check() -> int:
    status = 0
    print(f"\n{'file':58s} {'check':9s} result")
    for rel_path, (want, kind) in EXPECTED.items():
        p = ROOT / rel_path
        if not p.exists():
            res = "skipped (MHJ not present)" if rel_path in OPTIONAL else "MISSING"
            status |= rel_path not in OPTIONAL
        elif sha256(p) == want:
            res = "identical"
        else:
            res = "DIFFERS"
            status |= kind == "recorded"
        print(f"{rel_path:58s} {kind:9s} {res}")
    print("\nall recorded files identical" if status == 0 else "\nsome files differ from the recorded ones")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--download", action="store_true", help="fetch missing public source files into --raw-dir")
    ap.add_argument("--allow-unpinned", action="store_true", help="build even if a source file differs")
    ap.add_argument("--check-only", action="store_true", help="only compare existing outputs with EXPECTED")
    a = ap.parse_args()
    raw = a.raw_dir.resolve()
    if not a.check_only:
        if a.download:
            download(raw, a.allow_unpinned)
        else:
            check_raw(raw, a.allow_unpinned)
        build(raw)
    return check()


if __name__ == "__main__":
    sys.exit(main())
