# -*- coding: utf-8 -*-
"""Lay out the unpacked result package as the analysis scripts expect it, in a new work directory.

    python analysis/setup_layout.py <unpacked rcdp_r1_results_redacted> <new work dir>

Creates
  <work>/_r1_out/<original path>          every result file under each of its original paths (MANIFEST.json), after
                                          checking its published SHA-256;
  <work>/_r1_out/ORIGINAL_SHA256.json     original path -> SHA-256 of the unredacted file (read by file_sha256());
  <work>/ccfa-workfiles/tables, figures   copies of analysis/tables and analysis/figures;
  <work>/ccfa-workfiles/code/rcdp-r1      copies of rcdp_guard/ and data/ (verify_splits.py imports rcdp_guard);
  <work>/ccfa-workfiles/manuscript/rcdp-cmc-r1/figures   (the figure scripts copy their PDF there).
Then run a script from the work directory, for example
    python <work>/ccfa-workfiles/tables/single-turn/source/single_turn_stats.py
Each script writes its outputs to the build/ directory next to its source/ directory.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

PKG, WORK = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[1]
if WORK.exists():
    raise SystemExit(f"{WORK} exists")
man = json.loads((PKG / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
orig_sha = {}
for r in man:
    src = PKG / r["published"]
    data = src.read_bytes()
    if hashlib.sha256(data).hexdigest() != r["published_sha256"]:
        raise SystemExit(f"hash mismatch: {r['published']}")
    for o in [r["original"]] + r.get("identical_copies", []):
        dst = WORK / "_r1_out" / o
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        orig_sha[o] = r["original_sha256"]
(WORK / "_r1_out" / "ORIGINAL_SHA256.json").write_text(json.dumps(orig_sha, indent=1, sort_keys=True), encoding="utf-8")
wf = WORK / "ccfa-workfiles"
for part in ("tables", "figures"):
    shutil.copytree(REPO / "analysis" / part, wf / part)
for part in ("rcdp_guard", "data"):
    shutil.copytree(REPO / part, wf / "code" / "rcdp-r1" / part, ignore=shutil.ignore_patterns("__pycache__"))
(wf / "manuscript" / "rcdp-cmc-r1" / "figures").mkdir(parents=True)
print(f"{len(orig_sha)} result paths from {len(man)} files under {WORK / '_r1_out'}; scripts under {wf}")
