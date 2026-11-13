"""Pre-flight check on the server before the R1 runs: environment, data files, code version."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = {
    "data/paper_splits/full_pool_calibration.jsonl": 196,
    "data/paper_splits/heldout_204.jsonl": 204,
    "data/external/mixed_harm149_xstest250.jsonl": 399,
    "data/paper_splits/calibration_v2_external_benign.jsonl": 346,
    "data/paper_splits/mixed_harm149_xstest_test100.jsonl": 249,
}
CAL = "data/paper_splits/full_pool_calibration.jsonl"
CAL_V2 = "data/paper_splits/calibration_v2_external_benign.jsonl"
HELDOUT = "data/paper_splits/heldout_204.jsonl"
MIXED399 = "data/external/mixed_harm149_xstest250.jsonl"
TEST249 = "data/paper_splits/mixed_harm149_xstest_test100.jsonl"
# (calibration pool, evaluation set) pairs that r1_stage1_run.sh actually runs. Only these must be disjoint:
# the 346 pool contains 150 XSTest prompts of the 399 set by construction, so the 346 pool is evaluated on
# the 249 set (149 harmful + the 100 XSTest prompts outside the pool) and never on the 399 set.
PAIRS = [(CAL, HELDOUT), (CAL, MIXED399), (CAL, TEST249), (CAL_V2, HELDOUT), (CAL_V2, TEST249)]
OPTIONAL = (CAL_V2, TEST249)  # only needed for the 196-vs-346 pool contrast


def main() -> int:
    ok = True
    print("python", sys.version.split()[0], "|", platform.platform())
    for mod in ("numpy", "torch", "transformers"):
        try:
            m = importlib.import_module(mod)
            extra = ""
            if mod == "torch":
                extra = f" cuda={m.cuda.is_available()}" + (f" gpu={m.cuda.get_device_name(0)}" if m.cuda.is_available() else "")
            print(f"{mod} {m.__version__}{extra}")
        except Exception as exc:  # noqa: BLE001
            print(f"{mod} MISSING ({exc})")
            ok = ok and mod != "numpy"
    import rcdp_guard.calibration_protocol as cp

    h = hashlib.sha256()
    for f in sorted((ROOT / "rcdp_guard").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    print("rcdp_guard sha256", h.hexdigest()[:16], "| protocols", cp.PROTOCOLS, "| seed", cp.DEFAULT_SEED)
    print("RCDP_CALIBRATION_PROTOCOL =", os.environ.get("RCDP_CALIBRATION_PROTOCOL", "(unset -> r1)"))

    ids = {}
    for rel, expected in DATA.items():
        p = Path(rel)
        if not p.exists():
            print(f"MISSING  {rel}")
            ok = ok and rel in OPTIONAL
            continue
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        labels = {}
        for r in rows:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        flag = "ok" if len(rows) == expected else f"EXPECTED {expected}"
        print(f"{flag:12s} {rel}: {len(rows)} rows {labels}")
        ok = ok and len(rows) == expected
        ids[rel] = {str(r.get("example_id")) for r in rows}
    cal = ids.get(CAL, set())
    for crel, erel in PAIRS:
        if crel not in ids or erel not in ids:
            continue
        ov = ids[crel] & ids[erel]
        print(f"{'OVERLAP' if ov else 'disjoint':8s} {crel} x {erel}: {len(ov)} shared ids")
        ok = ok and not ov
    if CAL_V2 in ids and MIXED399 in ids:  # informational: this pair is never run
        print(f"info     {CAL_V2} x {MIXED399}: {len(ids[CAL_V2] & ids[MIXED399])} shared ids (expected 150; pair not run)")
    if TEST249 in ids and MIXED399 in ids:
        print(f"info     {TEST249} subset of {MIXED399}: {ids[TEST249] <= ids[MIXED399]}")
    print("PREFLIGHT", "OK" if ok and cal else "FAILED")
    return 0 if ok and cal else 1


if __name__ == "__main__":
    raise SystemExit(main())
