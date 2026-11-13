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
}


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
            ok = ok and "calibration_v2" in rel  # the 346 pool is only needed for the Table 2 contrast
            continue
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        labels = {}
        for r in rows:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        flag = "ok" if len(rows) == expected else f"EXPECTED {expected}"
        print(f"{flag:12s} {rel}: {len(rows)} rows {labels}")
        ok = ok and len(rows) == expected
        ids[rel] = {str(r.get("example_id")) for r in rows}
    cal = ids.get("data/paper_splits/full_pool_calibration.jsonl", set())
    for rel, s in ids.items():
        if "calibration" in rel:
            continue
        for crel in ("data/paper_splits/full_pool_calibration.jsonl", "data/paper_splits/calibration_v2_external_benign.jsonl"):
            ov = ids.get(crel, set()) & s
            if ov:
                print(f"OVERLAP  {rel} shares {len(ov)} ids with {crel}")
                ok = False
    print("PREFLIGHT", "OK" if ok and cal else "FAILED")
    return 0 if ok and cal else 1


if __name__ == "__main__":
    raise SystemExit(main())
