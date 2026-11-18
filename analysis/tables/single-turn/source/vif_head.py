# -*- coding: utf-8 -*-
"""Variance inflation factors of the risk-head features on D_head (100 examples), from the calibration records of the
R1 result files. Constant features (all trajectory features on single-turn data) have no VIF and are listed separately.

    python vif_head.py [--root <project root>]   -> ../build/vif_head.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = {"DistilBART": "_r1_out/stage1json/main_dist_full_pool_calibration.json",
         "DeBERTa": "_r1_out/extracted/results/r1/main_deb_full_pool_calibration.json",
         "LG-prob": "_r1_out/lgprob/main_lgp_full_pool_calibration.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(HERE.parents[3]))
root = Path(ap.parse_args().root)
out = {}
for name, rel in FILES.items():
    d = json.loads((root / rel).read_text(encoding="utf-8"))
    for setting in ("agnostic", "aware"):
        v = next(x for x in d["variants"] if x["variant"] == "rcdp-4state" and x["setting"] == setting)
        recs = [r for r in v["calibration"]["calibration_records"] if r["subset"] == "head"]
        keys = sorted(recs[0]["features"])
        X = np.array([[r["features"][k] for k in keys] for r in recs], dtype=float)
        keep = [i for i in range(len(keys)) if X[:, i].std() > 0]
        vif = {}
        for i in keep:
            A = np.column_stack([np.ones(len(X))] + [X[:, j] for j in keep if j != i])
            beta, *_ = np.linalg.lstsq(A, X[:, i], rcond=None)
            r2 = 1.0 - (X[:, i] - A @ beta).var() / X[:, i].var()
            vif[keys[i]] = float(1.0 / (1.0 - r2)) if r2 < 1 - 1e-12 else float("inf")
        out[f"{name}|{setting}"] = {"n": len(recs), "vif": vif, "constant": [keys[i] for i in range(len(keys)) if i not in keep]}
        print(name, setting, len(recs), {k: round(x, 2) for k, x in vif.items()}, "constant:", out[f"{name}|{setting}"]["constant"])
(HERE.parent / "build" / "vif_head.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
