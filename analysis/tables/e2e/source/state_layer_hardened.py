# -*- coding: utf-8 -*-
"""Split the state-layer effect on harmful requests that reach the model into instruction and output check, per state:
gate-only (undefended answer), with the state instruction (judged harmful before the output check), after the check."""
import argparse
from pathlib import Path

import collect_e2e as C

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(HERE.parents[3]))
root = Path(ap.parse_args().root)
for ep in C.ENDPOINTS:
    nd = C.load(root, ep, "no-defense")[1]
    for m in ("rcdp_dist", "rcdp_deb", "rcdp_lgprob"):
        recs = C.load(root, ep, m)[1]
        rows = {}
        for i, r in recs.items():
            if r["label"] != "harmful" or not r["generated"]:
                continue
            o = r["outcome"]
            t = rows.setdefault(r["state"], [0, 0, 0, 0])
            t[0] += 1
            t[1] += nd[i]["outcome"]["harmful_completion_primary"]
            t[2] += o["harmful_completion_primary_before_output_check"]
            t[3] += o["harmful_completion_primary"]
        tot = [sum(v[k] for v in rows.values()) for k in range(4)]
        print(f"{ep:8s} {m:12s} [n, gate-only, instruction, instruction+check] total {tot} | " +
              ", ".join(f"{s} {v}" for s, v in sorted(rows.items())))
