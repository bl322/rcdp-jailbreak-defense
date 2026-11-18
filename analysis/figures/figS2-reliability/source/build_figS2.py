# -*- coding: utf-8 -*-
"""Supplementary Figure S2: reliability diagrams of the raw score and the Platt output (benchmark-agnostic RCDP,
196-example pool) on the held-out partition and on the 399-item external suite, for the three scorers.

    python build_figS2.py [--root <project root>]
Bins: 10 equal-width bins on [0, 1], last bin closed (as run_r1_main.reliability); a bin is plotted if it holds at
least one item. The ECE printed in each panel is recomputed here and checked against the stored ece10 (Platt output).
Outputs build/reliability_single_turn.pdf, build/preview.png; the PDF is copied to the manuscript figures directory.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FILES = {"DistilBART-MNLI": "_r1_out/stage1json/main_dist_full_pool_calibration.json",
         "DeBERTa-v3-large": "_r1_out/extracted/results/r1/main_deb_full_pool_calibration.json",
         "Llama-Guard-3-1B, P(unsafe)": "_r1_out/lgprob/main_lgp_full_pool_calibration.json"}
DATA = (("heldout", "Held-out (204)"), ("mixed_harm149_xstest250", "Harmful + XSTest (399)"))
SERIES = (("raw_score", "Raw score", "#2a78d6", "o", "-"), ("calibrated_score", "Platt output", "#eb6834", "s", "-"))


def bins(s, y):
    b = np.minimum((s * 10).astype(int), 9)
    xs, ys, ns = [], [], []
    for k in range(10):
        m = b == k
        if m.any():
            xs.append(s[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    ece = sum(n * abs(x - q) for x, q, n in zip(xs, ys, ns)) / len(s)
    return np.array(xs), np.array(ys), ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharex=True, sharey=True)
    for c, (name, rel) in enumerate(FILES.items()):
        d = json.loads((root / rel).read_text(encoding="utf-8"))
        v = next(x for x in d["variants"] if x["variant"] == "rcdp-4state" and x["setting"] == "agnostic")
        for r, (ds, dlab) in enumerate(DATA):
            ev = v["eval"][ds]
            rec = ev["records"]
            y = np.array([x["label"] == "harmful" for x in rec], dtype=float)
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.8, ls=(0, (3, 2)), zorder=1)
            txt = []
            for key, lab, col, mk, ls in SERIES:
                s = np.array([x[key] for x in rec], dtype=float)
                xs, ys, ece = bins(s, y)
                if key == "calibrated_score":
                    assert abs(ece - ev["reliability"]["ece10"]) < 1e-9
                ax.plot(xs, ys, color=col, lw=1.4, ls=ls, marker=mk, ms=4.2, mec="white", mew=0.6, label=lab, zorder=3)
                txt.append(f"{lab}: ECE {ece:.3f}")
            head = (name + "\n") if r == 0 else ""
            ax.set_title(head + "ECE: raw " + txt[0].split()[-1] + ", Platt " + txt[1].split()[-1], fontsize=7.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.grid(True, color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{dlab}\nfraction harmful in bin", fontsize=7.5)
            if r == 1:
                ax.set_xlabel("mean predicted risk in bin")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE.parent / "build"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "reliability_single_turn.pdf")
    fig.savefig(out / "preview.png", dpi=200)
    dst = root / "ccfa-workfiles/manuscript/rcdp-cmc-r1/figures/reliability_single_turn.pdf"
    shutil.copy2(out / "reliability_single_turn.pdf", dst)
    print("written", dst)


if __name__ == "__main__":
    main()
