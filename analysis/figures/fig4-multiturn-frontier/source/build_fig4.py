# -*- coding: utf-8 -*-
"""Figure 4 (revision R1): session-level ABR-BAR frontiers, memory on vs off, Llama-Guard-3-1B probability scorer,
benchmark-agnostic setting, evaluation dialogues only.

Panels: (a) synthetic 612 sessions (held-out seeds), (b) ActorAttack / Attack_600 held-out half + 1,418 benign,
(c) MHJ + the same 1,418 benign. Each curve is the runner's frontier: the blocking threshold of the variant shifted
by delta (grid step 0.01), all other parameters fixed. Markers = default operating point (delta = 0).

Before plotting, the script checks the default points against the counts verified in the revision ledger, so a wrong
or stale result file stops the build.

    python build_fig4.py --root <project root>     (defaults to four levels above this file)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PANELS = [
    ("(a) Synthetic (447 / 165)", "_r1_out/lgprob/extracted/results/r1/mt/mt612_lgp.json", False),
    ("(b) ActorAttack (300 / 1,418)", "_r1_out/mtext/extracted/results/r1/mt_ext/ext_attack600_lgp.json", True),
    ("(c) MHJ (384 / 1,418)", "_r1_out/mtext/extracted/results/r1/mt_ext/ext_mhj_lgp.json", True),
]
# (variant, legend label, colour, line width, line style, marker, z-order)
SERIES = [
    ("4-state-downgrade", "Memory off (2/3/4 states)", "#2a78d6", 2.6, "-", "o", 2),
    ("4-state+memory", "4 states + memory", "#eb6834", 1.4, "-", "s", 3),
    ("2-state+memory", "2 states + memory", "#1baf7a", 1.1, (0, (3.0, 1.6)), "^", 4),
]
# default points verified in the ledger: (blocked harmful, allowed benign)
EXPECTED = {
    0: {"4-state-downgrade": (402, 144), "4-state+memory": (402, 144), "2-state+memory": (402, 144)},
    1: {"4-state-downgrade": (12, 1365), "4-state+memory": (14, 1362), "2-state+memory": (13, 1364)},
    2: {"4-state-downgrade": (216, 1365), "4-state+memory": (240, 1362), "2-state+memory": (230, 1364)},
}
INK, MUTED, GRID = "#1f1f1e", "#5f5e5b", "#e6e5e1"


def frontier(path: Path, name: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    v = d["settings"]["agnostic"]["variants"][name]
    assert d["meta"]["scorer"] == "llama-guard-prob", (path, d["meta"]["scorer"])
    pts = sorted(v["frontier"], key=lambda p: p["delta"])
    xs = [p["evaluation"]["allowed_benign"] / p["evaluation"]["n_benign"] for p in pts]
    ys = [p["evaluation"]["blocked_harmful"] / p["evaluation"]["n_harmful"] for p in pts]
    p0 = [p["evaluation"] for p in pts if p["delta"] == 0.0][0]
    return xs, ys, p0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    ap.add_argument("--manuscript-copy", default="ccfa-workfiles/manuscript/rcdp-cmc-r1/figures/multiturn_frontier.pdf")
    args = ap.parse_args()
    root = Path(args.root)

    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8, "pdf.fonttype": 42,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK, "axes.labelcolor": INK,
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.45), sharey=True)
    for k, (ax, (title, rel, guides)) in enumerate(zip(axes, PANELS)):
        path = root / rel
        for name, label, col, lw, ls, mk, z in SERIES:
            xs, ys, p0 = frontier(path, name)
            got = (p0["blocked_harmful"], p0["allowed_benign"])
            if got != EXPECTED[k][name]:
                raise SystemExit(f"{rel} {name}: default point {got} differs from the verified {EXPECTED[k][name]}")
            ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=z, label=label, solid_capstyle="round")
            ax.plot([p0["allowed_benign"] / p0["n_benign"]], [p0["blocked_harmful"] / p0["n_harmful"]], marker=mk,
                    ms=5.0 if mk != "o" else 9.0, color=col, mec="white", mew=0.8, ls="none", zorder=z + 5)
        if guides:
            for g in (0.80, 0.90, 0.95):
                ax.axvline(g, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel("Session benign allowance (BAR)")
    axes[0].set_ylabel("Session attack blocking (ABR)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
    out = HERE.parent / "build"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "multiturn_frontier.pdf")
    fig.savefig(out / "preview.png", dpi=220)
    dest = root / args.manuscript_copy
    shutil.copyfile(out / "multiturn_frontier.pdf", dest)
    print("wrote", out / "multiturn_frontier.pdf", "and", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
