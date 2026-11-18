"""Fixed false-positive operating points under the R1 calibration protocol (Fig. 3, Supplementary Table S3).

For each scorer the blocking threshold is selected, as in Section 3.4, on the calibrated scores of D_thresh only (the
37 harmful and 11 benign examples that neither the risk head nor the Platt map is fitted on), but with the objective J
replaced by a constraint: among the grid thresholds {0.01, ..., 0.99} whose false-positive rate on the D_thresh benign
examples is at most the target, take those with the highest D_thresh block rate, and among them the lower median grid
point (the tie rule of Section 3.4). As in the recorded search, a grid threshold tau acts as max(tau, 0.35), the
controller's floor when trajectory features are on. The selected threshold is then applied to the 204 held-out prompts (149 harmful, 55 benign) and to the 149 held-out harmful prompts with the
250 XSTest safe prompts. Realised false-positive rates and block rates carry exact (Clopper-Pearson) 95% intervals.

Every input is a ``scripts/run_r1_main.py`` output (variant rcdp-4state, setting agnostic). The D_thresh counts are
recomputed from the saved calibration records and checked against the threshold-search table the run recorded; the
default (J-selected) point is recomputed and checked against the run's own decisions. Nothing is typed in.
The threshold frontier drawn behind the points is traced on the evaluated prompts themselves, so it is an oracle
bound, not an operating point that could be selected without their labels.

    python scripts/fixed_fpr_r1.py --inputs DistilBART=<file> DeBERTa=<file> Llama-Guard=<file> \\
        [--oneshot <llama_guard_oneshot.json>] --output fixed_fpr_operating_points.pdf

Outputs: <stem>.json and <stem>.csv (always), the figure when matplotlib is installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

GRID = [k / 100 for k in range(1, 100)]
TARGETS = (0.01, 0.05, 0.10)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "D"]
INK, INK_2, GRID_C = "#0b0b0b", "#52514e", "#e4e3df"


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    hi = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return lo, hi


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lower_median_run(taus):
    taus = sorted(taus)
    return taus[(len(taus) - 1) // 2]


def counts(recs, tau):
    bh = sum(1 for r in recs if r["label"] == "harmful" and r["calibrated_score"] >= tau)
    bb = sum(1 for r in recs if r["label"] == "benign" and r["calibrated_score"] >= tau)
    nh = sum(1 for r in recs if r["label"] == "harmful")
    nb = sum(1 for r in recs if r["label"] == "benign")
    return bh, nh, bb, nb


def frontier(recs):
    """(FPR, ABR) of 'block iff calibrated >= t' for every distinct score, plus blocking nothing."""
    ts = sorted({r["calibrated_score"] for r in recs}) + [math.inf]
    pts = []
    for t in ts:
        bh, nh, bb, nb = counts(recs, t)
        pts.append((bb / nb, bh / nh))
    return sorted(set(pts))


def point(recs, tau, name):
    bh, nh, bb, nb = counts(recs, tau)
    return {"set": name, "blocked_harmful": bh, "n_harmful": nh, "abr": bh / nh, "abr_cp95": clopper_pearson(bh, nh),
            "blocked_benign": bb, "n_benign": nb, "fpr": bb / nb, "fpr_cp95": clopper_pearson(bb, nb)}


def analyse(name, path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(d.get("meta", {}).get("runner", "")).startswith("scripts/run_r1_main.py"):
        raise SystemExit(f"{path}: not a run_r1_main.py output")
    v = next(x for x in d["variants"] if x["variant"] == "rcdp-4state" and x["setting"] == "agnostic")
    cal = v["calibration"]
    floor = 0.35 if v["config"].get("use_trajectory_features") else 0.0
    thresh = [r for r in cal["calibration_records"] if r["subset"] == "thresh"]
    table = {round(row["tau"], 2): row for row in cal["threshold_search"]["table"]}
    search = []
    for tau in GRID:  # the recorded search runs the controller, whose floor also applies on D_thresh
        bh, nh, bb, nb = counts(thresh, max(tau, floor))
        rec = table[round(tau, 2)]
        assert (bh, nh, bb, nb) == (rec["blocked_harmful"], rec["n_harmful"], rec["blocked_benign"], rec["n_benign"]), \
            f"{name}: D_thresh counts at {tau} differ from the recorded threshold search"
        search.append((tau, bh, nh, bb, nb))
    held = v["eval"]["heldout"]["records"]
    ext = v["eval"]["mixed_harm149_xstest250"]["records"]
    tau_q = cal["thresholds"]["tau_q"]
    for recs in (held, ext):  # the run's own decisions follow the rule used below
        bad = sum(1 for r in recs if (r["decision"] == "block") != (r["calibrated_score"] >= max(tau_q, floor)))
        assert bad == 0, f"{name}: {bad} records violate block == (calibrated >= max(tau_q, floor))"
    out = {"scorer": name, "source": {"file": Path(path).name, "sha256": sha256(path)}, "floor": floor,
           "d_thresh": {"n_harmful": search[0][2], "n_benign": search[0][4]},
           "default": {"tau_q": tau_q, "tau_effective": max(tau_q, floor),
                       "heldout": point(held, max(tau_q, floor), "heldout"),
                       "xstest": point(ext, max(tau_q, floor), "harm149_xstest250")},
           "targets": [], "frontier": {"heldout": frontier(held), "xstest": frontier(ext)}}
    for t in TARGETS:
        feas = [s for s in search if s[3] / s[4] <= t + 1e-12]
        best = max(s[1] for s in feas)
        tau = lower_median_run([s[0] for s in feas if s[1] == best])
        s = next(x for x in search if x[0] == tau)
        out["targets"].append({"target_fpr": t, "tau": tau, "tau_effective": max(tau, floor),
                               "d_thresh": {"blocked_harmful": s[1], "blocked_benign": s[3]},
                               "heldout": point(held, max(tau, floor), "heldout"),
                               "xstest": point(ext, max(tau, floor), "harm149_xstest250")})
    return out


def oneshot_point(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d["records"]["heldout"]
    bh = sum(1 for r in recs if r["label"] == "harmful" and r["decision"] == "block")
    nh = sum(1 for r in recs if r["label"] == "harmful")
    bb = sum(1 for r in recs if r["label"] == "benign" and r["decision"] == "block")
    nb = sum(1 for r in recs if r["label"] == "benign")
    return {"blocked_harmful": bh, "n_harmful": nh, "blocked_benign": bb, "n_benign": nb,
            "source": {"file": Path(path).name, "sha256": sha256(path)}}


def render(path, res, oneshot, fs=7.0):
    """Filled markers: the 1% and 5% targets (no false positive among the D_thresh benign examples); hollow markers:
    the 10% target. Error bars: exact 95% intervals of both rates. Light lines: oracle threshold frontiers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": fs, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "legend.fontsize": fs - 0.5})
    W, H = 6.5, 2.25
    y0, h = 0.38, 1.62
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([0.45 / W, y0 / H, 2.3 / W, h / H]), fig.add_axes([2.95 / W, y0 / H, 2.3 / W, h / H])]
    titles = ["(a) Held-out partition (55 benign prompts)", "(b) XSTest (250 safe prompts)"]
    for ax, key, title in zip(axes, ("heldout", "xstest"), titles):
        for i, r in enumerate(res):
            c, m = SLOTS[i], MARKERS[i]
            fr = r["frontier"][key]
            ax.step([100 * x for x, _ in fr], [100 * y for _, y in fr], where="post", color=c, lw=0.8, alpha=0.35,
                    zorder=1)
            drawn = set()
            for tg in r["targets"]:
                p = tg[key]
                k = (p["blocked_harmful"], p["blocked_benign"])
                if k in drawn:
                    continue
                drawn.add(k)
                filled = tg["d_thresh"]["blocked_benign"] == 0
                x, y = 100 * p["fpr"], 100 * p["abr"]
                xl, xh = (100 * v for v in p["fpr_cp95"])
                yl, yh = (100 * v for v in p["abr_cp95"])
                ax.errorbar([x], [y], xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="none", ecolor=c,
                            elinewidth=0.8, capsize=1.6, alpha=0.85, zorder=3)
                ax.plot([x], [y], marker=m, ms=4.6, color=c, mfc=c if filled else "white", mec=c if not filled else "white",
                        mew=1.0 if not filled else 0.5, zorder=4)
        if oneshot is not None and key == "heldout":
            ax.plot([100 * oneshot["blocked_benign"] / oneshot["n_benign"]],
                    [100 * oneshot["blocked_harmful"] / oneshot["n_harmful"]], marker="x", ms=4.6, color=INK, mew=1.0,
                    zorder=5)
        ax.set_xlim(-2, 66 if key == "heldout" else 70)
        ax.set_ylim(0, 102)
        ax.set_xlabel("realised false-positive rate (%)", labelpad=1.0)
        ax.set_title(title, loc="left", fontsize=fs, pad=3)
        ax.grid(color=GRID_C, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(pad=1.5)
    axes[0].set_ylabel("harmful prompts blocked (%)", labelpad=1.5)
    axes[1].tick_params(labelleft=False)
    hd = [Line2D([], [], color=SLOTS[i], marker=MARKERS[i], ls="", ms=4.6, mec="white", label=r["scorer"])
          for i, r in enumerate(res)]
    hd.append(Line2D([], [], color=INK_2, marker="o", ls="", ms=4.6, mec="white", label="1% or 5% target"))
    hd.append(Line2D([], [], color=INK_2, marker="o", ls="", ms=4.6, mfc="white", mec=INK_2, mew=1.0,
                     label="10% target"))
    if oneshot is not None:
        hd.append(Line2D([], [], color=INK, marker="x", ls="", ms=4.6, mew=1.0, label="LG one-shot verdict"))
    hd.append(Line2D([], [], color=INK_2, lw=0.8, alpha=0.4, label="frontier (oracle)"))
    fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(5.33 / W, (y0 + h + 0.05) / H), frameon=False,
               handletextpad=0.25, labelspacing=0.45, borderaxespad=0.0, handlelength=1.2)
    fig.savefig(path, pad_inches=0.0)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", required=True, help="NAME=run_r1_main output, in plotting order")
    ap.add_argument("--oneshot", help="baselines llama_guard_oneshot.json (reference point in panel a)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    res = []
    for item in a.inputs:
        name, path = item.split("=", 1)
        res.append(analyse(name, path))
    one = oneshot_point(a.oneshot) if a.oneshot else None
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    payload = {"runner": "scripts/fixed_fpr_r1.py", "argv": sys.argv[1:], "rule": __doc__, "oneshot": one,
               "scorers": [{k: v for k, v in r.items() if k != "frontier"} for r in res]}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    with open(stem.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer", "point", "tau", "tau_effective", "dthresh_blocked_harmful", "dthresh_blocked_benign",
                     "set", "blocked_harmful", "n_harmful", "abr", "abr_lo", "abr_hi", "blocked_benign", "n_benign",
                     "fpr", "fpr_lo", "fpr_hi"])
        for r in res:
            rows = [("default (J)", r["default"]["tau_q"], r["default"]["tau_effective"], "", "", r["default"])]
            rows += [(f"target {int(round(100 * t['target_fpr']))}%", t["tau"], t["tau_effective"],
                      t["d_thresh"]["blocked_harmful"], t["d_thresh"]["blocked_benign"], t) for t in r["targets"]]
            for label, tau, te, dh, db, src in rows:
                for key in ("heldout", "xstest"):
                    p = src[key]
                    w.writerow([r["scorer"], label, tau, te, dh, db, p["set"], p["blocked_harmful"], p["n_harmful"],
                                f"{p['abr']:.4f}", f"{p['abr_cp95'][0]:.4f}", f"{p['abr_cp95'][1]:.4f}",
                                p["blocked_benign"], p["n_benign"], f"{p['fpr']:.4f}", f"{p['fpr_cp95'][0]:.4f}",
                                f"{p['fpr_cp95'][1]:.4f}"])
    for r in res:
        dp = r["default"]["heldout"]
        print(f"{r['scorer']:12s} D_thresh {r['d_thresh']}  default tau_q {r['default']['tau_q']:.2f}: "
              f"{dp['blocked_harmful']}/149 blocked, FP {dp['blocked_benign']}/55")
        for t in r["targets"]:
            p, q = t["heldout"], t["xstest"]
            print(f"   target {100 * t['target_fpr']:4.0f}%: tau {t['tau']:.2f} (D_thresh {t['d_thresh']['blocked_harmful']}/37, "
                  f"FP {t['d_thresh']['blocked_benign']}/11) -> held-out {p['blocked_harmful']}/149 ({100 * p['abr']:.2f}%), "
                  f"FP {p['blocked_benign']}/55 ({100 * p['fpr']:.2f}%, CP {100 * p['fpr_cp95'][0]:.2f}-{100 * p['fpr_cp95'][1]:.2f}); "
                  f"XSTest FP {q['blocked_benign']}/250 ({100 * q['fpr']:.1f}%)")
    if one:
        print("one-shot:", one["blocked_harmful"], "/", one["n_harmful"], "FP", one["blocked_benign"], "/", one["n_benign"])
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed: figure skipped")
        return 0
    render(out, res, one)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
