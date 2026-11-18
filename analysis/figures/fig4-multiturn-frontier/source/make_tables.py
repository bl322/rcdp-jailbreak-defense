# -*- coding: utf-8 -*-
"""Rows of Table 5 (main text) and Supplementary Tables S5a/S5b (multi-turn), read from the result files.

    python make_tables.py --root <project root>
Writes build/tab_multiturn_main.tex and build/tab_s5{a,b,c}_rows.tex (S5b LG-prob, S5c DistilBART) and prints them.
Percentages with one decimal; differences in percentage points (pp) with one decimal.
"""
from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYN = {"lgp": "_r1_out/lgprob/extracted/results/r1/mt/mt612_lgp.json",
       "dist": "_r1_out/stage2_mt/results/r1/mt/mt612_dist.json"}
EXT = "_r1_out/mtext/extracted/results/r1/mt_ext"
SETS = [("attack600", "ActorAttack"), ("mhj", "MHJ")]
SCORER = {"lgp": "Llama-Guard-3-1B (prob.)", "dist": "DistilBART-MNLI"}
PAIRS = [("4-state+memory", "4-state-downgrade", "4 states"), ("2-state+memory", "2-state-downgrade", "2 states")]
SYN_TARGETS = (0.70, 0.80, 0.90)
NOMEM = ["2-state", "3-state", "4-state", "2-state-downgrade", "3-state-downgrade", "4-state-downgrade"]


def r1(x):
    """Round half away from zero to one decimal (56.25 -> 56.3), on the decimal value of the fraction."""
    q = Decimal(repr(round(x, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return q


def pct(k, n):
    return f"{r1(100.0 * k / n)}"


def pp(x):
    v = r1(100.0 * x)
    return "0.0" if v == 0 else (f"+{v}" if v > 0 else f"$-${-v}")


def ci(lo, hi):
    return f"[{pp(lo)}, {pp(hi)}]"


def load(root, rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def syn_target(mb, t):
    for key, v in mb["targets"].items():
        if abs(v["target_bar_selection"] - t) < 1e-6:
            return v
    raise KeyError(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out = HERE.parent / "build"
    out.mkdir(parents=True, exist_ok=True)

    syn = {t: load(root, p) for t, p in SYN.items()}
    ext = {(s, t): load(root, f"{EXT}/ext_{s}_{t}.json") for s, _ in SETS for t in ("lgp", "dist")}
    mat = {(s, t): load(root, f"{EXT}/matched_{s}_{t}.json") for s, _ in SETS for t in ("lgp", "dist")}

    # ---------------- main table: LG-prob, 4 states + memory vs memory off ----------------
    main_rows = []
    mb = syn["lgp"]["settings"]["agnostic"]["matched_bar"]
    for j, t in enumerate(SYN_TARGETS):
        tg = syn_target(mb, t)
        be = tg["bootstrap_evaluation"]
        on, off = be["point"]["4-state+memory"], be["point"]["4-state-downgrade"]
        d = be["diff"]["4-state+memory - 4-state-downgrade"]["ABR"]
        lab = "Synthetic$^{a}$" if j == 0 else ""
        main_rows.append(f"{lab} & {t:.2f} & {pct(*off['ABR_k_n'])} & {pct(*off['BAR_k_n'])} & {pct(*on['ABR_k_n'])} & "
                         f"{pct(*on['BAR_k_n'])} & {pp(d['point'])} {ci(*d['clustered_paired'])} \\\\")
    for s, name in SETS:
        m = mat[(s, "lgp")]
        main_rows.append("\\midrule")
        for j, c in enumerate([c for c in m["contrasts"] if c["on"] == "4-state+memory"]):
            von, voff = m["variants"]["4-state+memory"][str(c["target_bar"])], m["variants"]["4-state-downgrade"][str(c["target_bar"])]
            lab = f"{name}$^{{b}}$" if j == 0 else ""
            main_rows.append(f"{lab} & {c['target_bar']:.2f} & {pct(voff['blocked_harmful'], voff['n_harmful'])} & "
                             f"{r1(100 * voff['bar'])} & {pct(von['blocked_harmful'], von['n_harmful'])} & {r1(100 * von['bar'])} & "
                             f"{pp(c['delta_abr'])} {ci(c['rematched_ci']['lo'], c['rematched_ci']['hi'])} \\\\")

    # ---------------- S5a: default operating points ----------------
    a_rows = []
    for t in ("lgp", "dist"):
        a_rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{SCORER[t]} scorer}}}} \\\\")
        for label, V, benches in [("Synthetic", syn[t]["settings"]["agnostic"]["variants"], None)] + \
                [(name, ext[(s, t)]["settings"]["agnostic"]["variants"], ("MT-Bench-101", "MS-XSB")) for s, name in SETS]:
            rows = {}
            for n in NOMEM + ["2-state+memory", "3-state+memory", "4-state+memory", "trajectory-policy"]:
                if n not in V:
                    continue
                ev = [r for r in V[n]["sessions"] if r["side"] == "evaluation"]
                h = [r for r in ev if r["label"] == "harmful"]
                b = [r for r in ev if r["label"] == "benign"]
                bh, ab = sum(r["blocked"] for r in h), sum(not r["blocked"] for r in b)
                per = ""
                if benches:
                    per = " / ".join(f"{sum(not r['blocked'] for r in b if r['benchmark'] == x)}/"
                                     f"{sum(1 for r in b if r['benchmark'] == x)}" for x in benches)
                ids = {r["example_id"] for r in ev}
                toks = [tr["prompt_tokens"] for tr in V[n]["turns"] if tr["session_id"] in ids]
                rows[n] = (bh, len(h), ab, len(b), per, sum(toks) / len(toks))
            nm = {rows[n][:4] for n in NOMEM if n in rows}
            if len(nm) != 1:
                raise SystemExit(f"{label}/{t}: memory-off variants differ: {nm}")
            show = [("Memory off, 2 states", rows["2-state-downgrade"]), ("Memory off, 3 states", rows["3-state-downgrade"]),
                    ("Memory off, 4 states", rows["4-state-downgrade"]), ("Memory on, 2 states", rows["2-state+memory"]),
                    ("Memory on, 3 states", rows["3-state+memory"]), ("Memory on, 4 states", rows["4-state+memory"])]
            if "trajectory-policy" in rows:
                show.append(("Trajectory policy", rows["trajectory-policy"]))
            for j, (vn, (bh, nh, ab, nb, per, tk)) in enumerate(show):
                lab = label if j == 0 else ""
                a_rows.append(f"{lab} & {vn} & {bh}/{nh} ({pct(bh, nh)}) & {ab}/{nb} ({pct(ab, nb)}) & {per or '--'} & "
                              f"{tk:.2f} \\\\")
            a_rows.append("\\midrule")
    a_rows = a_rows[:-1]

    # ---------------- S5b (LG-prob) and S5c (DistilBART): matched contrasts, both pairs ----------------
    bc_rows = {}
    for t in ("lgp", "dist"):
        b_rows = bc_rows[t] = []
        mb = syn[t]["settings"]["agnostic"]["matched_bar"]
        for pon, poff, pn in PAIRS:
            for j, tt in enumerate(SYN_TARGETS):
                tg = syn_target(mb, tt)
                be = tg["bootstrap_evaluation"]
                on, off = be["point"][pon], be["point"][poff]
                d = be["diff"][f"{pon} - {poff}"]
                lab = f"Synthetic, {pn}" if j == 0 else ""
                b_rows.append(f"{lab} & {tt:.2f} & {pct(*off['ABR_k_n'])} / {pct(*off['BAR_k_n'])} & "
                              f"{pct(*on['ABR_k_n'])} / {pct(*on['BAR_k_n'])} & {pp(d['ABR']['point'])} {ci(*d['ABR']['clustered_paired'])} & "
                              f"-- & {pp(d['BAR']['point'])} \\\\")
            b_rows.append("\\midrule")
        for s, name in SETS:
            m = mat[(s, t)]
            for pon, poff, pn in PAIRS:
                cs = [c for c in m["contrasts"] if c["on"] == pon]
                for j, c in enumerate(cs):
                    von, voff = m["variants"][pon][str(c["target_bar"])], m["variants"][poff][str(c["target_bar"])]
                    flag = lambda v: "" if v["target_reached"] else "$^{\\dagger}$"
                    lab = f"{name}, {pn}" if j == 0 else ""
                    b_rows.append(f"{lab} & {c['target_bar']:.2f} & {pct(voff['blocked_harmful'], voff['n_harmful'])} / "
                                  f"{r1(100 * voff['bar'])}{flag(voff)} & {pct(von['blocked_harmful'], von['n_harmful'])} / "
                                  f"{r1(100 * von['bar'])}{flag(von)} & {pp(c['delta_abr'])} "
                                  f"{ci(c['rematched_ci']['lo'], c['rematched_ci']['hi'])} & "
                                  f"{ci(c['conditional_ci']['lo'], c['conditional_ci']['hi'])} & {pp(von['bar'] - voff['bar'])} \\\\")
                b_rows.append("\\midrule")
        del b_rows[-1]

    # 4 states + memory vs 2 states + memory (synthetic CIs; independent sets: points)
    extra = []
    for t in ("lgp", "dist"):
        mb = syn[t]["settings"]["agnostic"]["matched_bar"]
        for tt in SYN_TARGETS:
            d = syn_target(mb, tt)["bootstrap_evaluation"]["diff"]["4-state+memory - 2-state+memory"]
            extra.append(f"synthetic {t} {tt:.2f}: dABR {pp(d['ABR']['point'])} {ci(*d['ABR']['clustered_paired'])}, "
                         f"dBAR {pp(d['BAR']['point'])} {ci(*d['BAR']['clustered_paired'])}")
        for s, name in SETS:
            m = mat[(s, t)]
            for c in [c for c in m["contrasts"] if c["on"] == "4-state+memory"]:
                a4 = m["variants"]["4-state+memory"][str(c["target_bar"])]
                a2 = m["variants"]["2-state+memory"][str(c["target_bar"])]
                extra.append(f"{name} {t} {c['target_bar']:.2f}: dABR {pp(a4['abr'] - a2['abr'])}, dBAR {pp(a4['bar'] - a2['bar'])}"
                             f"{'' if a4['target_reached'] and a2['target_reached'] else ' (target not reached)'}")

    for fn, rows in (("tab_multiturn_main.tex", main_rows), ("tab_s5a_rows.tex", a_rows),
                     ("tab_s5b_rows.tex", bc_rows["lgp"]), ("tab_s5c_rows.tex", bc_rows["dist"]),
                     ("four_vs_two_with_memory.txt", extra)):
        (out / fn).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"===== {fn}")
        print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
