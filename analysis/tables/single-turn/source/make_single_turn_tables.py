# -*- coding: utf-8 -*-
"""LaTeX rows for Table 1, Table 2 and the supplementary single-turn tables, from the R1 result files and
build/single_turn_stats.json (run single_turn_stats.py first).

    python make_single_turn_tables.py [--root <project root>]

Every count is recounted from per-example records; the blocking threshold reported for each variant is checked
against the records (the script stops if a record contradicts it). Rounding: decimal half-up.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = "_r1_out/"
FILES = {
    ("DistilBART", 196): R + "stage1json/main_dist_full_pool_calibration.json",
    ("DeBERTa", 196): R + "extracted/results/r1/main_deb_full_pool_calibration.json",
    ("LG-prob", 196): R + "lgprob/main_lgp_full_pool_calibration.json",
    ("LG-verdict", 196): R + "stage1json/main_lg_full_pool_calibration.json",
    ("DistilBART", 346): R + "extracted/results/r1/main_dist_calibration_v2_external_benign.json",
    ("DeBERTa", 346): R + "extracted/results/r1/main_deb_calibration_v2_external_benign.json",
    ("LG-prob", 346): R + "lgprob/main_lgp_calibration_v2_external_benign.json",
}
SCORER_TEX = {"DistilBART": "DistilBART-MNLI", "DeBERTa": "DeBERTa-v3-large", "LG-prob": "Llama-Guard-3-1B, $P(\\texttt{unsafe})$",
              "LG-verdict": "Llama-Guard-3-1B, verdict label"}
VARIANTS = [("rcdp-4state", "Full protocol, 4 states"), ("3-state", "3 states"), ("2-state", "2 states"),
            ("no-platt", "Without Platt map"), ("raw-threshold", "Threshold on raw score"),
            ("no-uncertainty-gate", "Without uncertainty gate"), ("drop-uncertainty-feature", "Without uncertainty feature"),
            ("drop-length-lexical-features", "Without length/lexical features"), ("legacy-submitted", "Submitted pipeline")]
SHORT = {"DistilBART": "DistilBART", "DeBERTa": "DeBERTa", "LG-prob": "Llama-Guard"}
TOK = {"normal": 22, "cautious": 35, "hardened": 47, "quarantine": 0}


def r1(x, nd=1):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(float(x), 9))).quantize(q, ROUND_HALF_UP))


def pc(k, n):
    return r1(100.0 * k / n)


def pp(x):
    s = r1(100.0 * x)
    return s if s.startswith("-") else "+" + s


def ci(c):
    return f"[{pp(c[0])}, {pp(c[1])}]".replace("-", "$-$")


def dec(x, nd=3):
    return r1(x, nd)


def stack(value, interval):
    """Point estimate over its interval in one cell; {} keeps [ from being read as the optional argument of \\\\."""
    return "\\makecell{" + value.replace("-", "$-$") + "\\\\{}" + interval + "}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    build = HERE.parent / "build"
    st = json.loads((build / "single_turn_stats.json").read_text(encoding="utf-8"))
    runs = {k: json.loads((root / f).read_text(encoding="utf-8")) for k, f in FILES.items()}
    log = []

    def var(sc, pool, name, setting="agnostic"):
        return next(v for v in runs[(sc, pool)]["variants"] if v["variant"] == name and v["setting"] == setting)

    def counts(ev):
        rec = ev["records"]
        bh = sum(bool(r["blocked_any_turn"]) for r in rec if r["label"] == "harmful")
        ab = sum(not r["blocked_any_turn"] for r in rec if r["label"] == "benign")
        nh = sum(r["label"] == "harmful" for r in rec)
        assert (bh, ab) == (ev["blocked_harmful"], ev["allowed_benign"])
        return bh, nh, ab, len(rec) - nh

    def states(ev):
        return {s: sum(n for k, n in ev["state_counts"].items() if k.endswith("/" + s)) for s in TOK}

    def blocking_threshold(v, ds="heldout"):
        """Threshold t on the calibrated score (raw score for raw-threshold) with blocked == (score >= t) on every
        record; None when no such threshold is recorded (submitted pipeline)."""
        th, cfg = v["calibration"]["thresholds"], v["config"]
        key = "tau_h" if cfg["num_states"] == 2 else "tau_q"
        floor = (0.25 if key == "tau_h" else 0.35) if cfg.get("use_trajectory_features") else 0.0
        if v["variant"] == "legacy-submitted":
            return None
        bth = v["calibration"].get("benchmark_thresholds")
        per = {b.lower(): max(x[key], floor) for b, x in bth.items()} if bth else None
        t = max(th[key], floor)
        field = "raw_score" if v["variant"] == "raw-threshold" else "calibrated_score"
        for d in v["eval"]:
            bad = sum(bool(r["blocked_any_turn"]) != (r[field] >= (per.get(r["benchmark"].lower(), t) if per else t))
                      for r in v["eval"][d]["records"])
            assert bad == 0, (v["variant"], v["setting"], d, bad, t)
        return (per["harmbench"], per["jailbreakbench"]) if per else t

    # ---------------- Table 1(a): RCDP rows (pool 196, agnostic, held-out) ----------------
    t1 = []
    base = st["baselines"]
    for name, label in (("global-temperature DistilBART", "Global temperature + threshold, DistilBART-MNLI"),
                        ("global-temperature DeBERTa", "Global temperature + threshold, DeBERTa-v3-large"),
                        ("Llama-Guard P(unsafe) >= tau", "Llama-Guard-3-1B, threshold on $P(\\texttt{unsafe})$"),
                        ("Llama-Guard one-shot verdict", "Llama-Guard-3-1B, one-shot verdict")):
        b = base[name]
        t1.append(f"{label} & {b['blocked']} ({pc(b['blocked'], b['n_h'])}) & {b['allowed']} ({pc(b['allowed'], b['n_b'])}) & {dec(b['auc'])} \\\\")
    t1.append("%% RCDP rows")
    for sc in ("DistilBART", "DeBERTa", "LG-prob", "LG-verdict"):
        v = var(sc, 196, "rcdp-4state")
        bh, nh, ab, nb = counts(v["eval"]["heldout"])
        t1.append(f"RCDP, {SCORER_TEX[sc]} & {bh} ({pc(bh, nh)}) & {ab} ({pc(ab, nb)}) & {dec(v['eval']['heldout']['auc_calibrated'])} \\\\")
    # ---------------- Table 1(b): state resolution ----------------
    t1b = []
    for sc in ("DistilBART", "DeBERTa", "LG-prob", "LG-verdict"):
        s = st["states"][sc]
        assert s["identical_decisions_2_3_4"]
        q = s["rcdp-4state"]["states"]
        tk = [s[k]["hidden_prompt_tokens_per_request"] for k in ("2-state", "3-state", "rcdp-4state")]
        t1b.append(f"{SCORER_TEX[sc]} & 204/204 & {q['quarantine']} / {q['hardened']} / {q['cautious']} / {q['normal']} & "
                   f"{tk[0]:.1f} / {tk[1]:.1f} / {tk[2]:.1f} \\\\")
    # check 3-state == 4-state item by item in all 14 configurations, and 2-state where it differs
    ident = []
    for (sc, pool), d in runs.items():
        for setting in ("agnostic", "aware"):
            dec4 = {r["example_id"]: r["blocked_any_turn"] for ds in ("heldout",) for r in var(sc, pool, "rcdp-4state", setting)["eval"][ds]["records"]}
            dec3 = {r["example_id"]: r["blocked_any_turn"] for r in var(sc, pool, "3-state", setting)["eval"]["heldout"]["records"]}
            dec2 = {r["example_id"]: r["blocked_any_turn"] for r in var(sc, pool, "2-state", setting)["eval"]["heldout"]["records"]}
            ident.append((sc, pool, setting, dec3 == dec4, sum(dec2[k] != dec4[k] for k in dec4)))
    log.append("3-state vs 4-state identical (held-out) / 2-state differing items: " + "; ".join(f"{a} {b} {c}: {d}/{e}" for a, b, c, d, e in ident))

    # ---------------- Table 2(a): scorer contrast on the 399-item suite ----------------
    t2a = []
    sizes = {"DistilBART": "256\\,M", "DeBERTa": "435\\,M", "LG-prob": "1.5\\,B"}
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        v = var(sc, 196, "rcdp-4state")
        bh, nh, ab, nb = counts(v["eval"]["heldout"])
        bh2, nh2, ae, ne = counts(v["eval"]["mixed_harm149_xstest250"])
        assert (bh2, nh2) == (bh, nh)
        cal = next(c for c in st["calibration"] if c["scorer"] == sc and c["pool"] == 196 and c["data"] == "mixed_harm149_xstest250" and c["score"] == "platt")
        t2a.append(f"{SCORER_TEX[sc].split(',')[0]} ({sizes[sc]}) & {bh} ({pc(bh, nh)}) & {ab} ({pc(ab, nb)}) & {ae}/{ne} ({pc(ae, ne)}) & "
                   f"{dec(v['eval']['mixed_harm149_xstest250']['auc_calibrated'])} \\\\")
    # ---------------- Table 2(b): calibration-pool contrast ----------------
    t2b = []
    pcs = {(c["scorer"], c["pipeline"], c["data"]): c for c in st["pool_contrasts"]}
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        for pipe, lab in (("legacy-submitted", "submitted"), ("rcdp-4state", "revised")):
            ho = pcs[(sc, pipe, "heldout")]
            ex = pcs[(sc, pipe, "mixed_harm149_xstest_test100")]
            for side, pool in (("A", 196), ("B", 346)):
                v = var(sc, pool, pipe)
                bh, nh, ab, nb = counts(v["eval"]["heldout"])
                assert abs(ho["ABR"][side] - bh / nh) < 1e-12 and abs(ho["BAR"][side] - ab / nb) < 1e-12
                ke = round(ex["BAR"][side] * 100)
                t2b.append(f"{SHORT[sc]}, {lab}, {pool} pool & {bh} ({pc(bh, nh)}) & {ab} ({pc(ab, nb)}) & "
                           f"{ke}/100 ({pc(ke, 100)}) & {dec(ex['AUC'][side])} \\\\")
            t2b.append("%% delta ABR " + f"{pp(ho['ABR']['delta'])} {ci(ho['ABR']['ci'])} disc {ho['ABR']['discordant_A_only_B_only']} p {ho['ABR']['mcnemar_p']:.3g}; "
                       f"delta BAR_ID {pp(ho['BAR']['delta'])} {ci(ho['BAR']['ci'])}; delta BAR_Ext {pp(ex['BAR']['delta'])} {ci(ex['BAR']['ci'])} "
                       f"disc {ex['BAR']['discordant_A_only_B_only']} p {ex['BAR']['mcnemar_p']:.3g}")
        t2b.append("\\midrule")
    t2b = t2b[:-1]

    # ---------------- Supplementary: all variants, pool 196 ----------------
    def variant_rows(setting, names):
        rows = []
        for sc in ("DistilBART", "DeBERTa", "LG-prob"):
            rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{SCORER_TEX[sc]}}}}} \\\\")
            for key, lab in VARIANTS:
                if key not in names:
                    continue
                v = var(sc, 196, key, setting)
                ev = v["eval"]["heldout"]
                bh, nh, ab, nb = counts(ev)
                t = blocking_threshold(v)
                s = states(ev)
                auc = ev["auc_calibrated"] if key != "raw-threshold" else ev["auc_raw"]
                rows.append(f"~~{lab} & {'--' if t is None else (dec(t[0], 2) + '/' + dec(t[1], 2) if isinstance(t, tuple) else dec(t, 2))} & {bh} & {ab} & {dec(auc)} & {dec(ev['reliability']['ece10'])} & "
                            f"{dec(ev['reliability']['brier'])} & {s['quarantine']}/{s['hardened']}/{s['cautious']}/{s['normal']} \\\\")
        return rows
    sv_agn = variant_rows("agnostic", [k for k, _ in VARIANTS])
    sv_awa = variant_rows("aware", ["rcdp-4state", "3-state", "2-state", "no-platt", "legacy-submitted"])

    # ---------------- Supplementary: calibration ----------------
    scal = []
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        scal.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{SCORER_TEX[sc]}}}}} \\\\")
        for pool, ds, lab in ((196, "heldout", "196 pool, held-out (204)"), (196, "mixed_harm149_xstest250", "196 pool, 399 items"),
                              (196, "subset249", "196 pool, disjoint 249 items"), (346, "heldout", "346 pool, held-out (204)"),
                              (346, "mixed_harm149_xstest_test100", "346 pool, disjoint 249 items")):
            cells = []
            for score in ("raw", "head", "platt"):
                c = next(x for x in st["calibration"] if x["scorer"] == sc and x["pool"] == pool and x["data"] == ds and x["score"] == score)
                cells.append(stack(dec(c['ece']), f"[{dec(c['ece_ci'][0])}, {dec(c['ece_ci'][1])}]"))
            briers = []
            for score in ("raw", "head", "platt"):
                c = next(x for x in st["calibration"] if x["scorer"] == sc and x["pool"] == pool and x["data"] == ds and x["score"] == score)
                briers.append(dec(c["brier"]))
            scal.append(f"~~{lab} & {cells[0]} & {cells[1]} & {cells[2]} & {' / '.join(briers)} \\\\")
    gts = [c for c in st["calibration"] if c["score"] == "global-temperature"]
    scal_note = "; ".join(f"{c['scorer']} {dec(c['ece'])} [{dec(c['ece_ci'][0])}, {dec(c['ece_ci'][1])}], Brier {dec(c['brier'])}" for c in gts)

    # ---------------- Supplementary: matched operating points ----------------
    smatch = []
    names = {"DistilBART": "DistilBART", "DeBERTa": "DeBERTa", "LG-prob": "Llama-Guard"}
    for c in st["scorer_contrasts"]:
        A, B_ = c["arms"]
        data = "held-out" if c["data"] == "heldout" else "399 items"
        m1, m2 = c["matched_at_BAR_of_A"], c["matched_at_ABR_of_A"]
        sd = lambda x: (("+" if x >= 0 else "$-$") + dec(abs(x), 3))
        smatch.append(f"{names[B_]} $-$ {names[A]}, {data} & {stack(sd(c['AUC']['delta']), '[' + sd(c['AUC']['ci'][0]) + ', ' + sd(c['AUC']['ci'][1]) + ']')} & "
                      f"{r1(100 * m1['target'])} & {r1(100 * m1['A'])} / {r1(100 * m1['B'])} & {stack(pp(m1['delta']), ci(m1['ci']))} & "
                      f"{r1(100 * m2['target'])} & {r1(100 * m2['A'])} / {r1(100 * m2['B'])} & {stack(pp(m2['delta']), ci(m2['ci']))} \\\\")
    smatch_targets = []
    for c in st["scorer_contrasts"]:
        A, B_ = c["arms"]
        for m in c["matched_at_BAR_targets"]:
            smatch_targets.append(f"{c['data']} {names[B_]}-{names[A]} BAR>={m['target']:.2f}: {r1(100 * m['A'])}/{r1(100 * m['B'])} "
                                  f"(BAR {r1(100 * m['A_realised'])}/{r1(100 * m['B_realised'])}) {pp(m['delta'])} {ci(m['ci'])}")

    out = {"tab1a_rows.tex": t1, "tab1b_rows.tex": t1b, "tab2a_rows.tex": t2a, "tab2b_rows.tex": t2b,
           "tabS_variants_agnostic_rows.tex": sv_agn, "tabS_variants_aware_rows.tex": sv_awa,
           "tabS_calibration_rows.tex": scal, "tabS_matched_rows.tex": smatch}
    for f, rows in out.items():
        (build / f).write_text("\n".join(rows) + "\n", encoding="utf-8")
    log += ["global-temperature ECE: " + scal_note, "matched at BAR targets:"] + smatch_targets
    (build / "tables_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    for f, rows in out.items():
        print("=====", f); print("\n".join(rows))
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
