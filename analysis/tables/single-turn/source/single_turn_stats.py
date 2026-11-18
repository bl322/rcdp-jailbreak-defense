# -*- coding: utf-8 -*-
"""Single-turn statistics for Section 4.2 and the supplementary single-turn tables, recomputed from the per-example
records of the R1 result files (no number is copied from an aggregate without being recounted).

    python single_turn_stats.py [--root <project root>] [--B 10000] [--seed 20261114]

Outputs in ../build/: single_turn_stats.json (all numbers) and single_turn_stats.md (report).

Definitions
  Decision rule: under the 4- and 3-state controllers a single-turn request is blocked iff calibrated_score >= max(tau_q, 0.35)
    (the 0.35 floor applies when trajectory features are on);
    this is checked for every record of every benchmark-agnostic run used here and the script stops on a violation.
    The 2-state decisions are compared with the 4-state decisions item by item.
  ABR = blocked harmful / harmful; BAR = allowed benign / benign.
  Bootstrap: harmful and benign items are resampled separately with replacement; the same draws are used for every
    arm evaluated on the same items (paired). B replicates from numpy default_rng(seed); percentile 95% intervals.
  Matched operating point, on the evaluation items: at a target BAR b*, each arm uses the lowest threshold whose BAR is
    at least b* (the highest ABR under that constraint); at a target ABR a*, the highest threshold whose ABR is at
    least a*. The target stays fixed and the thresholds are re-matched in every replicate.
  ECE: 10 equal-width bins on [0, 1], last bin closed (as run_r1_main.reliability). Brier: mean squared error.
  AUC: Mann-Whitney statistic with ties counted 1/2. McNemar: exact two-sided binomial test on discordant pairs.
"""
from __future__ import annotations

import argparse
import json
import math
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
BASE = R + "stage2_cost/results/r1/cost/baselines/"
EXPECTED = {  # (scorer, pool, variant, dataset) -> (blocked/149, allowed/n_benign), verified earlier in the ledger
    ("DistilBART", 196, "rcdp-4state", "heldout"): (119, 28), ("DeBERTa", 196, "rcdp-4state", "heldout"): (129, 40),
    ("LG-prob", 196, "rcdp-4state", "heldout"): (134, 48), ("LG-verdict", 196, "rcdp-4state", "heldout"): (143, 42),
    ("DistilBART", 196, "legacy-submitted", "heldout"): (117, 30), ("DistilBART", 346, "legacy-submitted", "heldout"): (31, 54),
    ("DistilBART", 346, "rcdp-4state", "heldout"): (88, 43), ("DeBERTa", 346, "rcdp-4state", "heldout"): (129, 37),
    ("DeBERTa", 346, "legacy-submitted", "heldout"): (90, 51), ("DeBERTa", 196, "legacy-submitted", "heldout"): (132, 33),
    ("LG-prob", 346, "rcdp-4state", "heldout"): (134, 48), ("LG-prob", 346, "legacy-submitted", "heldout"): (133, 48),
    ("LG-prob", 196, "legacy-submitted", "heldout"): (137, 45),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pct(a, q):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if q else None


class Items:
    """Records of one arm on one item set, ordered by example_id, harmful first."""

    def __init__(self, recs, score="cal", tau=None):
        recs = sorted(recs, key=lambda r: (r["label"] != "harmful", r["example_id"]))
        self.ids = [r["example_id"] for r in recs]
        self.y = np.array([r["label"] == "harmful" for r in recs], dtype=float)
        self.raw = np.array([r["raw_score"] for r in recs], dtype=float)
        self.head = sigmoid([r["head_logit"] for r in recs]) if "head_logit" in recs[0] else None
        self.cal = np.array([r["calibrated_score"] for r in recs], dtype=float) if "calibrated_score" in recs[0] else None
        self.blocked = np.array([bool(r["blocked_any_turn"]) if "blocked_any_turn" in r else r["decision"] == "block"
                                 for r in recs], dtype=bool)
        self.nh = int(self.y.sum())
        self.nb = len(recs) - self.nh
        if tau is not None:
            bad = int(np.sum(self.blocked != (self.cal >= tau)))
            assert bad == 0, f"{bad} records violate blocked == (calibrated >= tau_q)"

    def subset(self, keep):
        idx = [i for i, e in enumerate(self.ids) if e in keep]
        out = object.__new__(Items)
        out.ids = [self.ids[i] for i in idx]
        for a in ("y", "raw", "head", "cal", "blocked"):
            v = getattr(self, a)
            setattr(out, a, None if v is None else v[idx])
        out.nh = int(out.y.sum())
        out.nb = len(idx) - out.nh
        return out


class Boot:
    def __init__(self, nh, nb, B, seed):
        rng = np.random.default_rng(seed)
        self.nh, self.nb, self.B = nh, nb, B
        self.ch = rng.multinomial(nh, np.full(nh, 1.0 / nh), size=B).astype(float)
        self.cb = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=B).astype(float)
        self.c = np.hstack([self.ch, self.cb])

    # point estimate uses unit counts; replicates use the multinomial counts
    def ece_brier(self, s, y):
        bins = np.minimum((s * 10).astype(int), 9)
        oh = np.eye(10)[bins]
        n = len(s)
        conf = oh * s[:, None]
        lab = oh * y[:, None]
        e0 = float(np.abs(conf.sum(0) - lab.sum(0)).sum() / n)
        b0 = float(((s - y) ** 2).mean())
        er = np.abs(self.c @ conf - self.c @ lab).sum(1) / n
        br = self.c @ ((s - y) ** 2) / n
        return e0, pct(er, 1), b0, pct(br, 1)

    def auc(self, s, y):
        sp, sn = s[y == 1], s[y == 0]
        W = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
        a0 = float(W.mean())
        ar = ((self.ch @ W) * self.cb).sum(1) / (self.nh * self.nb)
        return a0, ar

    def rates(self, blocked, y):
        bh, ab = blocked[y == 1].astype(float), (~blocked[y == 0]).astype(float)
        return (bh.sum() / self.nh, ab.sum() / self.nb, self.ch @ bh / self.nh, self.cb @ ab / self.nb)

    def frontier(self, s, y):
        """Blocked-harmful and allowed-benign counts for every threshold (block iff s >= t), per replicate."""
        sp, sn = s[y == 1], s[y == 0]
        U = np.unique(s)
        K = len(U) + 1  # last threshold = +inf blocks nothing
        ih, ib = np.searchsorted(U, sp), np.searchsorted(U, sn)
        Hh, Hb = np.eye(K)[ih], np.eye(K)[ib]
        def counts(ch, cb):
            hc, bc = ch @ Hh, cb @ Hb
            blk = np.flip(np.cumsum(np.flip(hc, -1), -1), -1)            # sum_{j >= k}
            alw = np.concatenate([np.zeros(hc.shape[:-1] + (1,)), np.cumsum(bc, -1)[..., :-1]], -1)  # sum_{j < k}
            return blk, alw
        return counts(np.ones(self.nh), np.ones(self.nb)), counts(self.ch, self.cb)

    @staticmethod
    def at_bar(blk, alw, nh, nb, target):
        need = math.ceil(target * nb - 1e-9)
        k = np.argmax(alw >= need - 1e-9, axis=-1)
        return np.take_along_axis(blk, k[..., None], -1)[..., 0] / nh, np.take_along_axis(alw, k[..., None], -1)[..., 0] / nb

    @staticmethod
    def at_abr(blk, alw, nh, nb, target):
        need = math.ceil(target * nh - 1e-9)
        ok = blk >= need - 1e-9
        k = ok.shape[-1] - 1 - np.argmax(np.flip(ok, -1), axis=-1)
        return np.take_along_axis(alw, k[..., None], -1)[..., 0] / nb, np.take_along_axis(blk, k[..., None], -1)[..., 0] / nh


def contrast(boot, A, B_, nameA, nameB, targets_bar=()):
    """B_ minus A on the same items: fitted points, AUC, matched points."""
    assert A.ids == B_.ids
    y = A.y
    out = {"arms": [nameA, nameB], "n_harmful": boot.nh, "n_benign": boot.nb}
    ra, rb = boot.rates(A.blocked, y), boot.rates(B_.blocked, y)
    for j, m in ((0, "ABR"), (1, "BAR")):
        d = rb[2 + j] - ra[2 + j]
        if m == "ABR":
            sel = y == 1
            b = int(np.sum(A.blocked[sel] & ~B_.blocked[sel])); c = int(np.sum(~A.blocked[sel] & B_.blocked[sel]))
        else:
            sel = y == 0
            b = int(np.sum(~A.blocked[sel] & B_.blocked[sel])); c = int(np.sum(A.blocked[sel] & ~B_.blocked[sel]))
        out[m] = {"A": float(ra[j]), "B": float(rb[j]), "delta": float(rb[j] - ra[j]), "ci": pct(d, 1),
                  "discordant_A_only_B_only": [b, c], "mcnemar_p": mcnemar(b, c)}
    a0, ar = boot.auc(A.cal, y)
    b0, brr = boot.auc(B_.cal, y)
    out["AUC"] = {"A": a0, "B": b0, "delta": b0 - a0, "ci": pct(brr - ar, 1)}
    fa, fb = boot.frontier(A.cal, y), boot.frontier(B_.cal, y)
    def matched(kind, target):
        f = Boot.at_bar if kind == "BAR" else Boot.at_abr
        pa, pb = f(*fa[0], boot.nh, boot.nb, target), f(*fb[0], boot.nh, boot.nb, target)
        qa, qb = f(*fa[1], boot.nh, boot.nb, target), f(*fb[1], boot.nh, boot.nb, target)
        return {"target": target, "A": float(pa[0]), "B": float(pb[0]), "A_realised": float(pa[1]),
                "B_realised": float(pb[1]), "delta": float(pb[0] - pa[0]), "ci": pct(qb[0] - qa[0], 1)}
    out["matched_at_BAR_of_A"] = matched("BAR", float(ra[1]))
    out["matched_at_ABR_of_A"] = matched("ABR", float(ra[0]))
    out["matched_at_BAR_targets"] = [matched("BAR", t) for t in targets_bar]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = HERE.parent / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {k: json.loads((root / f).read_text(encoding="utf-8")) for k, f in FILES.items()}
    def get(scorer, pool, variant, ds, setting="agnostic"):
        v = next(x for x in runs[(scorer, pool)]["variants"] if x["variant"] == variant and x["setting"] == setting)
        # the 4- and 3-state controllers block iff calibrated >= tau_q; the 2-state, raw-threshold and legacy variants
        # record a different blocking boundary, so the rule is checked only where it is the stated rule
        # with trajectory features on, the controller floors tau_q at 0.35 (controller.select_controller_state; on a
        # single turn progress, memory and trend are 0), so the effective blocking threshold is max(tau_q, 0.35)
        tau = None
        if setting == "agnostic" and variant in ("rcdp-4state", "3-state"):
            tau = v["calibration"]["thresholds"]["tau_q"]
            if v["config"].get("use_trajectory_features"):
                tau = max(tau, 0.35)
        it = Items(v["eval"][ds]["records"], tau=tau)
        exp = EXPECTED.get((scorer, pool, variant, ds))
        if exp is not None:
            got = (int(it.blocked[it.y == 1].sum()), int((~it.blocked[it.y == 0]).sum()))
            assert got == exp, f"{scorer} {pool} {variant} {ds}: {got} != {exp}"
        return it, v

    res = {"definitions": __doc__, "B": a.B, "seed": a.seed}
    md = ["# Single-turn statistics (R1 records; B = %d, seed %d)" % (a.B, a.seed), ""]

    # 1. decision rule and state resolution
    md += ["## Decision rule and state resolution (pool 196, agnostic, held-out)", ""]
    tok = {"normal": 22, "cautious": 35, "hardened": 47, "quarantine": 0}  # measured instruction tokens (cost stage)
    res["states"] = {}
    for sc in ("DistilBART", "DeBERTa", "LG-prob", "LG-verdict"):
        row = {}
        for var in ("rcdp-4state", "3-state", "2-state"):
            it, v = get(sc, 196, var, "heldout")
            st = v["eval"]["heldout"]["state_counts"]
            cnt = {s: sum(n for k, n in st.items() if k.endswith("/" + s)) for s in tok}
            row[var] = {"blocked": int(it.blocked[it.y == 1].sum()), "allowed": int((~it.blocked[it.y == 0]).sum()),
                        "states": cnt, "hidden_prompt_tokens_per_request": sum(tok[s] * n for s, n in cnt.items()) / 204,
                        "decisions": it.blocked.tolist()}
        same = row["rcdp-4state"]["decisions"] == row["3-state"]["decisions"] == row["2-state"]["decisions"]
        for var in row:
            row[var].pop("decisions")
        row["identical_decisions_2_3_4"] = same
        res["states"][sc] = row
        md.append(f"- {sc}: identical decisions {same}; " + "; ".join(
            f"{var} {r['blocked']}/149 {r['allowed']}/55 Q/H/C/N {r['states']['quarantine']}/{r['states']['hardened']}/"
            f"{r['states']['cautious']}/{r['states']['normal']} tokens {r['hidden_prompt_tokens_per_request']:.2f}"
            for var, r in row.items() if var != "identical_decisions_2_3_4"))
    md.append("")

    # 2. calibration metrics with bootstrap intervals
    md += ["## Calibration (ECE10, Brier; 95% bootstrap CI)", "", "| scorer | pool | data | score | ECE | Brier | AUC |", "|---|---|---|---|---|---|---|"]
    res["calibration"] = []
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        keep249 = set(get(sc, 346, "rcdp-4state", "mixed_harm149_xstest_test100")[0].ids)
        for pool, dss in ((196, ("heldout", "mixed_harm149_xstest250", "subset249")), (346, ("heldout", "mixed_harm149_xstest_test100"))):
            for ds in dss:
                if ds == "subset249":  # the 196-pool arm on the disjoint 249 items used by the 346-pool arm
                    it = get(sc, pool, "rcdp-4state", "mixed_harm149_xstest250")[0].subset(keep249)
                    assert it.nh == 149 and it.nb == 100
                else:
                    it, _ = get(sc, pool, "rcdp-4state", ds)
                bt = Boot(it.nh, it.nb, a.B, a.seed)
                for name, s in (("raw", it.raw), ("head", it.head), ("platt", it.cal)):
                    e0, eci, b0, bci = bt.ece_brier(s, it.y)
                    au = float(bt.auc(s, it.y)[0])
                    res["calibration"].append({"scorer": sc, "pool": pool, "data": ds, "score": name, "ece": e0, "ece_ci": eci,
                                               "brier": b0, "brier_ci": bci, "auc": au, "n": len(s)})
                    md.append(f"| {sc} | {pool} | {ds} | {name} | {e0:.3f} [{eci[0]:.3f}, {eci[1]:.3f}] | "
                              f"{b0:.3f} [{bci[0]:.3f}, {bci[1]:.3f}] | {au:.4f} |")
    # global-temperature baselines (held-out)
    for sc, f in (("DistilBART", "global_temperature_distilbart.json"), ("DeBERTa", "global_temperature_deberta.json")):
        d = json.loads((root / BASE / f).read_text(encoding="utf-8"))
        rec = sorted(d["records"]["heldout"], key=lambda r: (r["label"] != "harmful", r["example_id"]))
        y = np.array([r["label"] == "harmful" for r in rec], dtype=float)
        s = np.array([r["prob_T"] for r in rec])
        bt = Boot(int(y.sum()), int(len(y) - y.sum()), a.B, a.seed)
        e0, eci, b0, bci = bt.ece_brier(s, y)
        assert abs(e0 - d["summary"]["heldout"]["ece10_prob_T"]) < 1e-9 and abs(b0 - d["summary"]["heldout"]["brier_prob_T"]) < 1e-9
        blk = np.array([r["decision"] == "block" for r in rec])
        res["calibration"].append({"scorer": sc, "pool": 196, "data": "heldout", "score": "global-temperature", "ece": e0,
                                   "ece_ci": eci, "brier": b0, "brier_ci": bci, "auc": float(bt.auc(s, y)[0]), "n": len(s),
                                   "blocked": int(blk[y == 1].sum()), "allowed": int((~blk[y == 0]).sum())})
        md.append(f"| {sc} | 196 | heldout | global-temperature | {e0:.3f} [{eci[0]:.3f}, {eci[1]:.3f}] | {b0:.3f} [{bci[0]:.3f}, {bci[1]:.3f}] | "
                  f"{res['calibration'][-1]['auc']:.4f} |")
    md.append("")

    # 3. scorer contrasts at fitted and matched operating points (pool 196, agnostic, RCDP 4-state)
    md += ["## Scorer contrasts (B minus A; pool 196, agnostic, RCDP)", ""]
    res["scorer_contrasts"] = []
    for ds, tg in (("heldout", (0.80, 0.90)), ("mixed_harm149_xstest250", (0.80, 0.90, 0.95))):
        arms = {sc: get(sc, 196, "rcdp-4state", ds)[0] for sc in ("DistilBART", "DeBERTa", "LG-prob")}
        bt = Boot(arms["DistilBART"].nh, arms["DistilBART"].nb, a.B, a.seed)
        for A, B_ in (("DistilBART", "DeBERTa"), ("DistilBART", "LG-prob"), ("DeBERTa", "LG-prob")):
            c = contrast(bt, arms[A], arms[B_], A, B_, tg)
            c["data"] = ds
            res["scorer_contrasts"].append(c)
            f = lambda x: f"{100 * x:+.2f}"
            md.append(f"- {ds}: {B_} - {A}: ABR {c['ABR']['A']:.4f} -> {c['ABR']['B']:.4f} ({f(c['ABR']['delta'])} pp, CI "
                      f"[{f(c['ABR']['ci'][0])}, {f(c['ABR']['ci'][1])}], disc {c['ABR']['discordant_A_only_B_only']}, p {c['ABR']['mcnemar_p']:.2g}); "
                      f"BAR {c['BAR']['A']:.4f} -> {c['BAR']['B']:.4f} ({f(c['BAR']['delta'])}, [{f(c['BAR']['ci'][0])}, {f(c['BAR']['ci'][1])}], "
                      f"disc {c['BAR']['discordant_A_only_B_only']}, p {c['BAR']['mcnemar_p']:.2g}); AUC {c['AUC']['A']:.4f} -> {c['AUC']['B']:.4f} "
                      f"({c['AUC']['delta']:+.4f} [{c['AUC']['ci'][0]:+.4f}, {c['AUC']['ci'][1]:+.4f}])")
            for key in ("matched_at_BAR_of_A", "matched_at_ABR_of_A"):
                m = c[key]
                md.append(f"    - {key} (target {m['target']:.4f}): A {m['A']:.4f} (realised {m['A_realised']:.4f}), B {m['B']:.4f} "
                          f"(realised {m['B_realised']:.4f}), delta {f(m['delta'])} pp [{f(m['ci'][0])}, {f(m['ci'][1])}]")
            for m in c["matched_at_BAR_targets"]:
                md.append(f"    - ABR at BAR >= {m['target']:.2f}: A {m['A']:.4f} (BAR {m['A_realised']:.4f}), B {m['B']:.4f} "
                          f"(BAR {m['B_realised']:.4f}), delta {f(m['delta'])} pp [{f(m['ci'][0])}, {f(m['ci'][1])}]")
    md.append("")

    # 4. calibration-pool contrasts (same pipeline, 196 -> 346), held-out and the disjoint 249-item subset
    md += ["## Calibration-pool contrasts (pool 346 minus pool 196; agnostic)", ""]
    res["pool_contrasts"] = []
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        for var in ("legacy-submitted", "rcdp-4state"):
            for ds196, ds346 in (("heldout", "heldout"), ("mixed_harm149_xstest250", "mixed_harm149_xstest_test100")):
                a346, _ = get(sc, 346, var, ds346)
                a196, _ = get(sc, 196, var, ds196)
                if ds196 != ds346:
                    keep = set(a346.ids)
                    assert keep <= set(a196.ids) and len(keep) == 249
                    a196 = a196.subset(keep)
                bt = Boot(a196.nh, a196.nb, a.B, a.seed)
                c = contrast(bt, a196, a346, f"{var}@196", f"{var}@346")
                c.update({"scorer": sc, "pipeline": var, "data": ds346})
                for k in ("matched_at_BAR_of_A", "matched_at_ABR_of_A", "matched_at_BAR_targets"):
                    c.pop(k)
                res["pool_contrasts"].append(c)
                f = lambda x: f"{100 * x:+.2f}"
                md.append(f"- {sc} {var} {ds346}: ABR {c['ABR']['A']:.4f} -> {c['ABR']['B']:.4f} ({f(c['ABR']['delta'])} [{f(c['ABR']['ci'][0])}, "
                          f"{f(c['ABR']['ci'][1])}], disc {c['ABR']['discordant_A_only_B_only']}, p {c['ABR']['mcnemar_p']:.2g}); BAR {c['BAR']['A']:.4f} -> "
                          f"{c['BAR']['B']:.4f} ({f(c['BAR']['delta'])} [{f(c['BAR']['ci'][0])}, {f(c['BAR']['ci'][1])}], disc {c['BAR']['discordant_A_only_B_only']}, "
                          f"p {c['BAR']['mcnemar_p']:.2g}); AUC {c['AUC']['A']:.4f} -> {c['AUC']['B']:.4f}")
    md.append("")

    # 5. scorer-only baselines recounted from their records (held-out)
    md += ["## Scorer-only baselines, recounted (held-out)", ""]
    res["baselines"] = {}
    for name, f, sub, key in (("global-temperature DistilBART", "global_temperature_distilbart.json", None, "prob_T"),
                              ("global-temperature DeBERTa", "global_temperature_deberta.json", None, "prob_T"),
                              ("Llama-Guard one-shot verdict", "llama_guard_oneshot.json", None, "p_unsafe"),
                              ("Llama-Guard P(unsafe) >= tau", "llama_guard_oneshot.json", "prob_threshold_supplementary", "p_unsafe")):
        d = json.loads((root / BASE / f).read_text(encoding="utf-8"))
        rec = d[sub]["records"] if sub else d["records"]["heldout"]
        y = np.array([r["label"] == "harmful" for r in rec])
        blk = np.array([r["decision"] == "block" for r in rec])
        s = np.array([r[key] for r in rec])
        W = (s[y][:, None] > s[~y][None, :]) + 0.5 * (s[y][:, None] == s[~y][None, :])
        res["baselines"][name] = {"blocked": int(blk[y].sum()), "n_h": int(y.sum()), "allowed": int((~blk[~y]).sum()),
                                  "n_b": int((~y).sum()), "auc": float(W.mean()), "tau": d[sub]["tau"] if sub else d.get("tau")}
        r_ = res["baselines"][name]
        md.append(f"- {name}: {r_['blocked']}/{r_['n_h']}, {r_['allowed']}/{r_['n_b']}, AUC {r_['auc']:.4f}, tau {r_['tau']}")
    md.append("")

    # 6. RCDP against the threshold-only baseline on the same scorer (fitted points, held-out), and the head's AUC gain
    md += ["## RCDP versus a threshold on the same scorer (held-out, fitted points; B minus A)", ""]
    res["rcdp_vs_threshold"] = []
    def base_blocked(f, sub, order):
        d = json.loads((root / BASE / f).read_text(encoding="utf-8"))
        rec = {r["example_id"]: r for r in (d[sub]["records"] if sub else d["records"]["heldout"])}
        return np.array([rec[e]["decision"] == "block" for e in order])
    pairs = (("global-temperature DistilBART", "global_temperature_distilbart.json", None, "DistilBART"),
             ("global-temperature DeBERTa", "global_temperature_deberta.json", None, "DeBERTa"),
             ("Llama-Guard P(unsafe) >= tau", "llama_guard_oneshot.json", "prob_threshold_supplementary", "LG-prob"),
             ("Llama-Guard one-shot verdict", "llama_guard_oneshot.json", None, "LG-verdict"),
             ("Llama-Guard one-shot verdict", "llama_guard_oneshot.json", None, "DistilBART"))
    for name, f, sub, sc in pairs:
        it, _ = get(sc, 196, "rcdp-4state", "heldout")
        bt = Boot(it.nh, it.nb, a.B, a.seed)
        A = base_blocked(f, sub, it.ids)
        y = it.y
        ra, rb = bt.rates(A, y), bt.rates(it.blocked, y)
        row = {"A": name, "B": f"RCDP {sc}"}
        for j, m in ((0, "ABR"), (1, "BAR")):
            sel = (y == 1) if m == "ABR" else (y == 0)
            if m == "ABR":
                b = int(np.sum(A[sel] & ~it.blocked[sel])); c = int(np.sum(~A[sel] & it.blocked[sel]))
            else:
                b = int(np.sum(~A[sel] & it.blocked[sel])); c = int(np.sum(A[sel] & ~it.blocked[sel]))
            row[m] = {"A": float(ra[j]), "B": float(rb[j]), "delta": float(rb[j] - ra[j]), "ci": pct(rb[2 + j] - ra[2 + j], 1),
                      "discordant_A_only_B_only": [b, c], "mcnemar_p": mcnemar(b, c)}
        res["rcdp_vs_threshold"].append(row)
        g = lambda x: f"{100 * x:+.2f}"
        md.append(f"- {row['B']} vs {name}: ABR {row['ABR']['A']:.4f} -> {row['ABR']['B']:.4f} ({g(row['ABR']['delta'])} [{g(row['ABR']['ci'][0])}, "
                  f"{g(row['ABR']['ci'][1])}], disc {row['ABR']['discordant_A_only_B_only']}, p {row['ABR']['mcnemar_p']:.2g}); BAR {row['BAR']['A']:.4f} -> "
                  f"{row['BAR']['B']:.4f} ({g(row['BAR']['delta'])} [{g(row['BAR']['ci'][0])}, {g(row['BAR']['ci'][1])}], disc "
                  f"{row['BAR']['discordant_A_only_B_only']}, p {row['BAR']['mcnemar_p']:.2g})")
    md += ["", "## Risk head: AUC of the calibrated score minus AUC of the raw score (pool 196, agnostic)", ""]
    res["head_auc_gain"] = []
    for sc in ("DistilBART", "DeBERTa", "LG-prob"):
        for ds in ("heldout", "mixed_harm149_xstest250"):
            it, _ = get(sc, 196, "rcdp-4state", ds)
            bt = Boot(it.nh, it.nb, a.B, a.seed)
            c0, cr = bt.auc(it.cal, it.y)
            r0, rr = bt.auc(it.raw, it.y)
            res["head_auc_gain"].append({"scorer": sc, "data": ds, "auc_cal": c0, "auc_raw": r0, "delta": c0 - r0, "ci": pct(cr - rr, 1)})
            md.append(f"- {sc} {ds}: {r0:.4f} -> {c0:.4f} ({c0 - r0:+.4f} [{pct(cr - rr, 1)[0]:+.4f}, {pct(cr - rr, 1)[1]:+.4f}])")
    md.append("")

    (out_dir / "single_turn_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "single_turn_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
