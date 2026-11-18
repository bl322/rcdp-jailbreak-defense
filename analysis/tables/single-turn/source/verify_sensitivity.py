# -*- coding: utf-8 -*-
"""Gate-only sensitivity runs (r4 package, --skip-generation, 204 held-out prompts): calibration split seed, Platt
ablation and a threshold on the raw score. Recounted from records; each block is checked against the rule
'block iff the routed state is Quarantine' and the stored effective risk against tau_q. Paired McNemar against the
reference R1 gate (seed rcdp-r1) from the end-to-end files.

    python -I -X utf8 verify_sensitivity.py [--root <project root>]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = "_r1_out/r4/extracted/r4_pack/sens/"
REF = {"dist": "_r1_out/e2e_rcdp/extracted/results/r1/e2e/rcdp_qwen_dist_agnostic.json",
       "lgprob": "_r1_out/e2e_rcdp/extracted/results/r1/e2e/rcdp_qwen_lgprob_agnostic.json"}
RUNS = ("seed_r2", "seed_r3", "noplatt", "rawthresh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    sys.path.insert(0, str(root / "ccfa-workfiles/tables/e2e/source"))
    import collect_e2e as C
    L, out = ["# Gate-only sensitivity (recounted)", ""], {}
    for sc in ("dist", "lgprob"):
        ref = {r["example_id"]: r for r in json.loads((root / REF[sc]).read_text(encoding="utf-8"))["records"]}
        rb = {i: r["outcome"]["blocked_pre_generation"] for i, r in ref.items()}
        L.append(f"- reference {sc} (seed rcdp-r1): blocked {sum(rb[i] for i in rb if ref[i]['label'] == 'harmful')}/149, "
                 f"allowed {sum(not rb[i] for i in rb if ref[i]['label'] == 'benign')}/55")
        for run in RUNS:
            d = json.loads((root / S / f"{sc}_{run}.json").read_text(encoding="utf-8"))
            recs = {r["example_id"]: r for r in d["records"]}
            assert len(recs) == 204 and set(recs) == set(ref)
            th = d["gate"]["thresholds"]
            bad = 0
            for r in recs.values():
                blk = r["outcome"]["blocked_pre_generation"]
                assert blk == (r["state"] == "quarantine") == (r["controller_decision"] == "block"), (sc, run, r["example_id"])
                eff = r["gate"]["details"]["effective_risk"]
                bad += (eff >= max(th["tau_q"], 0.35) - 1e-12) != blk  # single-turn floor of the Quarantine threshold (Section 3.4)
            H = [i for i in recs if recs[i]["label"] == "harmful"]; B = [i for i in recs if recs[i]["label"] == "benign"]
            bh = sum(recs[i]["outcome"]["blocked_pre_generation"] for i in H)
            ab = sum(not recs[i]["outcome"]["blocked_pre_generation"] for i in B)
            s = d["summary"]
            assert s["harmful"]["abr_ctrl"]["k"] == bh, (sc, run)
            h_a = sum(recs[i]["outcome"]["blocked_pre_generation"] and not rb[i] for i in H)
            h_b = sum(rb[i] and not recs[i]["outcome"]["blocked_pre_generation"] for i in H)
            b_a = sum(not recs[i]["outcome"]["blocked_pre_generation"] and rb[i] for i in B)
            b_b = sum(not rb[i] and recs[i]["outcome"]["blocked_pre_generation"] for i in B)
            cal = d["calibration_report"]
            out[f"{sc}_{run}"] = {"protocol": d["configuration"]["calibration_protocol"], "seed": d["configuration"].get("calibration_seed"),
                                  "blocked": bh, "allowed": ab, "tau": th, "rule_violations": bad,
                                  "harmful_disc_[only_this,only_ref]": [h_a, h_b], "p_harmful": C.mcnemar(h_a, h_b),
                                  "benign_disc_[allowed_only_this,allowed_only_ref]": [b_a, b_b], "p_benign": C.mcnemar(b_a, b_b),
                                  "platt": cal.get("platt"), "threshold_selected": cal.get("threshold_selected")}
            o = out[f"{sc}_{run}"]
            L.append(f"  - {sc}_{run} ({o['protocol']}, seed {o['seed']}): blocked {bh}/149, allowed {ab}/55; "
                     f"tau_q {th['tau_q']:.4g}; rule violations {bad}; vs ref harmful {h_a}/{h_b} (p={C.fmt_p(o['p_harmful'])}), "
                     f"benign allowed {b_a}/{b_b} (p={C.fmt_p(o['p_benign'])})")
            # threshold search recomputed from the D_thresh records (seeded runs; the raw-threshold run searches raw scores)
            ts = cal.get("threshold_search") or {}
            recs_t = [r for r in cal.get("calibration_records", []) if r["subset"] == "thresh"]
            if ts.get("table") and recs_t and run != "rawthresh":
                hs = [r["calibrated_score"] for r in recs_t if r["label"] == "harmful"]
                bs = [r["calibrated_score"] for r in recs_t if r["label"] == "benign"]
                tab = {round(row["tau"], 2): row for row in ts["table"]}
                mism = 0
                for tau, row in tab.items():
                    eff = max(tau, 0.35)  # single-turn floor of the Quarantine threshold, as in the rule check above
                    k_h = sum(x >= eff - 1e-12 for x in hs); k_b = sum(x >= eff - 1e-12 for x in bs)
                    mism += (k_h, k_b) != (row["blocked_harmful"], row["blocked_benign"])
                sel = ts["selected"]
                below = [row for t_, row in tab.items() if t_ < sel["tau"]]
                best_below = max(below, key=lambda r: r["J"]) if below else None
                o["thresh_search"] = {"n_harmful": len(hs), "n_benign": len(bs), "grid_mismatches": mism, "selected": sel,
                                      "best_below_selected": best_below,
                                      "benign_scores_desc": sorted(bs, reverse=True)[:3]}
                L.append(f"    D_thresh {len(hs)} harmful / {len(bs)} benign; grid recount mismatches {mism}; selected tau "
                         f"{sel['tau']} blocks {sel['blocked_harmful']}/{len(hs)} harmful, {sel['blocked_benign']}/{len(bs)} "
                         f"benign (J {sel['J']:.4f}); best below: tau {best_below['tau'] if best_below else None} "
                         f"J {best_below['J'] if best_below else float('nan'):.4f} ({best_below['blocked_harmful'] if best_below else '-'}"
                         f"/{best_below['blocked_benign'] if best_below else '-'}); top benign calibrated scores "
                         f"{[round(x, 3) for x in sorted(bs, reverse=True)[:3]]}")
    b = HERE.parent / "build"
    (b / "sensitivity_gate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "sensitivity_gate.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
