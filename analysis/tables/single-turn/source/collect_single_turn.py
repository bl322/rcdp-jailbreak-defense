# -*- coding: utf-8 -*-
"""Collect every single-turn number used in Section 4.2, Table 1, Table 2 and the supplementary single-turn tables
from the R1 result files, recomputing counts from per-example records and checking them against the stored aggregates.

    python collect_single_turn.py --root <project root>
Writes build/single_turn_summary.md (all runs) and build/single_turn.json (machine-readable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {
    # key: (scorer label, pool, relative path)
    "dist196": ("DistilBART-MNLI", 196, "_r1_out/stage1json/main_dist_full_pool_calibration.json"),
    "deb196": ("DeBERTa-v3-large", 196, "_r1_out/extracted/results/r1/main_deb_full_pool_calibration.json"),
    "lg196": ("Llama-Guard-3-1B (verdict)", 196, "_r1_out/stage1json/main_lg_full_pool_calibration.json"),
    "lgp196": ("Llama-Guard-3-1B (prob.)", 196, "_r1_out/lgprob/main_lgp_full_pool_calibration.json"),
    "dist346": ("DistilBART-MNLI", 346, "_r1_out/extracted/results/r1/main_dist_calibration_v2_external_benign.json"),
    "deb346": ("DeBERTa-v3-large", 346, "_r1_out/extracted/results/r1/main_deb_calibration_v2_external_benign.json"),
    "lgp346": ("Llama-Guard-3-1B (prob.)", 346, "_r1_out/lgprob/main_lgp_calibration_v2_external_benign.json"),
}
BASE = "_r1_out/stage2_cost/results/r1/cost/baselines/"
STATES = ("quarantine", "hardened", "cautious", "normal")


def recount(ev):
    recs = ev["records"]
    h = [r for r in recs if r["label"] == "harmful"]
    b = [r for r in recs if r["label"] == "benign"]
    bh = sum(bool(r["blocked_any_turn"]) for r in h)
    ab = sum(not r["blocked_any_turn"] for r in b)
    assert (bh, len(h), ab, len(b)) == (ev["blocked_harmful"], ev["n_harmful"], ev["allowed_benign"], ev["n_benign"]), "aggregate mismatch"
    sc = {s: sum(v for k, v in ev["state_counts"].items() if k.endswith("/" + s)) for s in STATES}
    rel = ev.get("reliability") or {}
    return {"blocked": bh, "n_h": len(h), "allowed": ab, "n_b": len(b), "auc_cal": ev.get("auc_calibrated"),
            "auc_raw": ev.get("auc_raw"), "ece": rel.get("ece10"), "brier": rel.get("brier"),
            "tokens": ev.get("mean_prompt_tokens"), "states": sc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out = HERE.parent / "build"
    out.mkdir(parents=True, exist_ok=True)
    allres, lines = {}, ["# Single-turn results (recomputed from per-example records)", ""]
    for key, (scorer, pool, rel) in RUNS.items():
        d = json.loads((root / rel).read_text(encoding="utf-8"))
        code = d["meta"].get("code_version", {}).get("sha256_rcdp_guard_and_runner", "?")[:12]
        lines += [f"## {key}: {scorer}, pool {pool} (`{rel}`, code {code})", "",
                  "| variant | setting | dataset | blocked | allowed | AUC cal/raw | ECE | Brier | tokens | Q/H/C/N |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for v in d["variants"]:
            for ds, ev in v["eval"].items():
                r = recount(ev)
                allres[f"{key}|{v['variant']}|{v['setting']}|{ds}"] = r
                s = r["states"]
                fmt = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
                lines.append(f"| {v['variant']} | {v['setting']} | {ds} | {r['blocked']}/{r['n_h']} | {r['allowed']}/{r['n_b']} | "
                             f"{fmt(r['auc_cal'])}/{fmt(r['auc_raw'])} | {fmt(r['ece'], 3)} | {fmt(r['brier'], 3)} | "
                             f"{fmt(r['tokens'], 2)} | {s['quarantine']}/{s['hardened']}/{s['cautious']}/{s['normal']} |")
        lines.append("")
    # scorer-only baselines (stage 2)
    lines += ["## Scorer-only baselines (stage 2, held-out 149/55)", "", "| baseline | blocked | allowed | AUC | ECE | Brier |", "|---|---|---|---|---|---|"]
    for name, f, sub in (("Global temperature + threshold, DistilBART", "global_temperature_distilbart.json", None),
                         ("Global temperature + threshold, DeBERTa", "global_temperature_deberta.json", None),
                         ("Llama-Guard-3-1B one-shot verdict", "llama_guard_oneshot.json", None),
                         ("Llama-Guard-3-1B P(unsafe) >= tau (D_thresh)", "llama_guard_oneshot.json", "prob_threshold_supplementary")):
        d = json.loads((root / BASE / f).read_text(encoding="utf-8"))
        h = (d[sub]["heldout"] if sub else d["summary"]["heldout"])
        a, b = h["attack_block_rate"], h["benign_allow_rate"]
        r = {"blocked": a["k"], "n_h": a["n"], "allowed": b["k"], "n_b": b["n"], "auc": h.get("auc"),
             "ece": h.get("ece10_prob_T"), "brier": h.get("brier_prob_T"), "tau": d[sub]["tau"] if sub else d.get("tau")}
        allres[f"baseline|{name}"] = r
        fmt = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
        lines.append(f"| {name} | {r['blocked']}/{r['n_h']} | {r['allowed']}/{r['n_b']} | {fmt(r['auc'])} | {fmt(r['ece'], 3)} | {fmt(r['brier'], 3)} |")
    (out / "single_turn_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "single_turn.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
