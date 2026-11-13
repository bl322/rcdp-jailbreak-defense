"""Memory on vs off at matched benign allowance on an independent multi-turn set (ledger R3-4).

Input: <output>.<setting>.delta_outcomes.json written by run_multiturn_cases.py --export-delta-outcomes
(evaluation-side blocked flag of every session at every shift of the blocking threshold).

Analysis fixed before any result was seen (authors' decision 2026-11-14):
  * targets: session-level BAR (all benign dialogues pooled) >= 0.80, 0.90, 0.95;
  * per variant and target: the threshold shift with the highest ABR among shifts reaching the target
    (ties: higher BAR, smaller |delta|, smaller delta), chosen on the evaluation dialogues themselves,
    identically for every variant; if no shift reaches the target, the shift with the highest BAR is
    used and the row is flagged;
  * contrasts (memory on - off, all else equal): 4-state+memory - 4-state-downgrade,
    2-state+memory - 2-state-downgrade;
  * uncertainty: clusters resampled within label (harmful: seed behaviour; benign: dialogue),
    B = 10000, seed 20261114. The matching is repeated inside every resample ("re-matched" CI);
    the CI with the shift held at its point-estimate value ("conditional") is reported as well;
  * BAR is also reported per benign benchmark at each matched shift; nothing is dropped.

    python scripts/multiturn_matched_external.py results/r1/mt_ext/ext_attack600_dist.agnostic.delta_outcomes.json \
        --output results/r1/mt_ext/matched_attack600_dist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_PAIRS = "4-state+memory:4-state-downgrade,2-state+memory:2-state-downgrade"


def choose(abr, bar, deltas, target):
    """Index of the matched shift; returns (index, reached)."""
    feas = np.flatnonzero(bar >= target - 1e-12)
    if feas.size:
        keys = (-deltas[feas], -np.abs(deltas[feas]), bar[feas], abr[feas])  # np.lexsort: last key is primary
        return int(feas[np.lexsort(keys)[-1]]), True
    keys = (-np.abs(deltas), abr, bar)
    return int(np.lexsort(keys)[-1]), False


def rates(B, w_h, w_b, h, b):
    """ABR and BAR per shift for session weights (vectors or matrices with one column per resample)."""
    blocked_h = B[:, h] @ w_h
    blocked_b = B[:, b] @ w_b
    return blocked_h / w_h.sum(axis=0), 1.0 - blocked_b / w_b.sum(axis=0)


def cluster_weights(rng, cl_index, n_clusters, n_rep):
    """Session weights (sessions x n_rep): each session counts as often as its cluster is drawn."""
    draws = rng.integers(0, n_clusters, size=(n_rep, n_clusters))
    counts = np.zeros((n_rep, n_clusters), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n_rep), n_clusters), draws.ravel()), 1.0)
    return counts[:, cl_index].T


def summarize_ci(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0)))
    return {"lo": float(lo), "hi": float(hi), "p_two_sided": float(p)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--targets", default="0.80,0.90,0.95")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="comma list of on:off variant pairs")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args(argv)

    d = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sessions = d["sessions"]
    labels = np.array([s["label"] for s in sessions])
    h, b = np.flatnonzero(labels == "harmful"), np.flatnonzero(labels == "benign")
    if not h.size or not b.size:
        raise SystemExit("the evaluation side needs both harmful and benign dialogues")
    benches = sorted({s["benchmark"] for s in sessions if s["label"] == "benign"})
    targets = [float(t) for t in args.targets.split(",")]
    pairs = [tuple(x.split(":")) for x in args.pairs.split(",")]
    names = sorted({n for pr in pairs for n in pr})
    missing = [n for n in names if n not in d["variants"]]
    if missing:
        raise SystemExit(f"variants missing from the export: {missing}")

    mats, dels = {}, {}
    for n in names:
        pts = d["variants"][n]
        mats[n] = np.array([[c == "1" for c in p["blocked"]] for p in pts], dtype=np.float64)
        dels[n] = np.array([p["delta"] for p in pts], dtype=float)
        if mats[n].shape[1] != len(sessions):
            raise SystemExit(f"{n}: {mats[n].shape[1]} flags per shift for {len(sessions)} sessions")

    # point estimates
    ones_h, ones_b = np.ones(h.size), np.ones(b.size)
    point = {}
    for n in names:
        abr, bar = rates(mats[n], ones_h, ones_b, h, b)
        point[n] = {}
        for t in targets:
            k, reached = choose(abr, bar, dels[n], t)
            row = mats[n][k]
            per = {}
            for bn in benches:
                idx = [i for i in b if sessions[i]["benchmark"] == bn]
                per[bn] = {"allowed": int(len(idx) - row[idx].sum()), "n": len(idx)}
            point[n][t] = {"index": k, "delta": float(dels[n][k]), "target_reached": reached,
                           "blocked_harmful": int(row[h].sum()), "n_harmful": int(h.size),
                           "allowed_benign": int(b.size - row[b].sum()), "n_benign": int(b.size),
                           "abr": float(abr[k]), "bar": float(bar[k]), "bar_by_benchmark": per}

    # cluster bootstrap within label
    rng = np.random.default_rng(args.seed)
    def cl_map(idx):
        cl = [sessions[i]["cluster_id"] for i in idx]
        uniq = {c: j for j, c in enumerate(sorted(set(cl)))}
        return np.array([uniq[c] for c in cl]), len(uniq)
    ch, nch = cl_map(h)
    cb, ncb = cl_map(b)
    rem = {(pr, t): [] for pr in pairs for t in targets}
    cond = {(pr, t): [] for pr in pairs for t in targets}
    unreached = {(n, t): 0 for n in names for t in targets}
    done = 0
    while done < args.bootstrap:
        m = min(args.batch, args.bootstrap - done)
        wh, wb = cluster_weights(rng, ch, nch, m), cluster_weights(rng, cb, ncb, m)
        rr = {n: rates(mats[n], wh, wb, h, b) for n in names}
        for j in range(m):
            for t in targets:
                abr_at = {}
                for n in names:
                    abr, bar = rr[n][0][:, j], rr[n][1][:, j]
                    k, reached = choose(abr, bar, dels[n], t)
                    unreached[(n, t)] += not reached
                    abr_at[n] = (abr[k], abr[point[n][t]["index"]])
                for pr in pairs:
                    rem[(pr, t)].append(abr_at[pr[0]][0] - abr_at[pr[1]][0])
                    cond[(pr, t)].append(abr_at[pr[0]][1] - abr_at[pr[1]][1])
        done += m

    out = {"input": args.input, "setting": d.get("setting"), "scorer": d.get("scorer"), "dataset": d.get("dataset"),
           "n_harmful": int(h.size), "n_benign": int(b.size), "harmful_clusters": nch, "benign_clusters": ncb,
           "benign_benchmarks": benches, "targets": targets, "B": args.bootstrap, "seed": args.seed,
           "rule": __doc__.split("Analysis fixed before any result was seen", 1)[1].split("    python", 1)[0].strip(),
           "variants": {n: {str(t): {k: v for k, v in point[n][t].items() if k != "index"} for t in targets}
                        for n in names},
           "unreached_share_in_resamples": {f"{n}@{t}": unreached[(n, t)] / args.bootstrap
                                            for n in names for t in targets},
           "contrasts": []}
    print(f"{Path(args.input).name}: {h.size} harmful ({nch} clusters), {b.size} benign ({ncb} clusters); "
          f"benign benchmarks {benches}")
    for pr in pairs:
        for t in targets:
            a, c = point[pr[0]][t], point[pr[1]][t]
            diff = a["abr"] - c["abr"]
            row = {"on": pr[0], "off": pr[1], "target_bar": t, "abr_on": a["abr"], "abr_off": c["abr"],
                   "bar_on": a["bar"], "bar_off": c["bar"], "delta_abr": diff,
                   "rematched_ci": summarize_ci(rem[(pr, t)]), "conditional_ci": summarize_ci(cond[(pr, t)]),
                   "targets_reached": [a["target_reached"], c["target_reached"]]}
            out["contrasts"].append(row)
            per_on = " ".join(f"{k} {v['allowed']}/{v['n']}" for k, v in a["bar_by_benchmark"].items())
            per_off = " ".join(f"{k} {v['allowed']}/{v['n']}" for k, v in c["bar_by_benchmark"].items())
            print(f"  BAR>={t:.2f}  {pr[0]} {a['blocked_harmful']}/{a['n_harmful']} BAR {a['bar']:.4f} [{per_on}]"
                  f"{'' if a['target_reached'] else ' (target not reached)'}\n"
                  f"             {pr[1]} {c['blocked_harmful']}/{c['n_harmful']} BAR {c['bar']:.4f} [{per_off}]"
                  f"{'' if c['target_reached'] else ' (target not reached)'}\n"
                  f"             dABR {diff * 100:+.2f} pp  re-matched 95% CI "
                  f"[{row['rematched_ci']['lo'] * 100:+.2f}, {row['rematched_ci']['hi'] * 100:+.2f}] "
                  f"p={row['rematched_ci']['p_two_sided']:.4f} | conditional "
                  f"[{row['conditional_ci']['lo'] * 100:+.2f}, {row['conditional_ci']['hi'] * 100:+.2f}]")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
