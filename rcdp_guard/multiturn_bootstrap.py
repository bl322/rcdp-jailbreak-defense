"""Seed-clustered bootstrap for session-level ABR/BAR (generalises b5_cluster_bootstrap.py).

Same algorithm, constants and draw order as the audit script
``responses/rcdp-cmc-r1/source/analysis/b5_cluster_bootstrap.py`` (session level):
for ABR (harmful sessions) and then BAR (benign sessions), seed clusters are
resampled with replacement *within the label*, all sessions of a cluster are
kept together, and the same draws are used for every column so differences are
paired. A naive i.i.d. session bootstrap is drawn next. B = 10,000,
numpy.random.default_rng(20261113), percentile 95% intervals. On the six
b5 variants (in b5's order) the results coincide with b5's "session" block.

Columns are {name: [(cluster_id, label, success), ...]} in a fixed session
order; success = blocked (harmful) or fully allowed (benign). All columns must
contain the same sessions in the same order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

B_DEFAULT = 10_000
SEED_DEFAULT = 20261113


def _np():
    import numpy as np  # local import: the runner works without numpy if --no-bootstrap

    return np


def _pct(np, x) -> List[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def cluster_bootstrap(
    columns: Dict[str, Sequence[Tuple[str, str, bool]]],
    contrasts: Sequence[Tuple[str, str]] = (),
    *,
    B: int = B_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Dict[str, Any]:
    np = _np()
    names = list(columns)
    if not names:
        return {}
    ref = [(c, lab) for c, lab, _ in columns[names[0]]]
    for v in names[1:]:
        if [(c, lab) for c, lab, _ in columns[v]] != ref:
            raise ValueError(f"column {v} does not cover the same sessions in the same order")
    rng = np.random.default_rng(seed)
    res: Dict[str, Any] = {"B": B, "rng_seed": seed, "level": "session", "point": {}, "clustered": {},
                           "naive": {}, "diff": {}, "n_clusters": {}, "n_units": {}}
    for metric, label in (("ABR", "harmful"), ("BAR", "benign")):
        clusters = sorted({c for c, lab in ref if lab == label})
        n = len(clusters)
        if n == 0:
            continue
        pos = {c: k for k, c in enumerate(clusters)}
        cl_idx = rng.integers(0, n, size=(B, n))
        units: Dict[str, Any] = {}
        km: Dict[str, Tuple[Any, Any]] = {}
        for v in names:
            K = np.zeros(n)
            M = np.zeros(n)
            vec = []
            for c, lab, ok in columns[v]:
                if lab != label:
                    continue
                K[pos[c]] += float(ok)
                M[pos[c]] += 1.0
                vec.append(float(ok))
            km[v] = (K, M)
            units[v] = np.array(vec)
        nu = len(units[names[0]])
        nv_idx = rng.integers(0, nu, size=(B, nu))
        boot_cl, boot_nv = {}, {}
        for v in names:
            K, M = km[v]
            boot_cl[v] = K[cl_idx].sum(1) / M[cl_idx].sum(1)
            boot_nv[v] = units[v][nv_idx].mean(1)
            ci_c, ci_n = _pct(np, boot_cl[v]), _pct(np, boot_nv[v])
            res["point"].setdefault(v, {})[metric] = float(K.sum() / M.sum())
            res["point"][v][metric + "_k_n"] = [int(K.sum()), int(M.sum())]
            res["clustered"].setdefault(v, {})[metric] = ci_c
            res["naive"].setdefault(v, {})[metric] = ci_n
            res["clustered"][v][metric + "_width"] = ci_c[1] - ci_c[0]
            res["naive"][v][metric + "_width"] = ci_n[1] - ci_n[0]
            res["clustered"][v][metric + "_within_seed_all_agree"] = float(np.mean((K == 0) | (K == M)))
        for a, b in contrasts:
            if a not in boot_cl or b not in boot_cl:
                continue
            d = boot_cl[a] - boot_cl[b]
            res["diff"].setdefault(f"{a} - {b}", {})[metric] = {
                "point": res["point"][a][metric] - res["point"][b][metric],
                "clustered_paired": _pct(np, d),
                "naive_paired": _pct(np, boot_nv[a] - boot_nv[b]),
                "clustered_p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
            }
        res["n_clusters"][metric] = n
        res["n_units"][metric] = nu
    return res


def format_table(res: Dict[str, Any]) -> str:
    lines = []
    for v, p in res.get("point", {}).items():
        c = res["clustered"][v]
        abr = p.get("ABR")
        bar = p.get("BAR")
        lines.append(
            f"{v:28s} ABR {abr if abr is None else round(abr, 4)} cl{[round(x, 4) for x in c.get('ABR', [])]} | "
            f"BAR {bar if bar is None else round(bar, 4)} cl{[round(x, 4) for x in c.get('BAR', [])]}"
        )
    for k, d in res.get("diff", {}).items():
        parts = []
        for m in ("ABR", "BAR"):
            if m in d:
                parts.append(f"d{m} {d[m]['point']:+.4f} cl{[round(x, 4) for x in d[m]['clustered_paired']]}")
        lines.append(f"  {k:50s} " + " | ".join(parts))
    return "\n".join(lines)
