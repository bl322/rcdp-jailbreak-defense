"""Statistics helpers for revision R1 (standard library only; no scipy, no numpy).

Functions
---------
clopper_pearson(k, n)                exact two-sided binomial interval for k/n
mcnemar_exact(a, b)                  paired exact McNemar test for two systems on the same items
bootstrap_ci(values, statistic)      percentile bootstrap (optionally stratified)
cluster_bootstrap_ci(items, ...)     percentile bootstrap that resamples whole clusters
paired_bootstrap_diff(a, b, metric)  paired bootstrap of metric(a) - metric(b) on the same items
abr / bar / fpr / auc                metric helpers for per-item record dicts
describe(xs)                         mean, sd, p50/p90/p95/p99 (linear interpolation, numpy's default)
resolve / example_records            read per-item records from any R1 result file
compare_systems(a, b)                McNemar + paired bootstrap of ABR/BAR/AUC differences

Determinism: every resampling function takes an integer ``seed`` and draws
indices with ``random.Random(seed).choices`` (one draw of n indices per
resample and stratum, strata in sorted order), so a given seed gives the same
interval on every platform. ``bootstrap_ci`` is ``cluster_bootstrap_ci`` with
one item per cluster, so the two agree exactly for singleton clusters.

Command line:
    python scripts/stats_r1.py cp 0 149
    python scripts/stats_r1.py mcnemar 1 9          # discordant counts b, c
    python scripts/stats_r1.py boot 135 149 --B 10000 --seed 20260903
    python scripts/stats_r1.py rate --input X.json --records records.heldout [--cluster-field seed_id]
    python scripts/stats_r1.py compare --a A.json --a-records records.heldout --a-score prob_T \
        --b B.json --b-records "variants.variant=rcdp-4state&setting=agnostic.eval.heldout.records" \
        --b-score calibrated_score --output results/r1/stats/A_vs_B.json
Record paths are dotted; a step into a list is an index or ``key=value[&key=value]``
(first matching element).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

__all__ = [
    "betainc_reg", "beta_ppf", "binom_cdf", "binom_sf", "clopper_pearson",
    "mcnemar_exact", "align_records", "quantile", "describe",
    "bootstrap_ci", "cluster_bootstrap_ci", "paired_bootstrap_diff",
    "abr", "bar", "fpr", "auc", "proportion_summary", "bootstrap_proportion_limit",
    "resolve", "example_records", "compare_systems",
]

# --------------------------------------------------------------------------
# Regularised incomplete beta function and the exact binomial interval
# --------------------------------------------------------------------------

_EPS = 1e-15
_TINY = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for I_x(a, b) (modified Lentz; Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise ArithmeticError(f"betacf did not converge for a={a}, b={b}, x={x}")


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) for a, b > 0."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-14) -> float:
    """Inverse of I_x(a, b) in x by bisection (I_x is increasing in x)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc_reg(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return betainc_reg(n - k, k + 1, 1.0 - p)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return betainc_reg(k, n - k + 1, p)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion k/n.

    lower = Beta^{-1}(alpha/2; k, n-k+1) (0 if k = 0),
    upper = Beta^{-1}(1-alpha/2; k+1, n-k) (1 if k = n).
    For k = 0 the upper bound is 1 - (alpha/2)^(1/n), e.g. 0/149 -> 0.02445.
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


# --------------------------------------------------------------------------
# Paired exact McNemar test
# --------------------------------------------------------------------------

def align_records(a: Sequence[dict], b: Sequence[dict], key: str = "example_id") -> Tuple[List[dict], List[dict]]:
    """Align two per-item record lists on ``key``; the item sets must be identical."""
    ma = {r[key]: r for r in a}
    mb = {r[key]: r for r in b}
    if len(ma) != len(a) or len(mb) != len(b):
        raise ValueError(f"duplicate {key} values in the record lists")
    if set(ma) != set(mb):
        only_a = sorted(set(ma) - set(mb))[:5]
        only_b = sorted(set(mb) - set(ma))[:5]
        raise ValueError(f"item sets differ (e.g. only in a: {only_a}; only in b: {only_b})")
    ids = [r[key] for r in a]
    return [ma[i] for i in ids], [mb[i] for i in ids]


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, object]:
    """Exact two-sided McNemar test on paired binary outcomes (e.g. 'blocked').

    a[i], b[i] are the outcomes of systems A and B on item i. Under H0 the
    discordant pairs split Binomial(n_d, 1/2), n_d = only_a + only_b;
    p = min(1, 2 * P(X <= min(only_a, only_b))), computed with exact integers.
    The mid-p variant subtracts P(X = min(only_a, only_b)).
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    neither = len(a) - both - only_a - only_b
    nd = only_a + only_b
    if nd == 0:
        p, p_mid = 1.0, 1.0
    else:
        m = min(only_a, only_b)
        denom = 2 ** nd
        tail = sum(math.comb(nd, i) for i in range(m + 1))
        p = min(1.0, 2.0 * tail / denom)
        p_mid = min(1.0, 2.0 * (tail - 0.5 * math.comb(nd, m)) / denom)
    return {
        "n_items": len(a), "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
        "discordant": nd, "p_exact_two_sided": p, "p_mid_two_sided": p_mid,
        "test": "exact McNemar (binomial on discordant pairs, p = 0.5), two-sided",
    }


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

def quantile(xs: Sequence[float], q: float) -> float:
    """Quantile with linear interpolation between order statistics (numpy 'linear', Hyndman-Fan type 7)."""
    if not xs:
        raise ValueError("empty sequence")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def describe(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None, "sum": 0.0}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": float(min(xs)), "p50": quantile(xs, 0.50),
            "p90": quantile(xs, 0.90), "p95": quantile(xs, 0.95), "p99": quantile(xs, 0.99),
            "max": float(max(xs)), "sum": float(sum(xs))}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------
# Bootstrap (percentile), stratified and clustered
# --------------------------------------------------------------------------

def _group(items: Sequence, cluster_key: Callable, strata_key: Optional[Callable]):
    """Clusters in first-appearance order, grouped by stratum (strata sorted by str)."""
    clusters: Dict[Hashable, List[int]] = {}
    order: List[Hashable] = []
    for i, it in enumerate(items):
        c = cluster_key(it)
        if c not in clusters:
            clusters[c] = []
            order.append(c)
        clusters[c].append(i)
    strata: Dict[Hashable, List[Hashable]] = {}
    for c in order:
        members = clusters[c]
        s = strata_key(items[members[0]]) if strata_key else None
        if strata_key and any(strata_key(items[j]) != s for j in members):
            raise ValueError(f"cluster {c!r} spans several strata")
        strata.setdefault(s, []).append(c)
    stratum_order = sorted(strata, key=lambda s: str(s))
    return clusters, [(s, strata[s]) for s in stratum_order]


def cluster_bootstrap_ci(items: Sequence, statistic: Callable[[List], float], cluster_key: Callable,
                         B: int = 10000, seed: int = 0, conf: float = 0.95,
                         strata_key: Optional[Callable] = None, keep_replicates: bool = False) -> Dict[str, object]:
    """Percentile bootstrap that resamples whole clusters with replacement.

    ``cluster_key(item)`` names the cluster (e.g. the seed prompt of a
    multi-turn trajectory, or the request id when latency is measured with
    repeats). With ``strata_key`` the clusters are resampled within each
    stratum (e.g. label), keeping the stratum sizes fixed in clusters.
    """
    if not items:
        raise ValueError("no items")
    clusters, strata = _group(items, cluster_key, strata_key)
    rng = random.Random(seed)
    reps: List[float] = []
    for _ in range(B):
        sample: List = []
        for _, cl in strata:
            for c in rng.choices(cl, k=len(cl)):
                sample.extend(items[j] for j in clusters[c])
        reps.append(statistic(sample))
    alpha = 1.0 - conf
    out = {
        "estimate": statistic(list(items)),
        # B <= 0 disables resampling: no interval (lo = hi = None)
        "lo": quantile(reps, alpha / 2.0) if reps else None, "hi": quantile(reps, 1.0 - alpha / 2.0) if reps else None,
        "conf": conf, "B": B, "seed": seed, "n_items": len(items), "n_clusters": len(clusters),
        "method": "percentile bootstrap, clusters resampled with replacement"
                  + (" within strata" if strata_key else ""),
    }
    if keep_replicates:
        out["replicates"] = reps
    return out


def bootstrap_ci(values: Sequence, statistic: Callable[[List], float] = _mean, B: int = 10000, seed: int = 0,
                 conf: float = 0.95, strata_key: Optional[Callable] = None) -> Dict[str, object]:
    """Percentile bootstrap over items (i.i.d. resampling, optionally within strata)."""
    idx = list(range(len(values)))
    vals = list(values)
    res = cluster_bootstrap_ci(
        idx, lambda sample: statistic([vals[i] for i in sample]), cluster_key=lambda i: i,
        B=B, seed=seed, conf=conf,
        strata_key=(lambda i: strata_key(vals[i])) if strata_key else None,
    )
    res["method"] = "percentile bootstrap, items resampled with replacement" + (" within strata" if strata_key else "")
    res.pop("n_clusters", None)
    return res


def paired_bootstrap_diff(a: Sequence[dict], b: Sequence[dict], metric: Callable[[List[dict]], float],
                          B: int = 10000, seed: int = 0, conf: float = 0.95, key: str = "example_id",
                          strata_key: Optional[Callable] = lambda r: r["label"],
                          cluster_key: Optional[Callable] = None) -> Dict[str, object]:
    """Paired bootstrap of metric(A) - metric(B) for two systems scored on the same items.

    Records are aligned on ``key``; each resample draws the same items (or
    clusters, with ``cluster_key``) for both systems, by default within label
    strata so the harmful/benign counts stay fixed. Reports the observed
    difference, the percentile interval and the share of resampled differences
    <= 0 and >= 0.
    """
    a2, b2 = align_records(a, b, key)
    pairs = [(ra, rb, i) for i, (ra, rb) in enumerate(zip(a2, b2))]

    def stat(sample):
        return metric([p[0] for p in sample]) - metric([p[1] for p in sample])

    ck = (lambda p: cluster_key(p[0])) if cluster_key else (lambda p: p[2])
    res = cluster_bootstrap_ci(pairs, stat, ck, B=B, seed=seed, conf=conf,
                               strata_key=(lambda p: strata_key(p[0])) if strata_key else None,
                               keep_replicates=True)
    reps = res.pop("replicates")
    res["share_le_0"] = sum(1 for d in reps if d <= 0) / len(reps)
    res["share_ge_0"] = sum(1 for d in reps if d >= 0) / len(reps)
    res["metric_a"] = metric(a2)
    res["metric_b"] = metric(b2)
    res["method"] = "paired percentile bootstrap of metric(A) - metric(B), same resampled items for both systems"
    return res


# --------------------------------------------------------------------------
# Metric helpers on per-item record dicts
# --------------------------------------------------------------------------

def _blocked(r: dict, decision_key: str = "decision") -> bool:
    return r[decision_key] == "block"


def abr(records: Sequence[dict], decision_key: str = "decision") -> float:
    h = [r for r in records if r["label"] == "harmful"]
    return sum(_blocked(r, decision_key) for r in h) / len(h) if h else float("nan")


def bar(records: Sequence[dict], decision_key: str = "decision") -> float:
    bn = [r for r in records if r["label"] == "benign"]
    return sum(not _blocked(r, decision_key) for r in bn) / len(bn) if bn else float("nan")


def fpr(records: Sequence[dict], decision_key: str = "decision") -> float:
    return 1.0 - bar(records, decision_key)


def auc(records: Sequence[dict], score_key: str = "score") -> float:
    """Mann-Whitney AUC of harmful vs benign scores (average ranks for ties)."""
    pts = [(float(r[score_key]), 1 if r["label"] == "harmful" else 0) for r in records]
    n1 = sum(y for _, y in pts)
    n0 = len(pts) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    pts.sort(key=lambda t: t[0])
    rank_sum, i = 0.0, 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and pts[j + 1][0] == pts[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum += avg_rank * sum(pts[t][1] for t in range(i, j + 1))
        i = j + 1
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def bootstrap_proportion_limit(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Percentile-bootstrap interval of k/n in the limit B -> infinity (no Monte Carlo).

    Resampling n binary outcomes with replacement gives exactly
    Binomial(n, k/n)/n, so the percentile endpoints are j/n with j the
    smallest count whose CDF reaches alpha/2 (resp. 1 - alpha/2). A seeded
    Monte Carlo run with finite B can land one count away when a CDF value
    sits close to alpha/2 (e.g. 41/55: P(X <= 34) = 0.0255).
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    alpha = 1.0 - conf
    p = k / n

    def inv(q):
        for j in range(n + 1):
            if binom_cdf(j, n, p) >= q - 1e-15:
                return j / n
        return 1.0

    return inv(alpha / 2.0), inv(1.0 - alpha / 2.0)


def proportion_summary(outcomes: Sequence[bool], B: int = 10000, seed: int = 0, conf: float = 0.95) -> Dict[str, object]:
    """k/n with the exact Clopper-Pearson interval and the percentile bootstrap (Monte Carlo and B -> inf limit)."""
    k, n = sum(1 for x in outcomes if x), len(outcomes)
    if n == 0:
        return {"k": 0, "n": 0, "estimate": None, "clopper_pearson": None, "bootstrap_percentile": None}
    lo, hi = clopper_pearson(k, n, conf)
    bs = bootstrap_ci([1.0 if x else 0.0 for x in outcomes], B=B, seed=seed, conf=conf) if B > 0 else None
    return {"k": k, "n": n, "estimate": k / n, "conf": conf,
            "clopper_pearson": [lo, hi],
            "bootstrap_percentile": [bs["lo"], bs["hi"]] if bs else None, "bootstrap_B": B, "bootstrap_seed": seed,
            "bootstrap_percentile_limit": list(bootstrap_proportion_limit(k, n, conf))}


# --------------------------------------------------------------------------
# Reading per-item records from result files
# --------------------------------------------------------------------------

def resolve(payload, dotted: str):
    """Follow a dotted path. A step into a list is an index or ``key=value[&key=value]``
    (the first element whose fields match, compared as strings)."""
    node = payload
    for part in [p for p in (dotted or "").split(".") if p]:
        if isinstance(node, list):
            if "=" in part:
                conds = [c.split("=", 1) for c in part.split("&")]
                hits = [x for x in node if isinstance(x, dict) and all(str(x.get(k)) == v for k, v in conds)]
                if not hits:
                    raise KeyError(f"no list element matches {part!r}")
                node = hits[0]
            else:
                node = node[int(part)]
        else:
            node = node[part]
    return node


def example_records(records: Sequence[dict], decision_field: str = "auto", score_field: Optional[str] = None,
                    cluster_field: Optional[str] = None) -> List[dict]:
    """One record per example: ``{example_id, label, decision, score, cluster, n_records}``.

    ``decision_field='auto'`` reads ``controller_decision`` (end-to-end files) or
    ``decision``. Per-turn records ``<id>:turn-<k>`` are collapsed: blocked if any
    turn is blocked, score = maximum over turns.
    """
    out: Dict[str, dict] = {}
    order: List[str] = []
    for row in records:
        field = decision_field
        if field == "auto":
            field = "controller_decision" if "controller_decision" in row else "decision"
        dec = row[field]
        if dec not in ("block", "allow"):
            raise ValueError(f"unexpected decision {dec!r} in field {field!r}")
        eid = str(row["example_id"]).split(":turn-")[0]
        label = getattr(row["label"], "value", row["label"])
        sc = row.get(score_field) if score_field else None
        if eid not in out:
            out[eid] = {"example_id": eid, "label": label, "decision": dec, "score": sc, "n_records": 1,
                        "cluster": row.get(cluster_field) if cluster_field else eid}
            order.append(eid)
            continue
        r = out[eid]
        if r["label"] != label:
            raise ValueError(f"example {eid} has records with different labels")
        if dec == "block":
            r["decision"] = "block"
        if sc is not None:
            r["score"] = sc if r["score"] is None else max(r["score"], sc)
        r["n_records"] += 1
    return [out[e] for e in order]


def _load(path: str, records_key: str, decision_field: str, score_field: Optional[str],
          cluster_field: Optional[str]) -> List[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = resolve(payload, records_key)
    if not isinstance(recs, list) or not recs:
        raise ValueError(f"no per-item records at '{records_key}' in {path}")
    return example_records(recs, decision_field, score_field, cluster_field)


def _rates(recs: Sequence[dict]) -> Dict[str, object]:
    h = [r for r in recs if r["label"] == "harmful"]
    b = [r for r in recs if r["label"] == "benign"]
    out = {}
    for name, items, positive in (("abr", h, "block"), ("bar", b, "allow")):
        k, n = sum(r["decision"] == positive for r in items), len(items)
        out[name] = {"k": k, "n": n, "estimate": k / n if n else None,
                     "clopper_pearson95": list(clopper_pearson(k, n)) if n else None}
    if h and b and all(r.get("score") is not None for r in recs):
        out["auc"] = auc(recs, "score")
    return out


def compare_systems(a: Sequence[dict], b: Sequence[dict], B: int = 10000, seed: int = 0,
                    clustered: bool = False) -> Dict[str, object]:
    """System A vs B on the same items (records from ``example_records``).

    Exact McNemar on harmful items (blocked) and on benign items (allowed), and
    paired bootstrap of ABR, BAR and (when both have scores) AUC differences,
    resampling items within label strata, or whole clusters with ``clustered``.
    """
    a2, b2 = align_records(a, b)
    ck = (lambda r: r["cluster"]) if clustered else None
    out: Dict[str, object] = {"n_items": len(a2), "a": _rates(a2), "b": _rates(b2)}
    h_idx = [i for i, r in enumerate(a2) if r["label"] == "harmful"]
    b_idx = [i for i, r in enumerate(a2) if r["label"] == "benign"]
    out["mcnemar_harmful_blocked"] = mcnemar_exact([a2[i]["decision"] == "block" for i in h_idx],
                                                   [b2[i]["decision"] == "block" for i in h_idx])
    out["mcnemar_benign_allowed"] = mcnemar_exact([a2[i]["decision"] == "allow" for i in b_idx],
                                                  [b2[i]["decision"] == "allow" for i in b_idx])
    metrics = {"abr": abr, "bar": bar}
    if "auc" in out["a"] and "auc" in out["b"]:
        metrics["auc"] = lambda recs: auc(recs, "score")
    strata_used = True
    for k, (name, fn) in enumerate(sorted(metrics.items())):
        kw = dict(B=B, seed=seed + k, cluster_key=ck)
        try:
            res = paired_bootstrap_diff(a2, b2, fn, strata_key=(lambda r: r["label"]) if strata_used else None, **kw)
        except ValueError as exc:  # a cluster mixes labels: resample clusters without strata
            if "spans several strata" not in str(exc):
                raise
            strata_used = False
            res = paired_bootstrap_diff(a2, b2, fn, strata_key=None, **kw)
        out[f"paired_bootstrap_{name}_diff"] = res
    out["note"] = "difference = A - B; " + ("whole clusters" if clustered else "items") + " resampled" + (
        " within label strata" if strata_used else " without strata (clusters mix labels)")
    return out


# --------------------------------------------------------------------------

def _cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description="R1 statistics: exact intervals, McNemar, paired/clustered bootstrap.")
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    p = sub.add_parser("cp", help="Clopper-Pearson interval for k/n")
    p.add_argument("k", type=int)
    p.add_argument("n", type=int)
    p.add_argument("--conf", type=float, default=0.95)
    p = sub.add_parser("mcnemar", help="exact McNemar p from discordant counts")
    p.add_argument("only_a", type=int)
    p.add_argument("only_b", type=int)
    p = sub.add_parser("boot", help="percentile bootstrap interval for k/n")
    p.add_argument("k", type=int)
    p.add_argument("n", type=int)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("rate", help="ABR/BAR of one result file: k/n, Clopper-Pearson, item or cluster bootstrap")
    p.add_argument("--input", required=True)
    p.add_argument("--records", default="records", help="dotted path to the record list")
    p.add_argument("--decision-field", default="auto")
    p.add_argument("--cluster-field", default=None, help="record field naming the cluster (e.g. the seed prompt)")
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--output", default=None)
    p = sub.add_parser("compare", help="paired comparison of two systems on the same items")
    for s in ("a", "b"):
        p.add_argument(f"--{s}", required=True, help=f"result file of system {s.upper()}")
        p.add_argument(f"--{s}-records", default="records")
        p.add_argument(f"--{s}-decision-field", default="auto")
        p.add_argument(f"--{s}-score", default=None, help="per-item score field (for the AUC difference)")
    p.add_argument("--cluster-field", default=None, help="field of A's records naming the cluster")
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "cp":
        lo, hi = clopper_pearson(args.k, args.n, args.conf)
        print(f"{args.k}/{args.n} = {args.k / args.n:.4f}; Clopper-Pearson {args.conf:.0%}: [{lo:.6f}, {hi:.6f}]")
        return
    if args.cmd == "mcnemar":
        a = [True] * args.only_a + [False] * args.only_b
        b = [False] * args.only_a + [True] * args.only_b
        print(json.dumps(mcnemar_exact(a, b), indent=1))
        return
    if args.cmd == "boot":
        r = proportion_summary([True] * args.k + [False] * (args.n - args.k), B=args.B, seed=args.seed)
        print(json.dumps(r, indent=1))
        return
    if args.cmd == "rate":
        recs = _load(args.input, args.records, args.decision_field, None, args.cluster_field)
        res = {"input": args.input, "records": args.records, "n_examples": len(recs), "rates": _rates(recs)}
        for name, label, pos in (("abr", "harmful", "block"), ("bar", "benign", "allow")):
            items = [r for r in recs if r["label"] == label]
            if not items:
                continue
            stat = (lambda s, p=pos: sum(r["decision"] == p for r in s) / len(s))
            res[f"{name}_bootstrap"] = cluster_bootstrap_ci(
                items, stat, cluster_key=(lambda r: r["cluster"]) if args.cluster_field else (lambda r: r["example_id"]),
                B=args.B, seed=args.seed)
            res[f"{name}_bootstrap"]["clustered_by"] = args.cluster_field
        for name in ("abr", "bar"):
            r = res["rates"][name]
            bs = res.get(f"{name}_bootstrap")
            if r["n"]:
                print(f"{name.upper()} {r['k']}/{r['n']} = {r['estimate']:.4f}  CP [{r['clopper_pearson95'][0]:.4f}, "
                      f"{r['clopper_pearson95'][1]:.4f}]  bootstrap [{bs['lo']:.4f}, {bs['hi']:.4f}] "
                      f"({bs['n_clusters']} clusters)")
    else:
        a = _load(args.a, args.a_records, args.a_decision_field, args.a_score, args.cluster_field)
        b = _load(args.b, args.b_records, args.b_decision_field, args.b_score, None)
        res = {"a": {"file": args.a, "records": args.a_records, "score": args.a_score},
               "b": {"file": args.b, "records": args.b_records, "score": args.b_score},
               "comparison": compare_systems(a, b, B=args.B, seed=args.seed, clustered=bool(args.cluster_field))}
        c = res["comparison"]
        for s in ("a", "b"):
            r = c[s]
            print(f"{s.upper()}: ABR {r['abr']['k']}/{r['abr']['n']}  BAR {r['bar']['k']}/{r['bar']['n']}"
                  + (f"  AUC {r['auc']:.4f}" if "auc" in r else ""))
        for key in ("mcnemar_harmful_blocked", "mcnemar_benign_allowed"):
            m = c[key]
            print(f"{key}: only A {m['only_a']}, only B {m['only_b']}, exact p = {m['p_exact_two_sided']:.4g}")
        for key in sorted(k for k in c if k.startswith("paired_bootstrap_")):
            d = c[key]
            print(f"{key}: {d['estimate']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    _cli()
