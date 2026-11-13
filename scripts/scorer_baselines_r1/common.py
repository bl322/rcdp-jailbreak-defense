"""Helpers shared by the R1 scorer-only baselines and scripts/measure_latency_r1.py.

* hardware / software description of the measuring host (``environment``),
* code fingerprint (``code_version``),
* scorer construction with the same classes as the RCDP runs (``build_scorer``),
* cache clearing so no measurement or baseline call is served from a scorer cache,
* data checks (partition counts, calibration/held-out disjointness),
* a stateless threshold router that plugs into
  ``rcdp_guard.calibration_protocol.search_threshold`` so scorer-only
  baselines use exactly RCDP's grid, objective J = BlockRate - alpha * FPR and
  tie-break on the same D_thresh subset.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rcdp_guard.calibration_protocol import DEFAULT_THRESHOLDS  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from stats_r1 import auc as auc_fn  # noqa: E402
from stats_r1 import proportion_summary  # noqa: E402

DEBERTA_ZS = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DISTILBART = "valhalla/distilbart-mnli-12-3"
SCORER_KINDS = ("keyword", "distilbart-mnli", "deberta-zeroshot", "llama-guard")


# --------------------------------------------------------------------------
# Host description
# --------------------------------------------------------------------------

def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=10)
            if out.stdout.strip():
                return out.stdout.strip()
    except Exception:  # noqa: BLE001 - descriptive only
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"


def _nvidia_smi() -> object:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,clocks.max.sm",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return f"nvidia-smi exit code {out.returncode}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def _version(mod: str):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def environment() -> dict:
    """Everything needed to state on which host a timing was taken (no host name)."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "numpy": _version("numpy"),
        "transformers": _version("transformers"),
        "tokenizers": _version("tokenizers"),
        "torch": _version("torch"),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")},
    }
    try:
        import torch  # noqa: WPS433

        env["torch_num_threads"] = torch.get_num_threads()
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = torch.backends.cudnn.version()
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["nvidia_smi"] = _nvidia_smi()
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
    return env


def model_info(obj) -> dict | None:
    """dtype / device / attention implementation of a scorer's (or a raw) HF model."""
    model = getattr(obj, "_model", obj)
    if model is None or not hasattr(model, "parameters"):
        return None
    try:
        p = next(model.parameters())
        cfg = getattr(model, "config", None)
        return {
            "class": type(model).__name__,
            "dtype": str(p.dtype),
            "device": str(p.device),
            "n_parameters": int(sum(t.numel() for t in model.parameters())),
            "attn_implementation": getattr(cfg, "_attn_implementation", None) if cfg is not None else None,
            "name_or_path_basename": os.path.basename(str(getattr(cfg, "_name_or_path", "")).rstrip("/")) or None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def code_version(extra: list | None = None) -> dict:
    h = hashlib.sha256()
    files = sorted((ROOT / "rcdp_guard").glob("*.py")) + [Path(f).resolve() for f in (extra or [])]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return {"sha256": h.hexdigest(), "files": [f.name for f in files]}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def build_scorer(kind: str, model_name: str | None = None, device: str = "cpu", local_files_only: bool = False,
                 dtype: str | None = None, cache_path: str | None = None):
    """Same scorer classes as the RCDP runs. ``deberta-zeroshot`` is the generic NLI
    scorer class (DistilBartMNLIRiskScorer) loaded with the DeBERTa-v3-large zero-shot checkpoint."""
    from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer

    if kind == "keyword":
        return KeywordRiskScorer()
    if kind in ("distilbart-mnli", "deberta-zeroshot"):
        name = model_name or (DISTILBART if kind == "distilbart-mnli" else DEBERTA_ZS)
        scorer = DistilBartMNLIRiskScorer(model_name=name, device=device, local_files_only=local_files_only,
                                          cache_path=cache_path)
        if dtype:
            import torch

            scorer._model.to(getattr(torch, dtype))
        return scorer
    if kind == "llama-guard":
        if not model_name:
            raise ValueError("--scorer llama-guard needs a model path/name")
        return LlamaGuardRiskScorer(model_name=model_name, device=device, local_files_only=local_files_only,
                                    torch_dtype=dtype, cache_path=cache_path)
    raise ValueError(f"unknown scorer kind {kind!r}; expected one of {SCORER_KINDS}")


_WRAPPED_ATTRS = ("_scorer", "_inner", "inner", "nli", "guard", "scorer", "classifier", "base")


def clear_scorer_cache(scorer, _seen=None) -> None:
    """Empty every in-memory score cache reachable from ``scorer`` (incl. wrappers).

    PersistentScoreCache keeps entries in memory even without a cache file, so
    repeated or duplicated texts would otherwise be served without a forward
    pass (the cause of the invalid concurrency rows of the submitted GPU file).
    Follows scorer wrappers, judges (``guard``) and prompt classifiers (``scorer``).
    """
    seen = set() if _seen is None else _seen
    if scorer is None or id(scorer) in seen:
        return
    seen.add(id(scorer))
    d = getattr(scorer, "__dict__", {})
    if "_memo" in d:
        d["_memo"].clear()
    cache = d.get("_cache")
    if cache is not None and hasattr(cache, "_entries"):
        cache._entries.clear()
    for attr in _WRAPPED_ATTRS:
        if attr in d:
            clear_scorer_cache(d[attr], seen)


def user_turn_scores(scorer, example) -> list:
    """Score every user turn with the controller's history convention (all prior turns)."""
    history: list = []
    out = []
    for idx, turn in enumerate(example.turns):
        if turn.role != "user":
            history.append(turn.text)
            continue
        res = scorer.score(turn.text, list(history), example.metadata)
        out.append({"turn": idx, "raw_score": float(res.raw_score), "reasons": list(res.reasons),
                    "metadata": dict(res.metadata)})
        history.append(turn.text)
    return out


# --------------------------------------------------------------------------
# Data checks
# --------------------------------------------------------------------------

def label_counts(examples) -> tuple:
    h = sum(1 for e in examples if e.label.value == "harmful")
    return h, len(examples) - h


def load_partition(path: str, expect: str | None):
    exs = load_examples(path)
    counts = label_counts(exs)
    if expect:
        want = tuple(int(x) for x in expect.split(","))
        if counts != want:
            raise SystemExit(f"{path}: {counts[0]} harmful / {counts[1]} benign, expected {want[0]}/{want[1]} "
                             "(pass --expect-counts '' to skip this check)")
    return exs


def ensure_disjoint(cal, held) -> None:
    overlap = {e.example_id for e in cal} & {e.example_id for e in held}
    if overlap:
        raise SystemExit(f"calibration and evaluation sets share {len(overlap)} example IDs, e.g. {sorted(overlap)[:3]}")


# --------------------------------------------------------------------------
# Threshold router for search_threshold
# --------------------------------------------------------------------------

class ScorerThresholdRouter:
    """Stateless scorer-only rule with the controller interface used by search_threshold.

    ``turn_probs(example)`` returns one dict per user turn with key ``prob``.
    A turn is blocked iff prob >= config.thresholds['tau_q'] (num_states = 4,
    so search_threshold's block key is tau_q and thresholds_for_block sets
    tau_q = tau exactly); an example is blocked if any user turn is.
    """

    def __init__(self, turn_probs):
        self.config = SimpleNamespace(num_states=4, thresholds=dict(DEFAULT_THRESHOLDS), benchmark_thresholds=None)
        self._turn_probs = turn_probs

    def evaluate_example(self, example):
        tau = self.config.thresholds["tau_q"]
        return [SimpleNamespace(decision="block" if t["prob"] >= tau else "allow", prob=t["prob"])
                for t in self._turn_probs(example)]


# --------------------------------------------------------------------------
# Summaries and output
# --------------------------------------------------------------------------

def evenly_spaced(items: list, k: int) -> list:
    """k items spread over the whole list (index floor(i * n / k)); all items when k <= 0 or k >= n.

    Used to time expensive alternative paths on a subset that still covers both
    labels and both benchmarks of a label-sorted dataset file.
    """
    n = len(items)
    if k <= 0 or k >= n:
        return list(items)
    return [items[(i * n) // k] for i in range(k)]


def binary_summary(records: list, decision_key: str = "decision", score_key: str | None = None,
                   B: int = 10000, seed: int = 20261113) -> dict:
    harm = [r for r in records if r["label"] == "harmful"]
    ben = [r for r in records if r["label"] == "benign"]
    out = {
        "attack_block_rate": proportion_summary([r[decision_key] == "block" for r in harm], B=B, seed=seed),
        "benign_allow_rate": proportion_summary([r[decision_key] == "allow" for r in ben], B=B, seed=seed + 1),
        "false_positive_rate": proportion_summary([r[decision_key] == "block" for r in ben], B=B, seed=seed + 2),
    }
    if score_key and harm and ben and all(r.get(score_key) is not None for r in records):
        out["auc"] = auc_fn(records, score_key)
        out["auc_score"] = score_key
    return out


def fmt_summary(name: str, s: dict) -> str:
    a, b = s["attack_block_rate"], s["benign_allow_rate"]

    def one(x):
        if not x["n"]:
            return "n/a"
        lo, hi = x["clopper_pearson"]
        return f"{x['k']}/{x['n']} = {x['estimate']:.4f} [CP {lo:.4f}, {hi:.4f}]"

    extra = f"  AUC {s['auc']:.4f}" if "auc" in s else ""
    return f"{name}: ABR {one(a)}; BAR {one(b)}{extra}"


def write_json(path: str, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p
