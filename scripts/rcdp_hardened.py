"""Hardened RCDP service used for the adaptive re-attack.

Design constraint: the RCDP method itself is untouched. Risk estimation,
calibration, the four-state sequential controller and the state-conditioned
prompt budget are exactly as in ``rcdp_guard``. Two *orthogonal* guards sit
outside the calibrated pipeline:

  G1  Principal binding (``HardenedService(bind=True)``).
      The controller sees all queries of one caller principal (account / API
      key / session token) as one dialogue of user turns.

      Service latch. Independently of the controller, the service can refuse
      every later query of a principal once one of its queries reached a state
      in ``latch_states``. The submitted service latched on
      {HARDENED, QUARANTINE} (``DEFAULT_LATCH_STATES``). Note that HARDENED is
      an *allowed* state of the controller, so this latch is stricter than the
      controller; and QUARANTINE is already absorbing inside the controller
      (``allow_state_downgrade=False``), so a QUARANTINE-only latch changes
      nothing except that the controller is no longer called. ``latch_states=()``
      disables the service latch; the controller's own monotone escalation
      still applies.

  G2  Escalate-only second risk view (``--scorer ensemble``).
      max(NLI, Llama-Guard-3-1B) raw score; the calibrator is re-fitted on the
      combined scores.

Calibration cache. Fitting is the paper's protocol by default (``r1``: three-way
split, L2 head, Platt layer, threshold search; see
``rcdp_guard/calibration_protocol.py``). The cache file stores the calibrator
*together with* the fitted operating point (``thresholds``,
``benchmark_thresholds``, ``legacy_first_turn_trend``) and the calibration
report, and its key covers the scorer, the full pre-fit controller config, the
calibration protocol, split seed, alpha and the SHA-256 of the calibration
file. A cached load is therefore identical to a fresh fit; any other pickle
(e.g. the submitted ``calibrator_<kind>_ba<bool>.pkl``, which held the
calibrator weights only) is ignored.

Run ``python scripts/rcdp_hardened.py --scorer nli`` to build (or verify) the
cache and print the operating point.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
# The code tree that holds this script wins over $RCDP_EXPERIMENTS, so a stale
# rcdp_guard/ inside the experiments directory cannot shadow the revised one.
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", CODE_ROOT))
if not (CODE_ROOT / "rcdp_guard").is_dir() and str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

CALIB_CACHE_DIR = EXPERIMENTS / "cache"

import rcdp_guard  # noqa: E402
from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.calibration_protocol import DEFAULT_ALPHA, DEFAULT_SEED, PROTOCOLS  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import calibration_protocol_from_env, fit_calibrator  # noqa: E402
from rcdp_guard.scorers import (  # noqa: E402
    DistilBartMNLIRiskScorer,
    KeywordRiskScorer,
    LlamaGuardRiskScorer,
)
from rcdp_guard.types import Example, Turn  # noqa: E402

DATA = EXPERIMENTS / "data"
CALIBRATION = DATA / "paper_splits" / "full_pool_calibration.jsonl"
LOCAL_LG = str(EXPERIMENTS / "llama-guard-3-1b")
NLI_MODEL = "valhalla/distilbart-mnli-12-3"
SCORER_KINDS = ("nli", "ensemble", "keyword")

CALIB_CACHE_FORMAT = "rcdp-hardened-calibration/v2"

DEFAULT_LATCH_STATES = ("hardened", "quarantine")
LATCH_PRESETS = {
    "none": (),
    "quarantine": ("quarantine",),
    "hardened,quarantine": ("hardened", "quarantine"),
}
_STATE_RANK = {"normal": 0, "cautious": 1, "hardened": 2, "quarantine": 3}


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
class EnsembleRiskScorer:
    """max(nli, llamaguard) -- escalate-only combination of two risk views."""

    def __init__(self, nli, guard) -> None:
        self.nli = nli
        self.guard = guard

    def score(self, text: str, history=None, metadata=None):
        n = self.nli.score(text, history, metadata)
        g = self.guard.score(text, history, metadata)
        raw = max(n.raw_score, g.raw_score)
        return type(n)(
            raw_score=raw,
            reasons=[f"nli={n.raw_score:.4f}", f"guard={g.raw_score:.4f}", *n.reasons],
            metadata={
                "view_nli": n.raw_score,
                "view_guard": g.raw_score,
                "aggregation": "max-escalate-only",
            },
        )


def scorer_identity(kind: str, *, device: str = "cpu", nli_model: str = NLI_MODEL,
                    guard_model: str = None) -> dict:
    """What determines the raw scores (part of the calibration cache key)."""
    if kind == "keyword":
        return {"kind": "keyword", "class": "KeywordRiskScorer"}
    out = {"kind": kind, "nli_model": nli_model, "device": device}
    if kind == "ensemble":
        out["guard_model"] = guard_model or LOCAL_LG
        out["aggregation"] = "max-escalate-only"
    return out


def build_scorer(kind: str, *, device: str = "cpu", nli_model: str = NLI_MODEL,
                 guard_model: str = None, torch_threads: int = 16,
                 local_files_only: bool = True):
    if kind == "keyword":
        # Offline smoke tests only: lexical scorer, no torch.
        return KeywordRiskScorer()
    if kind not in ("nli", "ensemble"):
        raise ValueError(f"unknown scorer kind: {kind}")
    import torch

    torch.set_num_threads(torch_threads)
    nli = DistilBartMNLIRiskScorer(
        model_name=nli_model, local_files_only=local_files_only, device=device
    )
    if kind == "nli":
        return nli
    guard = LlamaGuardRiskScorer(
        model_name=guard_model or LOCAL_LG,
        device=device,
        local_files_only=local_files_only,
        cache_path=str(EXPERIMENTS / "cache" / "llamaguard_tap_probe_cache.json"),
    )
    return EnsembleRiskScorer(nli, guard)


# ---------------------------------------------------------------------------
# Calibration cache
# ---------------------------------------------------------------------------
def jsonable(obj):
    """Deterministic JSON-friendly copy (Enum -> value, set -> sorted list)."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(jsonable(v) for v in obj)
    if isinstance(obj, float) or isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return repr(obj)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_calibration(protocol=None, seed=None, alpha=None) -> tuple:
    protocol = (protocol or calibration_protocol_from_env()).strip().lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown calibration protocol {protocol!r}; expected one of {PROTOCOLS}")
    seed = str(seed or os.environ.get("RCDP_SPLIT_SEED", DEFAULT_SEED))
    alpha = DEFAULT_ALPHA if alpha is None else float(alpha)
    return protocol, seed, alpha


def calibration_cache_key(*, scorer_id: dict, config: ControllerConfig, feature_config: FeatureConfig,
                          protocol: str, seed: str, alpha: float, calibration_path) -> dict:
    return {
        "format": CALIB_CACHE_FORMAT,
        "scorer": jsonable(scorer_id),
        "controller_config_prefit": jsonable(asdict(config)),
        "feature_groups": sorted(feature_config.enabled_groups),
        "protocol": protocol,
        "seed": seed if protocol != "legacy" else None,
        "alpha": alpha if protocol != "legacy" else None,
        "calibration_file_sha256": file_sha256(calibration_path),
        "calibration_file_name": Path(calibration_path).name,
    }


def key_digest(key: dict) -> str:
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_file_for(cache_dir, kind: str, benchmark_aware: bool, key: dict) -> Path:
    return Path(cache_dir) / (
        f"calibration_{kind}_ba{benchmark_aware}_{key['protocol']}_{key_digest(key)[:16]}.pkl"
    )


def _postfit_state(controller) -> dict:
    cfg = controller.config
    return {
        "thresholds": dict(cfg.thresholds),
        "benchmark_thresholds": copy.deepcopy(cfg.benchmark_thresholds),
        "legacy_first_turn_trend": bool(cfg.legacy_first_turn_trend),
    }


def save_calibration(path, controller, key: dict) -> None:
    payload = {
        "format": CALIB_CACHE_FORMAT,
        "key": key,
        "key_digest": key_digest(key),
        "calibrator": controller.calibrator,
        **_postfit_state(controller),
        "postfit_config": jsonable(asdict(controller.config)),
        "calibration_report": getattr(controller, "calibration_report", None),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temporary name: several TAP arms may build the same cache at once.
    tmp = path.with_suffix(path.suffix + ".%d.tmp" % os.getpid())
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh)
    os.replace(str(tmp), str(path))


def restore_calibration(path, controller, key: dict) -> tuple:
    """Restore a cached fit into ``controller``. Returns (ok, reason)."""
    path = Path(path)
    if not path.exists():
        return False, "no cache file"
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001 - any unreadable pickle means refit
        return False, f"unreadable cache ({exc.__class__.__name__})"
    if not isinstance(payload, dict) or payload.get("format") != CALIB_CACHE_FORMAT:
        return False, "stale cache format (calibrator-only pickle?)"
    if payload.get("key") != key:
        return False, "cache key mismatch"
    cfg = controller.config
    controller.calibrator = payload["calibrator"]
    cfg.thresholds = dict(payload["thresholds"])
    cfg.benchmark_thresholds = copy.deepcopy(payload["benchmark_thresholds"])
    cfg.legacy_first_turn_trend = bool(payload["legacy_first_turn_trend"])
    if jsonable(asdict(cfg)) != payload.get("postfit_config"):
        return False, "restored config differs from the fitted config"
    controller.calibration_report = payload.get("calibration_report")
    return True, "loaded"


def build_controller(
    kind: str,
    benchmark_aware: bool = False,
    *,
    protocol: str = None,
    seed: str = None,
    alpha: float = None,
    calibration_path=None,
    cache_dir=None,
    use_cache: bool = True,
    scorer=None,
    device: str = "cpu",
    nli_model: str = NLI_MODEL,
    guard_model: str = None,
    config: ControllerConfig = None,
    log=print,
) -> RCDPController:
    """Controller with the fitted calibration of ``protocol`` (default: env or r1).

    ``scorer`` may be passed to reuse an already loaded scorer; ``kind`` must
    still describe it (it is part of the cache key).
    """
    protocol, seed, alpha = resolve_calibration(protocol, seed, alpha)
    calibration_path = Path(calibration_path or CALIBRATION)
    if scorer is None:
        scorer = build_scorer(kind, device=device, nli_model=nli_model, guard_model=guard_model)
    if config is None:
        config = ControllerConfig(benchmark_aware=benchmark_aware)
    if config.use_learned_transition_policy:
        # The cache holds the calibration stages only, not a fitted trajectory policy.
        raise ValueError("build_controller does not fit/cache a learned transition policy")
    prefit = copy.deepcopy(config)
    controller = RCDPController(scorer=scorer, feature_config=FeatureConfig.all(), config=config)

    key = calibration_cache_key(
        scorer_id=scorer_identity(kind, device=device, nli_model=nli_model, guard_model=guard_model),
        config=config,
        feature_config=controller.feature_config,
        protocol=protocol,
        seed=seed,
        alpha=alpha,
        calibration_path=calibration_path,
    )
    cache_path = cache_file_for(cache_dir or CALIB_CACHE_DIR, kind, config.benchmark_aware, key)
    loaded, reason = (False, "cache disabled")
    if use_cache:
        loaded, reason = restore_calibration(cache_path, controller, key)
    if not loaded:
        if use_cache and reason != "no cache file" and log:
            log(f"[~] calibration cache {cache_path.name}: {reason}; refitting")
        # A failed restore may have touched the config: start from the pre-fit copy.
        controller.config = copy.deepcopy(prefit)
        fit_calibrator(controller, load_examples(str(calibration_path)), protocol=protocol, seed=seed, alpha=alpha)
        if use_cache:
            save_calibration(cache_path, controller, key)
    controller.calibration_provenance = {
        "protocol": protocol,
        "seed": seed,
        "alpha": alpha,
        "calibration_path": str(calibration_path),
        "calibration_file_sha256": key["calibration_file_sha256"],
        "cache_path": str(cache_path) if use_cache else None,
        "loaded_from_cache": loaded,
        "cache_key_digest": key_digest(key),
        "scorer": key["scorer"],
    }
    return controller


def calibration_summary(controller) -> dict:
    """Compact, JSON-serialisable description of the fitted operating point."""
    report = getattr(controller, "calibration_report", None) or {}
    search = report.get("threshold_search") or {}
    selected = search.get("selected") or {}
    split = report.get("split") or {}
    prov = getattr(controller, "calibration_provenance", None) or {}
    return {
        **prov,
        # Controllers fitted directly with fit_calibrator have no provenance block.
        "protocol": prov.get("protocol") or report.get("protocol"),
        "seed": prov.get("seed", report.get("seed")),
        "report_protocol": report.get("protocol"),
        "thresholds": dict(controller.config.thresholds),
        "benchmark_thresholds": controller.config.benchmark_thresholds,
        "legacy_first_turn_trend": controller.config.legacy_first_turn_trend,
        "head": report.get("head"),
        "platt": {k: v for k, v in (report.get("platt") or {}).items()} or None,
        "threshold_selected": {k: selected.get(k) for k in ("tau", "block_rate", "fpr", "J")} if selected else None,
        "split_sizes": split.get("sizes"),
        "controller_config": jsonable(asdict(controller.config)),
    }


def code_provenance(extra_files=()) -> dict:
    files = [
        HERE / "rcdp_hardened.py",
        CODE_ROOT / "rcdp_guard" / "controller.py",
        CODE_ROOT / "rcdp_guard" / "calibration_protocol.py",
        CODE_ROOT / "rcdp_guard" / "calibration.py",
        CODE_ROOT / "rcdp_guard" / "scorers.py",
    ] + [Path(p) for p in extra_files]
    return {
        "python": platform.python_version(),
        "rcdp_guard_path": str(Path(rcdp_guard.__file__).resolve().parent),
        "sha256": {p.name: file_sha256(p) for p in files if p.exists()},
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
def parse_latch_states(spec) -> tuple:
    """'none' | 'quarantine' | 'hardened,quarantine' | iterable of state names."""
    if spec is None:
        return tuple(DEFAULT_LATCH_STATES)
    if isinstance(spec, str):
        spec = spec.strip().lower()
        if spec in ("", "none", "off"):
            return ()
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = [str(s).strip().lower() for s in spec]
    bad = [n for n in names if n not in _STATE_RANK]
    if bad:
        raise ValueError(f"unknown latch state(s) {bad}; expected names from {sorted(_STATE_RANK)}")
    return tuple(sorted(set(names), key=lambda n: _STATE_RANK[n]))


@dataclass
class Principal:
    """Sequential state of one caller.  RCDP sees this as one dialogue."""

    texts: list = field(default_factory=list)
    latched: bool = False
    latched_by: str = None
    latched_at_turn: int = None


class HardenedService:
    """Black-box endpoint.  The attacker observes block/allow only."""

    def __init__(self, controller: RCDPController, example: Example, bind: bool = True,
                 latch_states=DEFAULT_LATCH_STATES):
        self.controller = controller
        self.example = example
        self.bind = bind
        self.latch_states = parse_latch_states(latch_states)
        self.principals = {}
        self.queries = 0

    def _example_for(self, texts) -> Example:
        return Example(
            example_id=self.example.example_id,
            benchmark=self.example.benchmark,
            label=self.example.label,
            turns=[Turn(role="user", text=t) for t in texts],
            metadata=dict(self.example.metadata),
        )

    def _last_record(self, texts):
        records = self.controller.evaluate_example(self._example_for(texts))
        return records[-1] if records else None

    @staticmethod
    def _record_fields(rec, prefix: str = "") -> dict:
        if rec is None:
            return {prefix + "state": None, prefix + "decision": None, prefix + "calibrated": None,
                    prefix + "raw_score": None, prefix + "effective_risk": None}
        return {
            prefix + "state": rec.state,
            prefix + "decision": rec.decision,
            prefix + "calibrated": rec.calibrated_score,
            prefix + "raw_score": rec.raw_score,
            prefix + "effective_risk": rec.metadata.get("effective_risk"),
        }

    def stateless_view(self, text: str) -> dict:
        """Decision on ``text`` as the first query of a fresh principal (diagnostic)."""
        return self._record_fields(self._last_record([text]), prefix="stateless_")

    def query(self, text: str, principal_id: str = "p0", shadow: bool = False) -> dict:
        self.queries += 1
        if not self.bind:
            self.principals[principal_id] = Principal()
        p = self.principals.setdefault(principal_id, Principal())
        latched_before = p.latched
        if p.latched:
            p.texts.append(text)
            out = {
                "allowed": False,
                "decision": "block",
                "state": None,
                "reason": "latched",
                "controller_called": False,
                "calibrated": None,
                "raw_score": None,
                "effective_risk": None,
                "latch_trigger_state": p.latched_by,
            }
            if shadow:
                # What the controller itself would decide on the full session.
                out.update(self._record_fields(self._last_record(p.texts), prefix="shadow_"))
        else:
            p.texts.append(text)
            last = self._last_record(p.texts)
            if last is None:
                out = {"allowed": False, "decision": "block", "state": "none", "reason": "no_records",
                       "controller_called": True, "calibrated": None, "raw_score": None,
                       "effective_risk": None, "latch_trigger_state": None}
            else:
                out = {"allowed": last.decision == "allow", "reason": "controller",
                       "controller_called": True, "latch_trigger_state": None,
                       **self._record_fields(last)}
                if last.state in self.latch_states:
                    p.latched = True
                    p.latched_by = last.state
                    p.latched_at_turn = len(p.texts)
        out.update({
            "principal_id": principal_id,
            "session_turn": len(p.texts),
            "latched_before": latched_before,
            "latched_after": p.latched,
            "latch_states": list(self.latch_states),
        })
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build or verify the cached calibration of the hardened service.")
    ap.add_argument("--scorer", choices=SCORER_KINDS, default="nli")
    ap.add_argument("--benchmark-aware", action="store_true")
    ap.add_argument("--calibration-protocol", default=None, help="r1 (default), r1-no-platt, raw-threshold, legacy")
    ap.add_argument("--split-seed", default=None)
    ap.add_argument("--calibration-dataset", default=str(CALIBRATION))
    ap.add_argument("--cache-dir", default=str(CALIB_CACHE_DIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    controller = build_controller(
        args.scorer, args.benchmark_aware, protocol=args.calibration_protocol, seed=args.split_seed,
        calibration_path=args.calibration_dataset, cache_dir=args.cache_dir,
        use_cache=not args.no_cache, device=args.device,
    )
    summary = calibration_summary(controller)
    summary.pop("controller_config", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
