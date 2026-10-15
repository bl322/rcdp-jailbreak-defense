"""Hardened RCDP service used for the adaptive re-attack.

Design constraint: the RCDP method itself is untouched.  Risk estimation,
Platt calibration, the four-state sequential controller and the state-
conditioned prompt budget are exactly as published.  Two *orthogonal* guards
are added, both of which sit outside the calibrated pipeline:

  G1  Principal binding.
      RCDP is a sequential controller, so the unit of the sequence has to be
      stated.  Published threat model: a service attacker who issues queries.
      If each query is treated as an independent sequence, the trajectory
      memory, the monotone state escalation and the hazard latch are all
      inert by construction -- that is not a weakness of the controller, it is
      an evaluation that never switches it on.  G1 binds the sequence to the
      caller principal (account / API key / session token), which is the unit
      real deployments rate-limit and ban.  Nothing in the controller changes.

  G2  Escalate-only second risk view.
      A second, independently trained safety classifier (Llama-Guard-3-1B,
      already used elsewhere in the paper as an alternative scorer) is
      combined with the zero-shot NLI view by ``max``.  Because the
      combination is a max, the second view can only raise the raw risk score,
      never lower it, so the NLI operating point is preserved for everything
      the second view considers safe.  The calibrator is re-fitted on the
      combined scores, which is required: calibrating combined scores with a
      calibrator fitted on NLI scores would be invalid.
"""

from __future__ import annotations

import os

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

CALIB_CACHE_DIR = EXPERIMENTS / "cache"

from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402
from rcdp_guard.scorers import (  # noqa: E402
    DistilBartMNLIRiskScorer,
    LlamaGuardRiskScorer,
)
from rcdp_guard.types import Example, Turn  # noqa: E402

DATA = EXPERIMENTS / "data"
CALIBRATION = DATA / "paper_splits" / "full_pool_calibration.jsonl"
LOCAL_LG = str(EXPERIMENTS / "llama-guard-3-1b")


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


def build_scorer(kind: str):
    import torch

    torch.set_num_threads(16)
    nli = DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3", local_files_only=True, device="cpu"
    )
    if kind == "nli":
        return nli
    if kind == "ensemble":
        guard = LlamaGuardRiskScorer(
            model_name=LOCAL_LG,
            device="cpu",
            local_files_only=True,
            cache_path=str(EXPERIMENTS / "cache" / "llamaguard_tap_probe_cache.json"),
        )
        return EnsembleRiskScorer(nli, guard)
    raise ValueError(f"unknown scorer kind: {kind}")


def build_controller(kind: str, benchmark_aware: bool = False) -> RCDPController:
    controller = RCDPController(
        scorer=build_scorer(kind),
        feature_config=FeatureConfig.all(),
        config=ControllerConfig(benchmark_aware=benchmark_aware),
    )
    cache_path = CALIB_CACHE_DIR / f"calibrator_{kind}_ba{benchmark_aware}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            controller.calibrator = pickle.load(fh)
    else:
        fit_calibrator(controller, load_examples(CALIBRATION))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(controller.calibrator, fh)
    return controller


@dataclass
class Principal:
    """Sequential state of one caller.  RCDP sees this as one dialogue."""

    texts: list[str] = field(default_factory=list)
    latched: bool = False


class HardenedService:
    """Black-box endpoint.  The attacker observes block/allow only."""

    def __init__(self, controller: RCDPController, example: Example, bind: bool = True):
        self.controller = controller
        self.example = example
        self.bind = bind
        self.principals: dict[str, Principal] = {}
        self.queries = 0

    def query(self, text: str, principal_id: str = "p0") -> dict:
        self.queries += 1
        if not self.bind:
            self.principals[principal_id] = Principal()
        p = self.principals.setdefault(principal_id, Principal())
        if p.latched:
            # Monotone escalation: QUARANTINE is terminal under
            # allow_state_downgrade=False, so this is exact, not a shortcut.
            p.texts.append(text)
            return {"allowed": False, "state": "quarantine", "reason": "latched"}
        turns = [Turn(role="user", text=t) for t in p.texts]
        turns.append(Turn(role="user", text=text))
        ex = Example(
            example_id=self.example.example_id,
            benchmark=self.example.benchmark,
            label=self.example.label,
            turns=turns,
            metadata=dict(self.example.metadata),
        )
        records = self.controller.evaluate_example(ex)
        p.texts.append(text)
        if not records:
            return {"allowed": False, "state": "none"}
        last = records[-1]
        if last.state in {"hardened", "quarantine"}:
            p.latched = True
        return {
            "allowed": last.decision == "allow",
            "state": last.state,
            "calibrated": last.calibrated_score,
        }
