"""Per-caller service policies over the calibrated gate (revision r5: lockout baselines and controller ablations).

Two families of policy share the R1 calibration (scorer, risk head, Platt layer, thresholds):

  rcdp      ``rcdp_hardened.HardenedService``: the controller sees all queries of a caller as one session in
            submission order (bound identity); optional service latch.  A controller ablation changes
            ``ControllerConfig`` fields, or wraps the scorer, *after* the calibration is fitted, so the
            single-turn gate is the reference one (``check_single_turn_equivalence`` verifies this on the
            held-out prompts).
  lockout   stateless gate + count-based lockout (``LockoutService``): every query is decided as the first
            turn of a fresh session, which is exactly the decision of the 'fresh' TAP arm; after ``k`` strikes
            (queries whose one-turn state is in ``strike_states``) the caller is blocked for the rest of the
            session.  ``k=None`` never locks: the stateless gate itself.

Policy spec strings (used by policy_benign_cost.py):
  stateless                 lockout with k=None
  lockout:K                 k=K, a strike is a blocked query (one-turn state Quarantine)
  lockout:K:hq              k=K, a strike is a query routed to Hardened or Quarantine
  rcdp                      RCDP controller, bound caller, no latch
  rcdp+latch                ... with the service latch on Hardened/Quarantine (the submitted service)
  rcdp:ABL[,ABL...][+latch] ... with controller ablations from ABLATIONS
"""
from __future__ import annotations

import copy
import json
from collections import OrderedDict

from rcdp_hardened import HardenedService, parse_latch_states
from rcdp_guard.multiturn import DOWNGRADE, MEMORY

POLICIES_VERSION = "rcdp_policies/r5.0"

# name -> {"config": ControllerConfig overrides, "scorer": scorer wrapper name or None, "note": ...}
ABLATIONS = OrderedDict([
    ("downgrade", {
        "config": dict(DOWNGRADE),
        "scorer": None,
        "note": "asymmetric de-escalation of the paper family (rcdp_guard.multiturn.DOWNGRADE): the state may fall "
                "one level, and only after safe_streak_to_downgrade (k_down = 2) consecutive turns with calibrated "
                "risk < tau_c; history counters decay by one per turn",
    }),
    ("no-trajectory", {
        "config": {"use_trajectory_features": False},
        "scorer": None,
        "note": "trajectory terms off: no effective-risk boost from the decayed pressure and the trend, no "
                "threshold lowering, no Cautious->Hardened->Quarantine promotion by progress/memory",
    }),
    ("no-history-text", {
        "config": {},
        "scorer": "history-free",
        "note": "the risk scorer receives no conversation history (each query is scored as in a fresh session); "
                "the controller state, streaks and trajectory terms are kept",
    }),
    ("memory", {
        "config": dict(MEMORY),
        "scorer": None,
        "note": "asymmetric risk memory of the paper (rcdp_guard.multiturn.MEMORY: lambda 0.80, eta 0.90) on top "
                "of the default controller",
    }),
])

STATE_CODE = {"normal": "n", "cautious": "c", "hardened": "h", "quarantine": "q", None: "-", "none": "-"}


class HistoryFreeScorer:
    """Scores every query without conversation history (as the first turn of a fresh session)."""

    history_free = True

    def __init__(self, inner) -> None:
        self.inner = inner

    def score(self, text, history=None, metadata=None):
        return self.inner.score(text, [], metadata)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def parse_ablations(spec) -> tuple:
    if spec is None:
        return ()
    if isinstance(spec, str):
        names = [s.strip() for s in spec.replace("+", ",").split(",") if s.strip() and s.strip() != "none"]
    else:
        names = [str(s) for s in spec]
    bad = [n for n in names if n not in ABLATIONS]
    if bad:
        raise ValueError(f"unknown controller ablation(s) {bad}; known: {list(ABLATIONS)}")
    return tuple(dict.fromkeys(names))


def apply_ablations(controller, names):
    """Copy of ``controller`` (same scorer, calibrator and thresholds) with the ablations applied."""
    names = parse_ablations(names)
    if not names:
        return controller
    c = copy.copy(controller)
    c.config = copy.deepcopy(controller.config)
    for n in names:
        spec = ABLATIONS[n]
        for k, v in spec["config"].items():
            if not hasattr(c.config, k):
                raise AttributeError(f"ControllerConfig has no field {k!r} (ablation {n})")
            setattr(c.config, k, v)
        if spec["scorer"] == "history-free":
            c.scorer = HistoryFreeScorer(c.scorer)
    return c


def check_single_turn_equivalence(reference, other, examples) -> dict:
    """Every example evaluated as one-turn sessions under both controllers: decisions and states must agree."""
    n = diff = 0
    for ex in examples:
        if len(ex.turns) != 1 or ex.turns[0].role != "user":
            raise ValueError(f"{ex.example_id}: single-turn check needs one-turn examples")
        a = reference.evaluate_example(ex)
        b = other.evaluate_example(ex)
        n += 1
        diff += (a[-1].decision, a[-1].state) != (b[-1].decision, b[-1].state)
    return {"examples": n, "differing_decision_or_state": diff}


class LockoutService:
    """Stateless gate + count-based lockout.  Same interface and record fields as HardenedService.

    A query of a caller that is not locked gets the one-turn decision of the gate.  The query is a strike if its
    one-turn state is in ``strike_states``; after ``k`` strikes the caller is locked and every later query is
    blocked without calling the controller (reason 'latched', as the service latch of HardenedService).
    """

    def __init__(self, controller, example, k=None, strike_states=("quarantine",)) -> None:
        if k is not None and int(k) < 1:
            raise ValueError("lockout k must be >= 1 (or None for the stateless gate)")
        self.controller = controller
        self.example = example
        self.k = None if k is None else int(k)
        self.strike_states = parse_latch_states(strike_states) if self.k is not None else ()
        self.inner = HardenedService(controller, example, bind=False, latch_states=())
        self.principals = {}
        self.queries = 0

    def stateless_view(self, text: str) -> dict:
        return self.inner.stateless_view(text)

    def query(self, text: str, principal_id: str = "p0", shadow: bool = False) -> dict:
        self.queries += 1
        p = self.principals.setdefault(principal_id, {"turns": 0, "strikes": 0, "locked_at": None, "locked_by": None})
        p["turns"] += 1
        locked_before = p["locked_at"] is not None
        if locked_before:
            out = {"allowed": False, "decision": "block", "state": None, "reason": "latched",
                   "controller_called": False, "calibrated": None, "raw_score": None, "effective_risk": None,
                   "latch_trigger_state": p["locked_by"]}
            if shadow:
                # What the gate alone would decide (for the lockout policy the 'controller' is the stateless gate).
                v = self.inner.stateless_view(text)
                out.update({"shadow_" + k[len("stateless_"):]: val for k, val in v.items()})
        else:
            v = self.inner.stateless_view(text)
            rec = {k[len("stateless_"):]: val for k, val in v.items()}
            out = {"allowed": rec["decision"] == "allow", "reason": "controller", "controller_called": True,
                   "latch_trigger_state": None, **rec}
            if self.k is not None and rec["state"] in self.strike_states:
                p["strikes"] += 1
                if p["strikes"] >= self.k:
                    p["locked_at"] = p["turns"]
                    p["locked_by"] = rec["state"]
        out.update({
            "principal_id": principal_id,
            "session_turn": p["turns"],
            "latched_before": locked_before,
            "latched_after": p["locked_at"] is not None,
            "latch_states": list(self.strike_states),
            "strikes": p["strikes"],
        })
        return out


def policy_record(kind: str, *, k=None, strike_states=("quarantine",), ablations=(), latch_states=()) -> dict:
    """JSON description of a policy (enters the TAP fingerprint)."""
    if kind == "lockout":
        return {
            "kind": "lockout",
            "k": k,
            "strike_states": list(parse_latch_states(strike_states)) if k is not None else [],
            "gate": "one-turn decision of the calibrated controller for every query (as the fresh TAP arm)",
            "rule": ("never locks (stateless gate)" if k is None else
                     f"caller blocked for the rest of the session after {k} queries whose one-turn state is in "
                     f"{list(parse_latch_states(strike_states))}"),
            "version": POLICIES_VERSION,
        }
    if kind != "rcdp":
        raise ValueError(f"unknown policy kind {kind!r}")
    abl = parse_ablations(ablations)
    return {
        "kind": "rcdp",
        "controller_ablations": list(abl),
        "controller_overrides": {k_: v for n in abl for k_, v in ABLATIONS[n]["config"].items()},
        "scorer_wrapper": next((ABLATIONS[n]["scorer"] for n in abl if ABLATIONS[n]["scorer"]), None),
        "ablation_notes": {n: ABLATIONS[n]["note"] for n in abl},
        "latch_states": list(parse_latch_states(latch_states)),
        "version": POLICIES_VERSION,
    }


def parse_policy_spec(spec: str) -> dict:
    """'stateless' | 'lockout:K[:hq]' | 'rcdp[:ABL,...][+latch]' -> policy record (see module docstring)."""
    s = spec.strip()
    if s == "stateless":
        rec = policy_record("lockout", k=None)
    elif s.startswith("lockout:"):
        parts = s.split(":")
        if len(parts) not in (2, 3) or not parts[1].isdigit() or (len(parts) == 3 and parts[2] != "hq"):
            raise ValueError(f"bad lockout spec {spec!r}; expected lockout:K or lockout:K:hq")
        strikes = ("hardened", "quarantine") if len(parts) == 3 else ("quarantine",)
        rec = policy_record("lockout", k=int(parts[1]), strike_states=strikes)
    elif s == "rcdp" or s.startswith("rcdp:") or s.startswith("rcdp+"):
        latch = ()
        if s.endswith("+latch"):
            latch = ("hardened", "quarantine")
            s = s[: -len("+latch")]
        abl = s.split(":", 1)[1] if ":" in s else ""
        rec = policy_record("rcdp", ablations=abl, latch_states=latch)
    else:
        raise ValueError(f"unknown policy spec {spec!r}")
    rec["spec"] = spec.strip()
    return rec


def make_service(record: dict, controller, example):
    """Service for one caller namespace.  ``controller`` must already carry the record's ablations."""
    if record["kind"] == "lockout":
        return LockoutService(controller, example, k=record["k"], strike_states=record["strike_states"] or ("quarantine",))
    return HardenedService(controller, example, bind=True, latch_states=record["latch_states"])


def record_digest(record: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
