"""Per-request latency of every stage of the RCDP defence path on one host (revision R1).

Ledger R2-4 / R3-5 (Fig. 5 and the appendix latency/token tables) and X-8 (judge
latency denominator). One run = one scorer x one device. Every number in the
output comes from per-request ``time.perf_counter`` timings taken in this
process, at batch size 1 and concurrency 1, after warm-up, with every
scorer/guard cache cleared before each request (no request is served from a
cache). The host (CPU model, GPU model, torch/transformers versions, threads,
dtypes) is recorded in ``meta.environment`` and ``meta.models``.

Stages (one request = one held-out example; ``protocol.boundaries`` in the output):

  scorer_ms                 RiskScorer.score (DistilBART/DeBERTa: two NLI forward passes;
                            Llama-Guard: chat template + generate + parse)
  features_head_platt_ms    controller.calibrate: feature building + risk head + Platt
    head_platt_ms           of which calibrator.predict_proba / decision_function
    features_ms             = features_head_platt_ms - head_platt_ms
  policy_router_ms          rest of controller.evaluate_example (uncertainty, trajectory
                            terms, select_state, decision, bookkeeping)
    select_state_ms         of which controller.select_state
  controller_total_ms       controller.evaluate_example = scorer + features_head_platt + policy_router
  controller_incremental_ms controller_total - scorer of the same request
  controller_incremental_replay_ms
                            evaluate_example with a replay scorer (dict lookup of the request's
                            own score), no instrumentation: the direct measurement of the
                            "risk estimation + Platt + routing" row (the submitted 1.20 ms was a
                            difference of two separately measured means)
  prompt_assembly_ms        [state instruction, request] -> chat messages (nothing when quarantined)
  pre_generation_path_ms    controller_total + prompt_assembly (text in -> refusal or prompt out)
  llm_ms                    protected-model call (allowed requests only; optional)
  output_guard_ms           Hardened output check (Hardened generations only; optional)
  response_judge_ms         evaluation judge on every generation; NOT part of the defence path
  defense_path_ms           pre_generation_path + llm + output_guard

Alternative paths, same boundary (text in -> decision/prompt out), same requests:

  static_assembly_ms        static prompt patch: [patch, request] -> messages (+ static_llm_ms,
                            the protected-model call on every request, with --static-patch-llm)
  lg_oneshot_ms             Llama-Guard-3-1B one-shot prompt verdict (block iff unsafe), through
                            e2e_common.LlamaGuardOneShotGate, the gate of the Table 1 baseline run
  smoothllm_ms              SmoothLLM-style prompt guard: N perturbed copies, Llama Guard on
                            each, majority vote (baselines_r1/smoothllm.SmoothLLMPromptGuardGate)

Conditional stages are summarised over the calls actually made (``n_calls``;
the X-8 fix) and, separately, amortised over all requests. ``counts`` gives the
number of calls of every model (scorer, protected model, output guard,
response judge, alternative-path guard).

Token overhead (Fig. 5b; needs ``--prompt-tokenizer`` or a local protected model):
per request, the tokens of the hidden instruction actually sent (0 when the
request is quarantined: no generation), the templated input tokens with and
without it, and the same for the static patch. ``token_overhead`` summarises
them; the configured budgets of ControllerConfig.state_prompt_tokens (6/28/52/18,
the submitted "Mean Tokens") are reported separately and are not token counts.

Sub-command ``e2e-summary`` re-reads saved end-to-end result files (the
submitted harness and the R1 harness) and reports judge / output-guard /
generation latency per actual call.

Examples:
    python scripts/measure_latency_r1.py measure --dataset data/paper_splits/heldout_204.jsonl \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --scorer distilbart-mnli --device cpu --threads 16 --warmup 10 --repeats 3 \
        --output results/r1/latency/distilbart_cpu.json
    python scripts/measure_latency_r1.py e2e-summary --inputs "results/e2e_*_heldout204.json" \
        --output results/r1/latency/e2e_saved_per_call.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e2e_common as E  # noqa: E402
from scorer_baselines_r1.common import (  # noqa: E402
    SCORER_KINDS, build_scorer, clear_scorer_cache, code_version, environment, evenly_spaced, model_info, now,
    sha256_text, write_json,
)
from stats_r1 import cluster_bootstrap_ci, describe, quantile  # noqa: E402

from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.calibration_protocol import DEFAULT_SEED  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import fit_calibrator  # noqa: E402
from rcdp_guard.types import ScoreResult  # noqa: E402

STATE_INSTRUCTIONS = E.STATE_INSTRUCTIONS
REFUSAL = E.CONTROLLER_REFUSAL
ALT_PATHS = ("oneshot", "smoothllm")

BOUNDARIES = {
    "scorer_ms": "RiskScorer.score incl. device sync (NLI: two forward passes; Llama-Guard: template+generate+parse)",
    "features_head_platt_ms": "controller.calibrate: feature building + risk head + Platt",
    "head_platt_ms": "calibrator.predict_proba + decision_function inside calibrate",
    "features_ms": "features_head_platt_ms - head_platt_ms",
    "policy_router_ms": "evaluate_example minus scorer and calibrate: uncertainty, trajectory terms, select_state, "
                        "decision, record bookkeeping",
    "select_state_ms": "controller.select_state, part of policy_router_ms",
    "controller_total_ms": "controller.evaluate_example (scorer + features_head_platt + policy_router)",
    "controller_incremental_ms": "controller_total_ms - scorer_ms of the same request (instrumented run)",
    "controller_incremental_replay_ms": "evaluate_example with a replay scorer (dict lookup of the request's own "
                                        "score), no instrumentation",
    "prompt_assembly_ms": "[state instruction, request] -> chat messages; nothing to build when quarantined",
    "pre_generation_path_ms": "controller_total + prompt_assembly (text in -> refusal or prompt out)",
    "llm_ms": "e2e_common generator.generate(messages): local = chat template + tokenisation + generate + decode; "
              "API = HTTP round trip incl. network and server queueing (allowed requests only)",
    "output_guard_ms": "Llama-Guard on (request, response) for Hardened generations (defence path)",
    "response_judge_ms": "evaluation judge on every generation (not part of the defence path)",
    "defense_path_ms": "pre_generation_path + llm + output_guard",
    "static_assembly_ms": "[static patch prompt, request] -> chat messages (no scorer, no controller)",
    "static_llm_ms": "protected-model call on the static path (every request; optional)",
    "static_path_ms": "static_assembly + static_llm",
    "lg_oneshot_ms": "e2e_common.LlamaGuardOneShotGate.decide: Llama-Guard prompt verdict on the last user turn "
                     "(template + generate + parse), block iff unsafe; guard cache cleared before the call",
    "smoothllm_ms": "baselines_r1.smoothllm.SmoothLLMPromptGuardGate.decide: N perturbed copies, one Llama-Guard "
                    "verdict per copy (cache cleared before every copy), majority vote",
}
PRIMARY_METRICS = [
    "scorer_ms", "features_head_platt_ms", "head_platt_ms", "features_ms", "policy_router_ms", "select_state_ms",
    "controller_total_ms", "controller_incremental_ms", "prompt_assembly_ms", "pre_generation_path_ms",
    "defense_path_ms", "static_assembly_ms", "static_path_ms",
]
CONDITIONAL = {  # metric -> flag marking whether the call happened
    "llm_ms": "llm_called", "output_guard_ms": "output_guard_called",
    "response_judge_ms": "response_judge_called", "static_llm_ms": "static_llm_called",
    "lg_oneshot_ms": "lg_oneshot_called", "smoothllm_ms": "smoothllm_called",
}
PRINT_ORDER = ("scorer_ms", "features_head_platt_ms", "policy_router_ms", "controller_incremental_ms",
               "controller_incremental_replay_ms", "controller_total_ms", "prompt_assembly_ms",
               "pre_generation_path_ms", "llm_ms", "output_guard_ms", "response_judge_ms", "defense_path_ms",
               "static_path_ms", "lg_oneshot_ms", "smoothllm_ms")


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

class Clock:
    def __init__(self, sync):
        self.sync = sync
        self.acc = defaultdict(float)
        self.calls = defaultdict(int)

    def reset(self):
        self.acc.clear()
        self.calls.clear()

    def wrap(self, name, fn):
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.acc[name] += (time.perf_counter() - t0) * 1000.0
                self.calls[name] += 1
        return timed


class TimedScorer:
    """Times score() with device synchronisation and forwards every other attribute.

    Each call is logged as (text, history snapshot, result) after the timer
    stops, so the replay scorer of the un-instrumented increment can be filled
    without re-scoring.
    """

    def __init__(self, inner, clock):
        self._inner = inner
        self._clock = clock
        self.log = []

    def score(self, text, history=None, metadata=None):
        c = self._clock
        c.sync()
        t0 = time.perf_counter()
        res = self._inner.score(text, history, metadata)
        c.sync()
        c.acc["scorer"] += (time.perf_counter() - t0) * 1000.0
        c.calls["scorer"] += 1
        self.log.append((text, tuple(history or ()), res))
        return res

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedCalibrator:
    def __init__(self, inner, clock):
        self._inner = inner
        self.predict_proba = clock.wrap("head_platt", inner.predict_proba)
        if hasattr(inner, "decision_function"):
            self.decision_function = clock.wrap("head_platt", inner.decision_function)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ReplayScorer:
    """Returns already-computed scores (dict lookup) for the un-instrumented increment."""

    def __init__(self):
        self.table = {}

    def put(self, text, history, result):
        self.table[(text, tuple(history or ()))] = result

    def score(self, text, history=None, metadata=None):
        return self.table[(text, tuple(history or ()))]


class NoCacheClassifier:
    """Prompt classifier wrapper: clears the guard's caches before every call and counts calls.

    SmoothLLM copies of a short prompt can be identical (q% of fewer than 1/q
    characters is 0 characters), and repeated passes resend the same texts;
    neither may be answered from the Llama-Guard scorer cache.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def classify(self, text):
        clear_scorer_cache(self.inner)
        self.calls += 1
        return self.inner.classify(text)

    def describe(self):
        return self.inner.describe() if hasattr(self.inner, "describe") else {"backend": type(self.inner).__name__}


# --------------------------------------------------------------------------
# Downstream components and stand-ins (smoke tests only)
# --------------------------------------------------------------------------

class StubGenerator:
    """Stand-in for e2e_common generators (smoke tests only)."""

    kind = "stub"

    def __init__(self, ms):
        self.ms = ms

    def generate(self, messages, item_seed=None):
        if self.ms:
            time.sleep(self.ms / 1000.0)
        return {"text": "stub response", "n_new_tokens": None}

    def settings(self):
        return {"backend": "stub", "latency_ms": self.ms}


class StubJudge:
    """Stand-in for E.LlamaGuardResponseJudge: 'unsafe' iff the request contains a listed term."""

    def __init__(self, ms, terms=("bomb", "explosive", "weapon")):
        self.ms, self.terms = ms, terms

    def judge(self, request, response):
        if self.ms:
            time.sleep(self.ms / 1000.0)
        return {"verdict": "unsafe" if any(t in request.lower() for t in self.terms) else "safe"}

    def describe(self):
        return {"backend": "stub", "latency_ms": self.ms}


class StubPromptClassifier:
    """Stand-in for E.LlamaGuardPromptClassifier: 'unsafe' iff the text contains a listed term."""

    def __init__(self, ms, terms=("bomb", "explosive", "weapon", "malware", "poison")):
        self.ms, self.terms = ms, terms

    def classify(self, text):
        if self.ms:
            time.sleep(self.ms / 1000.0)
        unsafe = any(t in text.lower() for t in self.terms)
        return {"verdict": "unsafe" if unsafe else "safe", "raw_score": 0.99 if unsafe else 0.01, "latency_ms": 0.0}

    def describe(self):
        return {"backend": "stub", "latency_ms": self.ms}


def build_generator(args, devices):
    if args.protected_backend == "none":
        return None
    if args.protected_backend == "stub":
        return StubGenerator(args.stub_latency_ms)
    if args.protected_backend == "openai-api":
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not (args.api_base_url and args.api_model and key):
            raise SystemExit("--protected-backend openai-api needs --api-base-url, --api-model and an API key")
        return E.OpenAIChatGenerator(args.api_base_url, args.api_model, key, temperature=args.temperature,
                                     max_new_tokens=args.max_new_tokens)
    if not args.model_path:
        raise SystemExit("--protected-backend local needs --model-path")
    devices.add(args.protected_device)
    return E.LocalHFGenerator(args.model_path, device=args.protected_device, torch_dtype=args.protected_dtype,
                              temperature=args.temperature, max_new_tokens=args.max_new_tokens)


def build_judge(kind, path, device, devices, stub_ms, role):
    if kind == "none":
        return None
    if kind == "stub":
        return StubJudge(stub_ms)
    if not path:
        raise SystemExit(f"--{role}-backend llama-guard needs --{role}-model-path")
    devices.add(device)
    return E.LlamaGuardResponseJudge(path, device=device, cache_path=None, role=role)


def build_alt_classifier(args, raw_scorer, devices):
    """Llama-Guard prompt classifier for the alternative paths; reuses the scorer when it is the same model."""
    if args.alt_guard_backend == "none":
        return None
    if args.alt_guard_backend == "stub":
        return StubPromptClassifier(args.stub_latency_ms)
    path = args.alt_guard_model_path
    if not path:
        raise SystemExit("--alt-guard-backend llama-guard needs --alt-guard-model-path")
    device = args.alt_guard_device or args.device
    devices.add(device)
    same = (args.scorer == "llama-guard" and str(args.scorer_model_name) == str(path) and args.device == device
            and (args.alt_guard_dtype or None) == (args.dtype or None)
            and getattr(args, "scorer_score_mode", "label") == "label")
    lg = raw_scorer if same else build_scorer("llama-guard", path, device, args.local_files_only,
                                              args.alt_guard_dtype)
    return E.LlamaGuardPromptClassifier(model_path=path, device=device, scorer=lg)


def generator_settings(gen):
    if gen is None:
        return None
    s = dict(gen.settings())
    s.pop("chat_template_text", None)
    s.pop("api_base_url", None)
    if s.get("model"):
        s["model"] = Path(str(s["model"])).name
    if hasattr(gen, "model"):
        s["model_info"] = model_info(gen.model) if hasattr(gen.model, "parameters") else None
    return s


def judge_settings(judge):
    if judge is None:
        return None
    d = dict(judge.describe())
    if d.get("model"):
        d["model"] = Path(str(d["model"])).name
    guard = getattr(judge, "guard", None) or getattr(judge, "scorer", None)
    if guard is not None:
        d["model_info"] = model_info(guard)
    return d


class TokenCounter:
    """Token counts with the protected model's tokenizer (outside every timed region)."""

    def __init__(self, tok, static_prompt, source):
        self.tok = tok
        self.source = source
        self.errors = defaultdict(int)
        self.instruction = {s: self.ntok(t) for s, t in STATE_INSTRUCTIONS.items()}
        self.static = self.ntok(static_prompt)
        self.refusal = self.ntok(REFUSAL)

    def ntok(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def templated(self, messages):
        """Input tokens of the rendered chat prompt (as e2e_common.LocalHFGenerator sends it), or None."""
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return len(self.tok(prompt, add_special_tokens=False)["input_ids"])
        except Exception as exc:  # noqa: BLE001 - e.g. no chat template, or no system role
            self.errors[f"{type(exc).__name__}: {str(exc)[:160]}"] += 1
            return None

    def describe(self):
        return {"tokenizer_class": type(self.tok).__name__, "source": self.source,
                "state_instruction_tokens": dict(self.instruction), "static_patch_tokens": self.static,
                "controller_refusal_tokens": self.refusal,
                "chat_template_errors": dict(self.errors)}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def token_overhead(rows, counter):
    """Fig. 5(b): hidden-prompt overhead of RCDP vs the static patch over the first pass."""
    first = [r for r in rows if r["rep"] == 0]
    out = {"n_requests": len(first)}
    if any("configured_prompt_tokens" in r for r in first):
        out["configured_budget_mean_all_requests"] = _mean(r.get("configured_prompt_tokens") for r in first)
        out["configured_budget_note"] = ("mean of ControllerConfig.state_prompt_tokens (6/28/52/18 incl. Quarantine) "
                                         "over all requests = the submitted 'Mean Tokens' quantity; configured "
                                         "budgets, not token counts")
    if counter is None:
        out["skipped"] = "no tokenizer (--prompt-tokenizer or a local protected model)"
        return out
    gen = [r for r in first if r.get("decision") == "allow"]
    out.update({
        "tokenizer": counter.describe(),
        "n_generated": len(gen),
        "user_tokens_mean": _mean(r.get("user_tokens") for r in first),
        "base_input_tokens_mean": _mean(r.get("base_input_tokens") for r in first),
        "rcdp": {
            "hidden_prompt_tokens_mean_all_requests": _mean(r.get("hidden_prompt_tokens") for r in first),
            "hidden_prompt_tokens_mean_generated": _mean(r.get("hidden_prompt_tokens") for r in gen),
            "added_input_tokens_mean_all_requests": _mean(r.get("added_input_tokens") for r in first),
            "added_input_tokens_mean_generated": _mean(r.get("added_input_tokens") for r in gen),
            "input_tokens_mean_generated": _mean(r.get("input_tokens") for r in gen),
            "hidden_prompt_tokens_sum": sum(r.get("hidden_prompt_tokens") or 0 for r in first),
        },
        "static_patch": {
            "hidden_prompt_tokens_mean": _mean(r.get("static_hidden_prompt_tokens") for r in first),
            "added_input_tokens_mean": _mean(r.get("static_added_input_tokens") for r in first),
            "input_tokens_mean": _mean(r.get("static_input_tokens") for r in first),
            "hidden_prompt_tokens_sum": sum(r.get("static_hidden_prompt_tokens") or 0 for r in first),
        },
        "definitions": {
            "hidden_prompt_tokens": "tokens of the hidden instruction text actually sent with the request "
                                    "(add_special_tokens=False); 0 for a quarantined request (refusal, no "
                                    "generation). Paper: Delta_tok(t) = tok(P(state)), averaged over requests",
            "added_input_tokens": "templated input tokens with the hidden instruction as system turn minus "
                                  "templated input tokens of the bare user turn (includes the chat template's "
                                  "system-turn markup); 0 for a quarantined request",
            "static_patch": "the same quantities for [static patch as system turn, request] on every request",
        },
    })
    return out


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    m = sub.add_parser("measure", help="time every stage per request on this host")
    m.add_argument("--dataset", required=True)
    m.add_argument("--calibration-dataset", default=None, help="required unless --scorer-only")
    m.add_argument("--output", required=True)
    m.add_argument("--scorer", default="distilbart-mnli", choices=SCORER_KINDS)
    m.add_argument("--scorer-model-name", default=None, help="default per --scorer; a local path for llama-guard")
    m.add_argument("--scorer-score-mode", default="label", choices=["label", "prob"],
                   help="llama-guard scorer only: label = verdict (default, submitted), prob = P(unsafe) "
                        "(the RCDP Llama-Guard configuration)")
    m.add_argument("--device", default="cpu", help="scorer device")
    m.add_argument("--dtype", default=None, help="scorer dtype (default: as loaded; Llama-Guard: bf16 on GPU)")
    m.add_argument("--local-files-only", action="store_true")
    m.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    m.add_argument("--scorer-only", action="store_true", help="time the scorer alone (no calibration fit/controller)")
    m.add_argument("--setting", default="agnostic", choices=["agnostic", "aware"])
    m.add_argument("--num-states", type=int, default=4, choices=[2, 3, 4])
    m.add_argument("--protocol", default=None, help="calibration protocol (default: RCDP_CALIBRATION_PROTOCOL or r1)")
    m.add_argument("--split-seed", default=DEFAULT_SEED)
    m.add_argument("--warmup", type=int, default=10)
    m.add_argument("--repeats", type=int, default=1, help="passes over the dataset")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--incremental-reps", type=int, default=20)
    m.add_argument("--prompt-tokenizer", default=None,
                   help="tokenizer of the protected model: hidden-prompt and input token counts (not timed)")
    m.add_argument("--static-patch-prompt-file", default=None,
                   help="static patch text (default: the Hardened instruction, = baselines_r1 preset 'hardened-state')")
    g = m.add_argument_group("alternative safeguards (Fig. 5a)")
    g.add_argument("--alt-guard-backend", default="none", choices=["none", "llama-guard", "stub"],
                   help="Llama-Guard-3-1B prompt classifier for the one-shot and SmoothLLM-style paths")
    g.add_argument("--alt-guard-model-path", default=None)
    g.add_argument("--alt-guard-device", default=None, help="default: --device")
    g.add_argument("--alt-guard-dtype", default=None, help="default: scorer default (bf16 on GPU, fp32 on CPU)")
    g.add_argument("--alt-paths", default="oneshot,smoothllm", help="comma list of: " + ", ".join(ALT_PATHS))
    g.add_argument("--oneshot-limit", type=int, default=0,
                   help="time the one-shot path on this many requests spread over the dataset (0 = all)")
    g.add_argument("--smoothllm-limit", type=int, default=0,
                   help="time the SmoothLLM-style path on this many requests spread over the dataset (0 = all)")
    g.add_argument("--alt-repeats", type=int, default=1,
                   help="time the alternative paths in the first K passes only (default 1; slow on CPU)")
    g.add_argument("--smoothllm-copies", type=int, default=10)
    g.add_argument("--smoothllm-pert-type", default="swap", choices=["swap", "patch", "insert"])
    g.add_argument("--smoothllm-pert-pct", type=float, default=10.0)
    g.add_argument("--smoothllm-vote-threshold", type=float, default=0.5)
    g.add_argument("--smoothllm-seed", type=int, default=20261018)
    g = m.add_argument_group("downstream components (optional)")
    g.add_argument("--protected-backend", default="none", choices=["none", "openai-api", "local", "stub"])
    g.add_argument("--api-base-url", default=None)
    g.add_argument("--api-model", default=None)
    g.add_argument("--api-key", default=None, help="prefer OPENAI_API_KEY in the environment")
    g.add_argument("--model-path", default=None, help="protected model directory (local backend)")
    g.add_argument("--protected-device", default="cuda")
    g.add_argument("--protected-dtype", default="bfloat16")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--static-patch-llm", action="store_true", help="also call the protected model on the static path")
    g.add_argument("--output-guard-backend", default="none", choices=["none", "llama-guard", "stub"])
    g.add_argument("--output-guard-model-path", default=None)
    g.add_argument("--output-guard-device", default="cpu")
    g.add_argument("--response-judge-backend", default="none", choices=["none", "llama-guard", "stub"])
    g.add_argument("--response-judge-model-path", default=None)
    g.add_argument("--response-judge-device", default="cpu")
    g.add_argument("--stub-latency-ms", type=float, default=0.0)
    m.add_argument("--bootstrap", type=int, default=2000)
    m.add_argument("--seed", type=int, default=20261113)
    e = sub.add_parser("e2e-summary", help="per-call latencies from saved end-to-end result files")
    e.add_argument("--inputs", nargs="+", required=True, help="files or glob patterns")
    e.add_argument("--output", required=True)
    return ap.parse_args(argv)


def summarise(rows, metric, flag=None, B=2000, seed=0, n_requests=None):
    use = [r for r in rows if r.get(metric) is not None and (flag is None or r.get(flag))]
    xs = [r[metric] for r in use]
    d = describe(xs)
    if flag is not None:
        d["n_calls"] = len(xs)
        d["n_requests"] = n_requests
        d["amortised_mean_per_request"] = (sum(xs) / n_requests) if n_requests else None
    if len(xs) >= 2 and B > 0:
        repeats = len(use) != len({r["example_id"] for r in use})
        for stat, fn in (("mean", lambda s: sum(r[metric] for r in s) / len(s)),
                         ("p50", lambda s: quantile([r[metric] for r in s], 0.5))):
            ci = cluster_bootstrap_ci(use, fn, cluster_key=lambda r: r["example_id"], B=B, seed=seed)
            d[f"{stat}_ci95"] = [ci["lo"], ci["hi"]]
        d["ci_method"] = (f"percentile bootstrap, B={B}, seed={seed}, "
                          + ("requests resampled as clusters of their repeats" if repeats else "requests resampled"))
    return d


def measure(args, scorer=None, tokenizer=None, alt_classifier=None, argv=None):
    devices = set()
    if str(args.device).startswith("cuda"):
        devices.add(args.device)
    torch = None
    try:
        import torch as _torch

        torch = _torch
        if args.threads:
            torch.set_num_threads(args.threads)
    except ModuleNotFoundError:
        needs_torch = (scorer is None and args.scorer != "keyword") or \
                      (alt_classifier is None and args.alt_guard_backend == "llama-guard")
        if needs_torch:
            raise

    def sync():
        if torch is not None:
            for d in devices:
                if str(d).startswith("cuda"):
                    torch.cuda.synchronize(d)

    t_start = time.perf_counter()
    examples = load_examples(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    raw_scorer = scorer or build_scorer(args.scorer, args.scorer_model_name, args.device, args.local_files_only,
                                        args.dtype, score_mode=getattr(args, "scorer_score_mode", "label"))
    clock = Clock(sync)
    timed_scorer = TimedScorer(raw_scorer, clock)
    replay = ReplayScorer()

    controller, calib = None, None
    if not args.scorer_only:
        if not args.calibration_dataset:
            raise SystemExit("--calibration-dataset is required unless --scorer-only")
        cal_examples = load_examples(args.calibration_dataset)
        overlap = {e.example_id for e in cal_examples} & {e.example_id for e in examples}
        if overlap:
            raise SystemExit(f"calibration and evaluation sets share {len(overlap)} example IDs")
        controller = RCDPController(scorer=raw_scorer, feature_config=FeatureConfig.all(),
                                    config=ControllerConfig(benchmark_aware=args.setting == "aware",
                                                            num_states=args.num_states))
        report = fit_calibrator(controller, cal_examples, protocol=args.protocol, seed=args.split_seed)
        calib = {k: report.get(k) for k in ("protocol", "seed", "thresholds", "benchmark_thresholds", "platt")}
        calib["split_sizes"] = (report.get("split") or {}).get("sizes")
        controller.scorer = timed_scorer
        controller.calibrator = TimedCalibrator(controller.calibrator, clock)
        controller.calibrate = clock.wrap("calibrate", controller.calibrate)
        controller.select_state = clock.wrap("select_state", controller.select_state)

    gen = build_generator(args, devices)
    guard = build_judge(args.output_guard_backend, args.output_guard_model_path, args.output_guard_device, devices,
                        args.stub_latency_ms, "output-guard")
    judge = build_judge(args.response_judge_backend, args.response_judge_model_path, args.response_judge_device,
                        devices, args.stub_latency_ms, "response-judge")

    alt_paths = [p.strip() for p in args.alt_paths.split(",") if p.strip()]
    bad = [p for p in alt_paths if p not in ALT_PATHS]
    if bad:
        raise SystemExit(f"unknown --alt-paths {bad}; expected {ALT_PATHS}")
    if alt_classifier is None:
        alt_classifier = build_alt_classifier(args, raw_scorer, devices)
    alt = NoCacheClassifier(alt_classifier) if alt_classifier is not None else None
    oneshot_gate = smooth_gate = None
    if alt is not None and "oneshot" in alt_paths:
        oneshot_gate = E.LlamaGuardOneShotGate(alt)
    if alt is not None and "smoothllm" in alt_paths:
        if str(HERE / "baselines_r1") not in sys.path:
            sys.path.insert(0, str(HERE / "baselines_r1"))
        import smoothllm as S  # noqa: WPS433 - baselines_r1/smoothllm.py

        smooth_gate = S.SmoothLLMPromptGuardGate(alt, args.smoothllm_copies, args.smoothllm_pert_type,
                                                 args.smoothllm_pert_pct, args.smoothllm_seed,
                                                 args.smoothllm_vote_threshold)
    oneshot_ids = {e.example_id for e in evenly_spaced(examples, args.oneshot_limit)} if oneshot_gate else set()
    smooth_ids = {e.example_id for e in evenly_spaced(examples, args.smoothllm_limit)} if smooth_gate else set()

    tok, tok_source = tokenizer, ("injected" if tokenizer is not None else None)
    if tok is None and args.prompt_tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.prompt_tokenizer, local_files_only=args.local_files_only)
        tok_source = Path(str(args.prompt_tokenizer)).name
    elif tok is None and getattr(gen, "tokenizer", None) is not None:
        tok, tok_source = gen.tokenizer, "protected model (--model-path)"
    static_prompt = (Path(args.static_patch_prompt_file).read_text(encoding="utf-8").strip()
                     if args.static_patch_prompt_file else STATE_INSTRUCTIONS["hardened"])
    counter = TokenCounter(tok, static_prompt, tok_source) if tok is not None else None

    def timed(fn, *a, gpu=False):
        if gpu:
            sync()
        t0 = time.perf_counter()
        out = fn(*a)
        if gpu:
            sync()
        return out, (time.perf_counter() - t0) * 1000.0

    def alternatives(ex, row, warmup=False, first_warmup=False):
        """Alternative safeguards on the same request (warm-up: one-shot always, SmoothLLM once)."""
        if not warmup and row["rep"] >= args.alt_repeats:
            return
        if oneshot_gate is not None and (warmup or ex.example_id in oneshot_ids):
            calls = alt.calls
            out, ms = timed(oneshot_gate.decide, ex, gpu=True)
            row.update({"lg_oneshot_called": True, "lg_oneshot_ms": ms, "lg_oneshot_decision": out["decision"],
                        "lg_oneshot_verdict": ((out.get("details") or {}).get("classifier") or {}).get("verdict"),
                        "lg_oneshot_guard_calls": alt.calls - calls})
        if smooth_gate is not None and ((warmup and first_warmup) or (not warmup and ex.example_id in smooth_ids)):
            calls = alt.calls
            out, ms = timed(smooth_gate.decide, ex, gpu=True)
            det = out.get("details") or {}
            row.update({"smoothllm_called": True, "smoothllm_ms": ms, "smoothllm_decision": out["decision"],
                        "smoothllm_unsafe_fraction": det.get("unsafe_fraction"),
                        "smoothllm_unknown_copies": det.get("n_unknown"),
                        "smoothllm_guard_calls": alt.calls - calls})

    def static_path(request, row):
        smsg, row["static_assembly_ms"] = timed(
            lambda: [{"role": "system", "content": static_prompt}, {"role": "user", "content": request}])
        row["static_llm_called"] = bool(gen is not None and args.static_patch_llm)
        if row["static_llm_called"]:
            g, row["static_llm_ms"] = timed(gen.generate, smsg, gpu=True)
            row["static_generated_tokens"] = g.get("n_new_tokens")
        row["static_path_ms"] = row["static_assembly_ms"] + row.get("static_llm_ms", 0.0)
        return smsg

    def count_tokens(row, request, messages, smsg):
        if counter is None:
            return
        base = counter.templated([{"role": "user", "content": request}])
        row["user_tokens"] = counter.ntok(request)
        row["base_input_tokens"] = base
        if "decision" in row:
            if messages is not None:
                row["hidden_prompt_tokens"] = counter.instruction[row["state"]]
                row["input_tokens"] = counter.templated(messages)
                row["added_input_tokens"] = (row["input_tokens"] - base
                                             if row["input_tokens"] is not None and base is not None else None)
            else:
                row["hidden_prompt_tokens"] = 0
                row["input_tokens"] = None
                row["added_input_tokens"] = 0
        row["static_hidden_prompt_tokens"] = counter.static
        row["static_input_tokens"] = counter.templated(smsg)
        row["static_added_input_tokens"] = (row["static_input_tokens"] - base
                                            if row["static_input_tokens"] is not None and base is not None else None)

    def one_request(ex, rep, warmup=False, first_warmup=False):
        request = E.last_user_text(ex)
        row = {"rep": rep, "example_id": ex.example_id, "label": ex.label.value, "benchmark": ex.benchmark,
               "n_user_turns": sum(t.role == "user" for t in ex.turns)}
        clear_scorer_cache(raw_scorer)
        clock.reset()
        if controller is None:
            history = [t.text for t in ex.turns[:-1]]
            _, row["scorer_ms"] = timed(timed_scorer.score, request, history, ex.metadata, gpu=True)
            timed_scorer.log.clear()
            row["scorer_calls"] = 1
            smsg = static_path(request, row)
            count_tokens(row, request, None, smsg)
            alternatives(ex, row, warmup, first_warmup)
            return row
        recs, total = timed(controller.evaluate_example, ex, gpu=True)
        for text, hist, res in timed_scorer.log:
            replay.put(text, hist, ScoreResult(res.raw_score, list(res.reasons), dict(res.metadata)))
        timed_scorer.log.clear()
        last = recs[-1]
        state, decision = last.state, last.decision
        row.update({
            "state": state, "decision": decision, "calibrated_score": last.calibrated_score,
            "configured_prompt_tokens": last.prompt_tokens,
            "scorer_ms": clock.acc["scorer"], "scorer_calls": clock.calls["scorer"],
            "features_head_platt_ms": clock.acc["calibrate"], "head_platt_ms": clock.acc["head_platt"],
            "features_ms": clock.acc["calibrate"] - clock.acc["head_platt"],
            "select_state_ms": clock.acc["select_state"], "controller_total_ms": total,
        })
        row["policy_router_ms"] = total - row["scorer_ms"] - row["features_head_platt_ms"]
        row["controller_incremental_ms"] = total - row["scorer_ms"]

        if decision == "allow":
            messages, row["prompt_assembly_ms"] = timed(
                lambda: [{"role": "system", "content": STATE_INSTRUCTIONS[state]}, {"role": "user", "content": request}])
        else:
            messages, row["prompt_assembly_ms"] = timed(lambda: None)
        row["pre_generation_path_ms"] = total + row["prompt_assembly_ms"]

        response = REFUSAL
        row["llm_called"] = bool(gen is not None and decision == "allow")
        if row["llm_called"]:
            g, row["llm_ms"] = timed(gen.generate, messages, gpu=True)
            response = g.get("text") or ""
            row["generated_tokens"] = g.get("n_new_tokens")
        row["output_guard_called"] = bool(guard is not None and row["llm_called"] and state == "hardened")
        if row["output_guard_called"]:
            clear_scorer_cache(guard)
            v, row["output_guard_ms"] = timed(guard.judge, request, response, gpu=True)
            row["output_guard_verdict"] = v.get("verdict")
            if v.get("verdict") == "unsafe":
                response = REFUSAL
        row["defense_path_ms"] = row["pre_generation_path_ms"] + row.get("llm_ms", 0.0) + row.get("output_guard_ms", 0.0)
        row["response_judge_called"] = bool(judge is not None and row["llm_called"])
        if row["response_judge_called"]:
            clear_scorer_cache(judge)
            v, row["response_judge_ms"] = timed(judge.judge, request, response, gpu=True)
            row["response_judgment"] = v.get("verdict")

        smsg = static_path(request, row)
        count_tokens(row, request, messages, smsg)
        alternatives(ex, row, warmup, first_warmup)
        return row

    for i in range(args.warmup if examples else 0):  # warm-up, discarded
        one_request(examples[i % len(examples)], -1, warmup=True, first_warmup=(i == 0))
    alt_calls_warmup = alt.calls if alt is not None else 0
    rows = [one_request(ex, rep) for rep in range(args.repeats) for ex in examples]
    n_requests = len(rows)

    incr_rows, mism = [], None
    if controller is not None and args.incremental_reps > 0:
        plain = RCDPController(scorer=replay, feature_config=controller.feature_config, config=controller.config)
        plain.calibrator = controller.calibrator._inner
        plain.trajectory_policy = controller.trajectory_policy
        for ex in examples[:5]:  # warm-up
            plain.evaluate_example(ex)
        for ex in examples:
            for k in range(args.incremental_reps):
                t0 = time.perf_counter()
                plain.evaluate_example(ex)
                incr_rows.append({"example_id": ex.example_id, "rep": k,
                                  "controller_incremental_replay_ms": (time.perf_counter() - t0) * 1000.0})
        by_id = {e.example_id: e for e in examples}
        mism = sum(1 for r in rows if plain.evaluate_example(by_id[r["example_id"]])[-1].decision != r["decision"])

    summary = {m: summarise(rows, m, B=args.bootstrap, seed=args.seed)
               for m in PRIMARY_METRICS if any(r.get(m) is not None for r in rows)}
    for m, flag in CONDITIONAL.items():
        if any(r.get(flag) for r in rows):
            summary[m] = summarise(rows, m, flag, B=args.bootstrap, seed=args.seed, n_requests=n_requests)
    if incr_rows:
        summary["controller_incremental_replay_ms"] = summarise(incr_rows, "controller_incremental_replay_ms",
                                                                B=args.bootstrap, seed=args.seed)
    by_decision, states = {}, None
    if controller is not None:
        for dec in ("block", "allow"):
            sub = [r for r in rows if r.get("decision") == dec]
            if sub:
                by_decision[dec] = {"n": len(sub), "defense_path_ms": summarise(sub, "defense_path_ms", B=0)}
        states = defaultdict(int)
        for r in rows:
            if r["rep"] == 0:
                states[f"{r['label']}/{r['state']}"] += 1
        states = dict(sorted(states.items()))

    def decisions(key):
        first = [r for r in rows if r["rep"] == 0 and r.get(key)]
        if not first:
            return None
        h = [r for r in first if r["label"] == "harmful"]
        b = [r for r in first if r["label"] == "benign"]
        return {"harmful_blocked": sum(r[key] == "block" for r in h), "n_harmful": len(h),
                "benign_allowed": sum(r[key] == "allow" for r in b), "n_benign": len(b),
                "note": "decisions on the timed subset of the first pass; rates belong to the baseline runs, "
                        "not to this latency file"}

    counts = {
        "requests": n_requests, "examples": len(examples), "repeats": args.repeats,
        "warmup_requests": args.warmup if examples else 0,
        "scorer_calls": sum(r.get("scorer_calls", 0) for r in rows),
        "llm_calls": sum(bool(r.get("llm_called")) for r in rows),
        "output_guard_calls": sum(bool(r.get("output_guard_called")) for r in rows),
        "response_judge_calls": sum(bool(r.get("response_judge_called")) for r in rows),
        "static_llm_calls": sum(bool(r.get("static_llm_called")) for r in rows),
        "lg_oneshot_requests": sum(bool(r.get("lg_oneshot_called")) for r in rows),
        "smoothllm_requests": sum(bool(r.get("smoothllm_called")) for r in rows),
        "alt_guard_calls": sum((r.get("lg_oneshot_guard_calls") or 0) + (r.get("smoothllm_guard_calls") or 0)
                               for r in rows),
        "alt_guard_calls_incl_warmup": alt.calls if alt is not None else 0,
        "alt_guard_calls_warmup": alt_calls_warmup,
    }

    out = {
        "meta": {
            "runner": "scripts/measure_latency_r1.py measure",
            "argv": E.redacted_argv(list(sys.argv[1:] if argv is None else argv)),
            "code_version": code_version([__file__, HERE / "stats_r1.py", HERE / "e2e_common.py",
                                          HERE / "scorer_baselines_r1" / "common.py",
                                          HERE / "baselines_r1" / "smoothllm.py"]),
            "started": now(), "seconds": None, "environment": environment(),
            "models": {
                "scorer": {"kind": args.scorer,
                           "name": (Path(str(args.scorer_model_name)).name if args.scorer == "llama-guard"
                                    else args.scorer_model_name),
                           "device": args.device, "dtype_arg": args.dtype, "model": model_info(raw_scorer),
                           "score_mode": getattr(args, "scorer_score_mode", "label")},
                "protected": generator_settings(gen), "output_guard": judge_settings(guard),
                "response_judge": judge_settings(judge),
                "alt_guard": None if alt is None else {
                    **judge_settings(alt_classifier),
                    "backend": args.alt_guard_backend,
                    "device": args.alt_guard_device or args.device, "dtype_arg": args.alt_guard_dtype,
                    "paths": [p for p in alt_paths if (p == "oneshot" and oneshot_gate) or
                              (p == "smoothllm" and smooth_gate)],
                    "smoothllm": None if smooth_gate is None else smooth_gate.describe().get("smoothllm"),
                    "oneshot_rule": None if oneshot_gate is None else oneshot_gate.describe().get("block_rule"),
                }},
            "dataset": args.dataset, "calibration_dataset": args.calibration_dataset,
        },
        "protocol": {
            "batch_size": 1, "concurrency": 1, "warmup_requests": args.warmup, "repeats": args.repeats,
            "warmup_alternatives": "one-shot on every warm-up request; SmoothLLM-style on the first only",
            "timer": "time.perf_counter; torch.cuda.synchronize on every used CUDA device before/after each GPU stage",
            "cache": "all scorer/guard in-memory caches cleared before every request and before every "
                     "alternative-guard call (no cache hits)",
            "setting": args.setting, "num_states": args.num_states, "scorer_only": args.scorer_only,
            "threads_arg": args.threads, "incremental_reps": args.incremental_reps,
            "alternative_repeats": args.alt_repeats,
            "alternative_subsets": {
                "oneshot": None if oneshot_gate is None else {"limit": args.oneshot_limit, "n": len(oneshot_ids),
                                                              "rule": "evenly spaced over the dataset order"},
                "smoothllm": None if smooth_gate is None else {"limit": args.smoothllm_limit, "n": len(smooth_ids),
                                                               "rule": "evenly spaced over the dataset order"},
            },
            "boundaries": BOUNDARIES,
            "conditional_stage_denominator": "n_calls = number of calls actually made; "
                                             "amortised_mean_per_request = sum over calls / all requests",
        },
        "prompts": {"state_instructions_sha256": {s: sha256_text(t) for s, t in STATE_INSTRUCTIONS.items()},
                    "static_patch_prompt": static_prompt, "static_patch_prompt_sha256": sha256_text(static_prompt),
                    "static_patch_source": args.static_patch_prompt_file or "STATE_INSTRUCTIONS['hardened'] "
                                                                            "(baselines_r1 preset 'hardened-state')"},
        "calibration": calib,
        "counts": counts,
        "routing_state_counts_rep0": states,
        "alternative_decisions_rep0": {"lg_oneshot": decisions("lg_oneshot_decision"),
                                       "smoothllm": decisions("smoothllm_decision")},
        "instrumented_vs_plain_decision_mismatches": mism,
        "summary": summary,
        "summary_by_decision": by_decision,
        "token_overhead": token_overhead(rows, counter),
        "records": rows,
        "incremental_replay_records": incr_rows,
    }
    out["meta"]["seconds"] = time.perf_counter() - t_start
    path = write_json(args.output, out)
    print(f"{n_requests} requests ({len(examples)} examples x {args.repeats}); calls: {counts}")
    for m in PRINT_ORDER:
        if m in summary:
            s = summary[m]
            ci = s.get("mean_ci95")
            calls = f" calls={s['n_calls']}/{s['n_requests']}" if "n_calls" in s else ""
            print(f"  {m:34s} n={s['n']:5d} mean={s['mean']:.4f} p50={s['p50']:.4f} p90={s['p90']:.4f} "
                  f"p99={s['p99']:.4f}" + (f" mean95=[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "") + calls)
    t = out["token_overhead"]

    def f2(x):
        return "n/a" if x is None else f"{x:.2f}"

    if "rcdp" in t:
        print(f"  hidden-prompt tokens per request: RCDP {f2(t['rcdp']['hidden_prompt_tokens_mean_all_requests'])} "
              f"(generated only {f2(t['rcdp']['hidden_prompt_tokens_mean_generated'])}), static patch "
              f"{f2(t['static_patch']['hidden_prompt_tokens_mean'])}; added templated input tokens: RCDP "
              f"{f2(t['rcdp']['added_input_tokens_mean_all_requests'])}, static patch "
              f"{f2(t['static_patch']['added_input_tokens_mean'])}")
    for name, dec in out["alternative_decisions_rep0"].items():
        if dec:
            print(f"  {name} decisions on its timed subset: harmful blocked {dec['harmful_blocked']}/{dec['n_harmful']}, "
                  f"benign allowed {dec['benign_allowed']}/{dec['n_benign']}")
    print("wrote", path)
    return out


# --------------------------------------------------------------------------
# Saved end-to-end files: per-call latencies (X-8)
# --------------------------------------------------------------------------

def _per_call(values):
    return describe([v for v in values if v is not None])


def _e2e_entry_legacy(recs, d):
    """Submitted harness (run_end_to_end_eval*.py): flat *_latency_ms fields; the judge runs on generations."""
    gen = [r for r in recs if r.get("generated")]
    guard = [r for r in recs if r.get("output_guard_checked")]
    entry = {
        "schema": "submitted (flat *_latency_ms fields)",
        "n_records": len(recs), "generation_calls": len(gen),
        "judge_calls": len(gen), "second_judge_calls": None, "output_guard_calls": len(guard),
        "judge_ms_per_call": _per_call(r.get("judge_latency_ms") for r in gen),
        "judge_ms_mean_over_all_records_as_printed": d.get("mean_judge_latency_ms"),
        "output_guard_ms_per_call": _per_call(r.get("output_guard_latency_ms") for r in guard),
        "output_guard_ms_mean_over_all_records_as_printed": d.get("mean_output_guard_latency_ms"),
        "generation_ms_per_call": _per_call(r.get("generation_latency_ms") for r in gen),
        "controller_ms_all_records": _per_call(r.get("controller_latency_ms") for r in recs),
        "end_to_end_ms_all_records": _per_call(r.get("end_to_end_latency_ms") for r in recs),
        "records_with_judge_latency_gt0": sum(1 for r in recs if (r.get("judge_latency_ms") or 0) > 0),
        "notes": [],
    }
    if entry["records_with_judge_latency_gt0"] != len(gen):
        entry["notes"].append("records with judge latency > 0 differ from generated records")
    return entry


def _e2e_entry_r1(recs, d):
    """R1 harness (e2e_common.Pipeline): nested latency_ms dict; judge results stored per record."""
    ok = [r for r in recs if "error" not in r]
    gen = [r for r in ok if r.get("generated")]
    judged = [r for r in gen if r.get("judge_primary")]
    judged2 = [r for r in gen if r.get("judge_secondary")]
    guard = [r for r in ok if (r.get("output_guard") or {}).get("checked")
             and (r.get("output_guard") or {}).get("backend") == "separate"]
    lat = (d.get("summary") or {}).get("latency") if isinstance(d.get("summary"), dict) else None
    return {
        "schema": "R1 (e2e_common.Pipeline, nested latency_ms)",
        "n_records": len(recs), "n_errors": len(recs) - len(ok), "generation_calls": len(gen),
        "judge_calls": len(judged), "second_judge_calls": len(judged2), "output_guard_calls": len(guard),
        "output_guard_reuse_primary": sum(1 for r in ok if (r.get("output_guard") or {}).get("backend")
                                          == "reuse-primary-judge"),
        "judge_ms_per_call": _per_call(r["latency_ms"].get("judge_primary") for r in judged),
        "second_judge_ms_per_call": _per_call(r["latency_ms"].get("judge_secondary") for r in judged2),
        "judge_ms_mean_over_all_records_as_printed": None,
        "output_guard_ms_per_call": _per_call(r["latency_ms"].get("output_guard") for r in guard),
        "output_guard_ms_mean_over_all_records_as_printed": None,
        "generation_ms_per_call": _per_call(r["latency_ms"].get("generation") for r in gen),
        "controller_ms_all_records": _per_call(r["latency_ms"].get("gate") for r in ok),
        "service_path_ms_all_records": _per_call(r["latency_ms"].get("service_path") for r in ok),
        "end_to_end_ms_all_records": _per_call(r["latency_ms"].get("end_to_end_incl_evaluation") for r in ok),
        "harness_latency_summary": lat,
        "notes": [],
    }


def e2e_summary(args):
    files = []
    for pat in args.inputs:
        hits = sorted(glob.glob(pat))
        if not hits and glob.has_magic(pat):
            print("no file matches %s (skipped)" % pat)
            continue
        files.extend(f for f in (hits or [pat]) if f not in files)
    out = {"meta": {"runner": "scripts/measure_latency_r1.py e2e-summary", "started": now(),
                    "denominator": "per-call statistics use only the records on which the component was called "
                                   "(ledger X-8); the submitted means divided by all records"},
           "files": []}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        recs = d.get("records") or []
        proto = d.get("protocol", {}) if isinstance(d.get("protocol"), dict) else {}
        r1 = bool(recs) and isinstance(recs[0].get("latency_ms"), dict)
        entry = _e2e_entry_r1(recs, d) if r1 else _e2e_entry_legacy(recs, d)
        entry.update({
            "file": Path(f).name,
            "protected_model": Path(str(proto.get("protected_model") or "")).name or None,
            "inference_backend": proto.get("inference_backend"),
            "controller_scorer_model": Path(str(proto.get("controller_scorer_model") or "")).name or None,
            "controller_scorer_cache_path_set": bool(proto.get("controller_scorer_cache_path")),
        })
        if entry["controller_scorer_cache_path_set"]:
            entry["notes"].append("a scorer cache file was set: controller latencies may be cache hits")
        out["files"].append(entry)
        j, g, c = entry["judge_ms_per_call"], entry["output_guard_ms_per_call"], entry["controller_ms_all_records"]
        printed = entry["judge_ms_mean_over_all_records_as_printed"]

        def f2(x):
            return "n/a" if x is None else f"{x:.2f}"

        print(f"{entry['file']}: judge {entry['judge_calls']} calls, mean/call {f2(j['mean'])} ms "
              f"(printed mean over {len(recs)} records: {f2(printed)}); output guard {entry['output_guard_calls']} "
              f"calls, mean/call {f2(g['mean'])} ms; controller mean {f2(c['mean'])} ms"
              + (" [scorer cache file set]" if entry["controller_scorer_cache_path_set"] else ""))
    path = write_json(args.output, out)
    print("wrote", path)
    return out


def main(argv=None, scorer=None, tokenizer=None, alt_classifier=None):
    args = parse_args(argv)
    if args.cmd == "measure":
        return measure(args, scorer=scorer, tokenizer=tokenizer, alt_classifier=alt_classifier, argv=argv)
    return e2e_summary(args)


if __name__ == "__main__":
    main()
