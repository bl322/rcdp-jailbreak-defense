"""Re-run the adaptive attacks AND the Table-5 RCDP rows against the CURRENT
reproducible controller, so every number in the manuscript is generated from the
committed code + data.

Revision R1:
  * calibration follows ``--calibration-protocol`` (default: $RCDP_CALIBRATION_PROTOCOL
    or r1, the paper's three-split protocol); ``legacy`` reproduces the submitted
    file ``results/adaptive_and_table5_current.json``. The fitted operating
    point of every controller is written to the output.
  * the adaptive rows now save every rewrite (``details``) and, for the API
    rows, every candidate with its raw preservation-judge reply and the exact
    prompts/generation settings (``protocol``), so semantic preservation can be
    measured afterwards with ``tap_semantic_preservation.py``.
  * ``--skip-api`` runs the deterministic rows only (no API key needed).
  * one scorer instance is shared by all controllers (scores are identical;
    the scorer is only loaded once).

Submitted findings this script resolved: the manuscript's "105" denominator
came from an old controller snapshot; the submitted controller blocks 117/149
when benchmark-agnostic. The revision denominator is whatever this run measures.
"""
from __future__ import annotations

import os
import json, sys, time, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for _p in (str(HERE), str(CODE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", CODE_ROOT))

from rcdp_guard.calibration import FeatureConfig
from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.datasets import load_examples
from rcdp_guard.evaluation import calibration_protocol_from_env, fit_calibrator
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer
from rcdp_guard import adaptive as ADAPT
from rcdp_hardened import calibration_summary

HELDOUT149 = EXPERIMENTS / "data" / "paper_splits" / "heldout_harmful_149.jsonl"
HELDOUT204 = EXPERIMENTS / "data" / "paper_splits" / "heldout_204.jsonl"
CALIBRATION = EXPERIMENTS / "data" / "paper_splits" / "full_pool_calibration.jsonl"


def make_scorer(kind, device="cpu"):
    if kind == "keyword":
        return KeywordRiskScorer()
    return DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3", local_files_only=True, device=device)


def build(benchmark_aware, scorer, calibration, protocol):
    c = RCDPController(scorer=scorer, feature_config=FeatureConfig.all(),
                       config=ControllerConfig(benchmark_aware=benchmark_aware))
    fit_calibrator(c, load_examples(str(calibration)), protocol=protocol)
    return c


def _cal(controller):
    s = calibration_summary(controller)
    s.pop("controller_config", None)
    return s


def measure_controller(controller, path):
    exs = load_examples(str(path))
    blocked_h = allowed_b = blocked_b = 0
    n_h = n_b = 0
    for ex in exs:
        recs = controller.evaluate_example(ex)
        if not recs:
            continue
        last = recs[-1]
        if ex.label == "harmful":
            n_h += 1
            if last.decision == "block":
                blocked_h += 1
        elif ex.label == "benign":
            n_b += 1
            if last.decision == "allow":
                allowed_b += 1
            else:
                blocked_b += 1
    return {
        "harmful_n": n_h, "harmful_blocked": blocked_h,
        "abrctrl": blocked_h / max(n_h, 1),
        "benign_n": n_b, "benign_allowed": allowed_b, "benign_blocked": blocked_b,
        "bar": allowed_b / max(n_b, 1),
        "fpr": blocked_b / max(n_b, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base-url", default=os.environ.get("RCDP_API_BASE", "http://localhost:30001/v1"))
    ap.add_argument("--api-key", default=None, help="default $RCDP_API_KEY; not needed with --skip-api")
    ap.add_argument("--attacker-model", default="qwen3-next-80b-a3b-instruct")
    ap.add_argument("--judge-model", default="qwen3-next-80b-a3b-instruct",
                    help="preservation filter used *inside* the attack (submitted: the attacker model)")
    ap.add_argument("--num-candidates", type=int, default=2)
    ap.add_argument("--calibration-protocol", default=None, help="r1 (default), r1-no-platt, raw-threshold, legacy")
    ap.add_argument("--skip-api", action="store_true", help="deterministic rows only")
    ap.add_argument("--skip-table5", action="store_true")
    ap.add_argument("--scorer", choices=["distilbart-mnli", "keyword"], default="distilbart-mnli",
                    help="keyword = offline smoke tests only")
    ap.add_argument("--scorer-device", default="cpu")
    ap.add_argument("--heldout149", default=str(HELDOUT149))
    ap.add_argument("--heldout204", default=str(HELDOUT204))
    ap.add_argument("--calibration", default=str(CALIBRATION))
    ap.add_argument("--output", default="results/adaptive_and_table5_current.json")
    args = ap.parse_args(argv)
    protocol = (args.calibration_protocol or calibration_protocol_from_env()).strip().lower()
    api_key = args.api_key or os.environ.get("RCDP_API_KEY")
    if not args.skip_api and not api_key:
        raise SystemExit("No API key (--api-key or RCDP_API_KEY); use --skip-api for the deterministic rows.")

    report = {"protocol": {"heldout149": str(args.heldout149), "heldout204": str(args.heldout204),
                           "calibration": str(args.calibration), "calibration_protocol": protocol,
                           "scorer": args.scorer if args.scorer == "keyword"
                           else "distilbart-mnli (valhalla/distilbart-mnli-12-3, %s)" % args.scorer_device}}
    t0 = time.time()
    scorer = make_scorer(args.scorer, args.scorer_device)

    # --- Table 5 RCDP rows: both benchmark settings, measured on heldout_204 ---
    if not args.skip_table5:
        for ba in (False, True):
            c = build(ba, scorer, args.calibration, protocol)
            m = measure_controller(c, args.heldout204)
            m["calibration"] = _cal(c)
            report[f"table5_ba{ba}"] = m
            print(f"[=] BA={ba}: ABRctrl={m['abrctrl']:.4f} ({m['harmful_blocked']}/{m['harmful_n']}) "
                  f"BAR={m['bar']:.4f} FPR={m['fpr']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # --- Adaptive table rows: benchmark-agnostic controller ---
    c_agn = build(False, scorer, args.calibration, protocol)
    report["adaptive_baFalse"] = {
        "calibration": _cal(c_agn),
        "heuristic": ADAPT.run_simple_adaptive_eval(str(args.heldout149), c_agn),
        "two_stage": ADAPT.run_multiturn_state_manipulation_eval(str(args.heldout149), c_agn),
    }
    print(f"[=] local adaptive (heuristic/two-stage) done ({time.time()-t0:.0f}s): "
          f"{report['adaptive_baFalse']['heuristic']['newly_evaded']}/"
          f"{report['adaptive_baFalse']['heuristic']['originally_blocked']}, "
          f"{report['adaptive_baFalse']['two_stage']['newly_evaded']}/"
          f"{report['adaptive_baFalse']['two_stage']['originally_blocked']}", flush=True)
    if not args.skip_api:
        report["adaptive_baFalse"]["api_rewrite"] = ADAPT.run_api_rewrite_eval(
            str(args.heldout149), c_agn, api_base_url=args.api_base_url, api_key=api_key,
            api_model=args.attacker_model, judge_api_model=args.judge_model,
            num_candidates=args.num_candidates, preservation_mode="strict")
        print(f"[=] api rewrite done ({time.time()-t0:.0f}s)", flush=True)
        report["adaptive_baFalse"]["api_two_stage"] = ADAPT.run_api_multiturn_state_manipulation_eval(
            str(args.heldout149), c_agn, api_base_url=args.api_base_url, api_key=api_key,
            api_model=args.attacker_model, judge_api_model=args.judge_model,
            preservation_mode="strict")
        print(f"[=] api two-stage done ({time.time()-t0:.0f}s)", flush=True)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
