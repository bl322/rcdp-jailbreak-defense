"""Re-run the adaptive attacks AND the Table-5 RCDP rows against the CURRENT
reproducible controller, so every number in the manuscript is generated from the
committed code + data.

Findings this resolves:
  * The manuscript's "105" denominator (adaptive table) and ABRctrl 0.7047
    (Table 5) come from an OLD controller snapshot.  The current controller,
    under the identical calibration split + DistilBART scorer, blocks 117/149
    when benchmark-agnostic (BA=False) and 135/149 when benchmark-aware
    (BA=True).  So we regenerate the adaptive rows on the 117-blocking
    controller and the Table-5 rows at their reproducible rates.
  * The script is the single source of truth: no number is hand-typed.
"""
from __future__ import annotations

import os
import json, sys, time, argparse
from pathlib import Path

EXPERIMENTS = Path(os.environ.get("RCDP_EXPERIMENTS", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(EXPERIMENTS))

from rcdp_guard.calibration import FeatureConfig
from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.datasets import load_examples
from rcdp_guard.evaluation import fit_calibrator
from rcdp_guard.scorers import DistilBartMNLIRiskScorer
from rcdp_guard import adaptive as ADAPT

HELDOUT149 = EXPERIMENTS / "data" / "paper_splits" / "heldout_harmful_149.jsonl"
HELDOUT204 = EXPERIMENTS / "data" / "paper_splits" / "heldout_204.jsonl"
CALIBRATION = EXPERIMENTS / "data" / "paper_splits" / "full_pool_calibration.jsonl"


def build(benchmark_aware):
    scorer = DistilBartMNLIRiskScorer(
        model_name="valhalla/distilbart-mnli-12-3", local_files_only=True, device="cpu")
    c = RCDPController(scorer=scorer, feature_config=FeatureConfig.all(),
                       config=ControllerConfig(benchmark_aware=benchmark_aware))
    fit_calibrator(c, load_examples(str(CALIBRATION)))
    return c


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base-url", default=os.environ.get("RCDP_API_BASE", "http://localhost:30001/v1"))
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--attacker-model", default="qwen3-next-80b-a3b-instruct")
    ap.add_argument("--judge-model", default="qwen3-next-80b-a3b-instruct")
    ap.add_argument("--num-candidates", type=int, default=2)
    ap.add_argument("--output", default="results/adaptive_and_table5_current.json")
    args = ap.parse_args()

    report = {"protocol": {"heldout149": str(HELDOUT149), "calibration": str(CALIBRATION),
                          "scorer": "distilbart-mnli (current code)"}}
    t0 = time.time()

    # --- Table 5 RCDP rows: both benchmark settings, measured on heldout_204 ---
    for ba in (False, True):
        c = build(ba)
        m = measure_controller(c, HELDOUT204)
        report[f"table5_ba{ba}"] = m
        print(f"[=] BA={ba}: ABRctrl={m['abrctrl']:.4f} ({m['harmful_blocked']}/{m['harmful_n']}) "
              f"BAR={m['bar']:.4f} FPR={m['fpr']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # --- Adaptive table rows: benchmark-agnostic controller (BA=False, blocks 117) ---
    c_agn = build(False)
    report["adaptive_baFalse"] = {
        "heuristic": ADAPT.run_simple_adaptive_eval(str(HELDOUT149), c_agn),
        "two_stage": ADAPT.run_multiturn_state_manipulation_eval(str(HELDOUT149), c_agn),
    }
    print(f"[=] local adaptive (heuristic/two-stage) done ({time.time()-t0:.0f}s)", flush=True)
    report["adaptive_baFalse"]["api_rewrite"] = ADAPT.run_api_rewrite_eval(
        str(HELDOUT149), c_agn, api_base_url=args.api_base_url, api_key=args.api_key,
        api_model=args.attacker_model, judge_api_model=args.judge_model,
        num_candidates=args.num_candidates, preservation_mode="strict")
    print(f"[=] api rewrite done ({time.time()-t0:.0f}s)", flush=True)
    report["adaptive_baFalse"]["api_two_stage"] = ADAPT.run_api_multiturn_state_manipulation_eval(
        str(HELDOUT149), c_agn, api_base_url=args.api_base_url, api_key=args.api_key,
        api_model=args.attacker_model, judge_api_model=args.judge_model,
        preservation_mode="strict")
    print(f"[=] api two-stage done ({time.time()-t0:.0f}s)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
