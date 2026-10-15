from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.adaptive import (
    run_api_multiturn_state_manipulation_eval,
    run_api_rewrite_eval,
    run_multiturn_state_manipulation_eval,
    run_simple_adaptive_eval,
)
from rcdp_guard.calibration import FeatureConfig
from rcdp_guard.controller import RCDPController
from rcdp_guard.datasets import load_examples
from rcdp_guard.evaluation import fit_calibrator
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", default="keyword", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    parser.add_argument("--attacker-api-base-url", default=None)
    parser.add_argument("--attacker-api-model", default=None)
    parser.add_argument("--attacker-api-key", default=None)
    parser.add_argument("--judge-api-model", default=None)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--preservation-mode", default="strict", choices=["strict", "relaxed", "none"])
    return parser.parse_args()


def build_scorer(args: argparse.Namespace):
    if args.scorer == "keyword":
        return KeywordRiskScorer()
    if args.scorer == "llama-guard":
        return LlamaGuardRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
        )
    return DistilBartMNLIRiskScorer(
        model_name=args.scorer_model_name,
        local_files_only=args.scorer_local_files_only,
        device=args.scorer_device,
        cache_path=args.scorer_cache_path,
    )


def main() -> None:
    args = parse_args()

    controller = RCDPController(
        scorer=build_scorer(args),
        feature_config=FeatureConfig.all(),
    )
    fit_calibrator(controller, load_examples(args.calibration_dataset))

    result = {
        "protocol": {
            "dataset": args.dataset,
            "calibration_dataset": args.calibration_dataset,
            "scorer": args.scorer,
            "scorer_model_name": args.scorer_model_name if args.scorer != "keyword" else None,
            "scorer_cache_path": args.scorer_cache_path,
        },
        "rewrite_heuristic": run_simple_adaptive_eval(args.dataset, controller),
        "multiturn_state_manipulation_heuristic": run_multiturn_state_manipulation_eval(args.dataset, controller),
    }

    api_key = args.attacker_api_key or os.environ.get("OPENAI_API_KEY")
    if args.attacker_api_base_url and args.attacker_api_model and api_key:
        result["protocol"]["attacker_api_base_url"] = args.attacker_api_base_url
        result["protocol"]["attacker_api_model"] = args.attacker_api_model
        result["protocol"]["judge_api_model"] = args.judge_api_model or args.attacker_api_model
        result["protocol"]["num_candidates"] = args.num_candidates
        result["protocol"]["preservation_mode"] = args.preservation_mode
        result["rewrite_api"] = run_api_rewrite_eval(
            args.dataset,
            controller,
            api_base_url=args.attacker_api_base_url,
            api_key=api_key,
            api_model=args.attacker_api_model,
            judge_api_model=args.judge_api_model,
            num_candidates=args.num_candidates,
            preservation_mode=args.preservation_mode,
        )
        result["multiturn_state_manipulation_api"] = run_api_multiturn_state_manipulation_eval(
            args.dataset,
            controller,
            api_base_url=args.attacker_api_base_url,
            api_key=api_key,
            api_model=args.attacker_api_model,
            judge_api_model=args.judge_api_model,
            preservation_mode=args.preservation_mode,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
