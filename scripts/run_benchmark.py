from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.calibration import FeatureConfig
from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.evaluation import evaluate_dataset, write_summary
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-states", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--benchmark-agnostic", action="store_true")
    parser.add_argument("--disable-uncertainty", action="store_true")
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--enable-downgrade", action="store_true")
    parser.add_argument("--drop-feature-group", action="append", default=[])
    parser.add_argument("--scorer", default="keyword", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_groups = {"raw", "uncertainty", "length", "lexical", "history", "benchmark"}
    feature_groups -= set(args.drop_feature_group)
    feature_config = FeatureConfig(enabled_groups=feature_groups)

    if args.scorer == "distilbart-mnli":
        scorer = DistilBartMNLIRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
        )
    elif args.scorer == "llama-guard":
        scorer = LlamaGuardRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
        )
    else:
        scorer = KeywordRiskScorer()

    controller = RCDPController(
        scorer=scorer,
        feature_config=feature_config,
        config=ControllerConfig(
            num_states=args.num_states,
            allow_state_downgrade=args.enable_downgrade,
            benchmark_aware=not args.benchmark_agnostic,
            use_uncertainty_gate=not args.disable_uncertainty,
            use_history=not args.disable_history,
        ),
    )
    summary = evaluate_dataset(args.dataset, controller)
    summary["config"] = {
        "num_states": args.num_states,
        "benchmark_aware": not args.benchmark_agnostic,
        "use_uncertainty_gate": not args.disable_uncertainty,
        "use_history": not args.disable_history,
        "allow_state_downgrade": args.enable_downgrade,
        "enabled_feature_groups": sorted(feature_groups),
        "scorer": args.scorer,
        "scorer_model_name": args.scorer_model_name if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
        "scorer_local_files_only": args.scorer_local_files_only if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
        "scorer_device": args.scorer_device if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
        "scorer_cache_path": args.scorer_cache_path if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
    }
    write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
