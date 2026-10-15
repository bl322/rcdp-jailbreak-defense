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
from rcdp_guard.datasets import load_examples
from rcdp_guard.evaluation import fit_calibrator, summarize, write_summary
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen RCDP calibration on an external benign benchmark."
    )
    parser.add_argument(
        "--calibration-dataset",
        required=True,
        help="JSONL containing only the calibration portion of the original benchmark pool.",
    )
    parser.add_argument("--external-dataset", required=True, help="External benign JSONL, e.g. XSTest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", default="distilbart-mnli", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    parser.add_argument("--benchmark-aware", action="store_true")
    parser.add_argument("--use-hazard-latch", action="store_true")
    parser.add_argument("--enable-transformation-views", action="store_true")
    parser.add_argument("--transformation-max-views", type=int, default=3)
    parser.add_argument("--transformation-min-confidence", type=float, default=0.80)
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
        config=ControllerConfig(
            benchmark_aware=args.benchmark_aware,
            use_hazard_latch=args.use_hazard_latch,
            use_transformation_views=args.enable_transformation_views,
            transformation_max_views=args.transformation_max_views,
            transformation_min_confidence=args.transformation_min_confidence,
        ),
    )

    # Fit exclusively on the original calibration partition. The external suite is never
    # used to learn the calibrator or choose an operating point.
    fit_calibrator(controller, load_examples(args.calibration_dataset))
    records = []
    for example in load_examples(args.external_dataset):
        records.extend(controller.evaluate_example(example))

    summary = summarize(records)
    summary["protocol"] = {
        "evaluation": "zero-shot external benign generalization",
        "calibration_dataset": args.calibration_dataset,
        "external_dataset": args.external_dataset,
        "benchmark_aware": args.benchmark_aware,
        "use_transformation_views": args.enable_transformation_views,
        "transformation_max_views": args.transformation_max_views,
        "transformation_min_confidence": args.transformation_min_confidence,
        "scorer": args.scorer,
        "scorer_model_name": args.scorer_model_name if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
        "scorer_cache_path": args.scorer_cache_path if args.scorer in {"distilbart-mnli", "llama-guard"} else None,
    }
    write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
