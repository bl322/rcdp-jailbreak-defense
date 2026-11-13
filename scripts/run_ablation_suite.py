from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcdp_guard.calibration import FeatureConfig
from rcdp_guard.controller import ControllerConfig, RCDPController
from rcdp_guard.evaluation import evaluate_dataset, evaluate_with_split
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--calibration-dataset", default=None)
    parser.add_argument("--evaluation-dataset", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer", default="keyword", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    parser.add_argument("--scorer-cache-only", action="store_true")
    args = parser.parse_args()
    if bool(args.calibration_dataset) != bool(args.evaluation_dataset):
        parser.error("--calibration-dataset and --evaluation-dataset must be supplied together")
    return args


def build_scorer(args: argparse.Namespace):
    if args.scorer == "distilbart-mnli":
        return DistilBartMNLIRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
            cache_only=args.scorer_cache_only,
        )
    if args.scorer == "llama-guard":
        return LlamaGuardRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
        )
    return KeywordRiskScorer()


def build_controller(
    *,
    scorer,
    num_states: int = 4,
    benchmark_aware: bool = True,
    use_uncertainty_gate: bool = True,
    use_history: bool = True,
    allow_state_downgrade: bool = False,
    feature_groups: set[str] | None = None,
) -> RCDPController:
    return RCDPController(
        scorer=scorer,
        feature_config=FeatureConfig(enabled_groups=feature_groups or {"raw", "uncertainty", "length", "lexical", "history", "benchmark"}),
        config=ControllerConfig(
            num_states=num_states,
            benchmark_aware=benchmark_aware,
            use_uncertainty_gate=use_uncertainty_gate,
            use_history=use_history,
            allow_state_downgrade=allow_state_downgrade,
        ),
    )


def build_single_threshold_controller(*, scorer, benchmark_aware: bool = False) -> RCDPController:
    """Use the same calibrated risk interface without state/history routing."""
    return build_controller(
        scorer=scorer,
        num_states=2,
        benchmark_aware=benchmark_aware,
        use_uncertainty_gate=False,
        use_history=False,
    )


def main() -> None:
    args = parse_args()
    scorer = build_scorer(args)

    rows = []
    detailed_results = []
    suites = [
        ("full-4state", build_controller(scorer=scorer)),
        ("2-state", build_controller(scorer=scorer, num_states=2)),
        ("3-state", build_controller(scorer=scorer, num_states=3)),
        ("benchmark-agnostic", build_controller(scorer=scorer, benchmark_aware=False)),
        ("no-uncertainty-gate", build_controller(scorer=scorer, use_uncertainty_gate=False)),
        ("no-history", build_controller(scorer=scorer, use_history=False)),
        ("downgrade-enabled", build_controller(scorer=scorer, allow_state_downgrade=True)),
        ("learned-transition-policy", build_controller(
            scorer=scorer,
            allow_state_downgrade=True,
        )),
        ("drop-uncertainty-feature", build_controller(
            scorer=scorer,
            feature_groups={"raw", "length", "lexical", "history", "benchmark"},
        )),
        ("drop-length-lexical-features", build_controller(
            scorer=scorer,
            feature_groups={"raw", "uncertainty", "history", "benchmark"},
        )),
        ("drop-benchmark-feature", build_controller(scorer=scorer, feature_groups={"raw", "uncertainty", "length", "lexical", "history"})),
        ("drop-history-feature", build_controller(scorer=scorer, feature_groups={"raw", "uncertainty", "length", "lexical", "benchmark"})),
        ("drop-lexical-feature", build_controller(scorer=scorer, feature_groups={"raw", "uncertainty", "length", "history", "benchmark"})),
        ("risk-memory-routing", build_controller(
            scorer=scorer,
            allow_state_downgrade=True,
        )),
    ]
    # Keep the proposed continuous trajectory-memory branch separate from the
    # historical counter-based controller so its incremental effect is visible.
    for name, controller in suites:
        if name == "risk-memory-routing":
            controller.config.use_risk_memory = True
        if name == "learned-transition-policy":
            controller.config.use_learned_transition_policy = True

    for name, controller in suites:
        if args.calibration_dataset and args.evaluation_dataset:
            summary = evaluate_with_split(
                args.calibration_dataset,
                args.evaluation_dataset,
                controller,
            )
        else:
            summary = evaluate_dataset(args.dataset, controller)
        detailed_results.append(
            {
                "setting": name,
                "summary": {key: value for key, value in summary.items() if key != "records"},
                "records": summary["records"],
            }
        )
        rows.append(
            {
                "setting": name,
                "num_records": summary["num_records"],
                "num_harmful": sum(r["label"] == "harmful" for r in summary["records"]),
                "num_benign": sum(r["label"] == "benign" for r in summary["records"]),
                "blocked_harmful": sum(
                    r["label"] == "harmful" and r["decision"] == "block"
                    for r in summary["records"]
                ),
                "allowed_benign": sum(
                    r["label"] == "benign" and r["decision"] == "allow"
                    for r in summary["records"]
                ),
                "attack_block_rate": summary["attack_block_rate"],
                "benign_allow_rate": summary["benign_allow_rate"],
                "false_positive_rate": summary["false_positive_rate"],
                "mean_prompt_tokens": summary["mean_prompt_tokens"],
                "mean_calibrated_harmful": summary["mean_calibrated_harmful"],
                "mean_calibrated_benign": summary["mean_calibrated_benign"],
                "calibrated_score_gap": summary["calibrated_score_gap"],
                "brier_score": summary["brier_score"],
                "ranking_auc": summary["ranking_auc"],
                "state_counts": str(summary["state_counts"]),
                "state_counts_harmful": str(summary["state_counts_harmful"]),
                "state_counts_benign": str(summary["state_counts_benign"]),
                "mean_prompt_tokens_harmful": summary["mean_prompt_tokens_harmful"],
                "mean_prompt_tokens_benign": summary["mean_prompt_tokens_benign"],
                "scorer": args.scorer,
                "scorer_model_name": args.scorer_model_name if args.scorer in {"distilbart-mnli", "llama-guard"} else "",
                "scorer_local_files_only": args.scorer_local_files_only if args.scorer in {"distilbart-mnli", "llama-guard"} else "",
                "scorer_device": args.scorer_device if args.scorer in {"distilbart-mnli", "llama-guard"} else "",
                "scorer_cache_path": args.scorer_cache_path if args.scorer in {"distilbart-mnli", "llama-guard"} else "",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "setting",
                "num_records",
                "num_harmful",
                "num_benign",
                "blocked_harmful",
                "allowed_benign",
                "attack_block_rate",
                "benign_allow_rate",
                "false_positive_rate",
                "mean_prompt_tokens",
                "mean_calibrated_harmful",
                "mean_calibrated_benign",
                "calibrated_score_gap",
                "brier_score",
                "ranking_auc",
                "state_counts",
                "state_counts_harmful",
                "state_counts_benign",
                "mean_prompt_tokens_harmful",
                "mean_prompt_tokens_benign",
                "scorer",
                "scorer_model_name",
                "scorer_local_files_only",
                "scorer_device",
                "scorer_cache_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Keep the per-example records alongside the compact CSV. This makes
    # identical endpoint rates auditable rather than forcing interpretation
    # from aggregate counts alone.
    detail_output = output.with_suffix(".json")
    detail_output.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "calibration_dataset": args.calibration_dataset,
                "evaluation_dataset": args.evaluation_dataset,
                "scorer": args.scorer,
                "variants": detailed_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
