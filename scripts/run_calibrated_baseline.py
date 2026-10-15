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
from rcdp_guard.evaluation import fit_calibrator
from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a calibrated one-threshold scorer baseline on a held-out split."
    )
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--scorer", default="llama-guard", choices=["keyword", "distilbart-mnli", "llama-guard"])
    parser.add_argument("--scorer-model-name", default="meta-llama/Llama-Guard-3-1B")
    parser.add_argument("--scorer-local-files-only", action="store_true")
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--scorer-cache-path", default=None)
    parser.add_argument("--benchmark-aware", action="store_true")
    return parser.parse_args()


def build_scorer(args: argparse.Namespace):
    if args.scorer == "keyword":
        return KeywordRiskScorer()
    if args.scorer == "distilbart-mnli":
        return DistilBartMNLIRiskScorer(
            model_name=args.scorer_model_name,
            local_files_only=args.scorer_local_files_only,
            device=args.scorer_device,
            cache_path=args.scorer_cache_path,
        )
    return LlamaGuardRiskScorer(
        model_name=args.scorer_model_name,
        local_files_only=args.scorer_local_files_only,
        device=args.scorer_device,
        cache_path=args.scorer_cache_path,
    )


def calibrated_probabilities(controller: RCDPController, examples) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for example in examples:
        history: list[str] = []
        for turn in example.turns:
            if turn.role != "user":
                history.append(turn.text)
                continue
            score = controller.scorer.score(turn.text, history, example.metadata)
            calibrated = controller.calibrate(
                raw_score=score.raw_score,
                text=turn.text,
                benchmark=example.benchmark,
                prior_high_risk_turns=0,
                prior_cautious_turns=0,
            )
            rows.append((calibrated.probability, example.label.value))
            history.append(turn.text)
    return rows


def choose_threshold(rows: list[tuple[float, str]], alpha: float) -> float:
    candidates = sorted({0.0, 1.0, *(score for score, _ in rows)})
    best_threshold = candidates[0]
    best_value = float("-inf")
    for threshold in candidates:
        harmful = [label for score, label in rows if label == "harmful"]
        benign = [label for score, label in rows if label == "benign"]
        block_rate = sum(score >= threshold for score, label in rows if label == "harmful") / max(len(harmful), 1)
        fpr = sum(score >= threshold for score, label in rows if label == "benign") / max(len(benign), 1)
        value = block_rate - alpha * fpr
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return best_threshold


def evaluate(controller: RCDPController, examples, threshold: float) -> dict:
    scores: list[tuple[float, str]] = []
    for example in examples:
        history: list[str] = []
        for turn in example.turns:
            if turn.role != "user":
                history.append(turn.text)
                continue
            score = controller.scorer.score(turn.text, history, example.metadata)
            calibrated = controller.calibrate(
                raw_score=score.raw_score,
                text=turn.text,
                benchmark=example.benchmark,
                prior_high_risk_turns=0,
                prior_cautious_turns=0,
            )
            scores.append((calibrated.probability, example.label.value))
            history.append(turn.text)

    harmful = [score for score, label in scores if label == "harmful"]
    benign = [score for score, label in scores if label == "benign"]
    block_rate = sum(score >= threshold for score in harmful) / max(len(harmful), 1)
    allow_rate = sum(score < threshold for score in benign) / max(len(benign), 1)
    return {
        "num_examples": len(scores),
        "harmful_examples": len(harmful),
        "benign_examples": len(benign),
        "threshold": threshold,
        "attack_block_rate": block_rate,
        "benign_allow_rate": allow_rate,
        "false_positive_rate": 1.0 - allow_rate,
    }


def main() -> None:
    args = parse_args()
    feature_config = FeatureConfig.all()
    controller = RCDPController(
        scorer=build_scorer(args),
        feature_config=feature_config,
        config=ControllerConfig(
            num_states=2,
            benchmark_aware=args.benchmark_aware,
            use_uncertainty_gate=False,
            use_history=False,
        ),
    )
    calibration_examples = load_examples(args.calibration_dataset)
    evaluation_examples = load_examples(args.dataset)
    fit_calibrator(controller, calibration_examples)
    threshold = choose_threshold(calibrated_probabilities(controller, calibration_examples), args.alpha)
    result = evaluate(controller, evaluation_examples, threshold)
    result["protocol"] = {
        "baseline": "calibrated scorer with one global threshold",
        "calibration_dataset": args.calibration_dataset,
        "evaluation_dataset": args.dataset,
        "alpha": args.alpha,
        "benchmark_aware": args.benchmark_aware,
        "scorer": args.scorer,
        "scorer_model_name": args.scorer_model_name if args.scorer != "keyword" else None,
        "scorer_cache_path": args.scorer_cache_path,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
