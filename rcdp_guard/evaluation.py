from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .controller import ControllerConfig, RCDPController
from .datasets import load_examples
from .trajectory_policy import fit_trajectory_policy
from .types import EvalRecord, Label


def stable_split(examples, ratio: float = 0.5):
    cal, test = [], []
    for ex in examples:
        bucket = sum(ord(c) for c in ex.example_id) % 100
        (cal if bucket < ratio * 100 else test).append(ex)
    return cal, test


def fit_calibrator(controller: RCDPController, examples) -> None:
    rows = []
    labels = []
    for ex in examples:
        history: list[str] = []
        for turn in ex.turns:
            if turn.role != "user":
                history.append(turn.text)
                continue
            score_result = controller.scorer.score(turn.text, history, ex.metadata)
            features = controller.calibrate(
                raw_score=score_result.raw_score,
                text=turn.text,
                benchmark=ex.benchmark,
                prior_high_risk_turns=0,
                prior_cautious_turns=0,
            ).features
            rows.append(features)
            labels.append(1 if ex.label == Label.HARMFUL else 0)
            history.append(turn.text)
    controller.calibrator.fit(rows, labels)


def evaluate_dataset(dataset_path: str, controller: RCDPController) -> dict:
    examples = load_examples(dataset_path)
    cal_examples, test_examples = stable_split(examples)
    return evaluate_with_split(cal_examples, test_examples, controller)


def evaluate_with_split(
    calibration_path: str,
    evaluation_path: str,
    controller: RCDPController,
) -> dict:
    """Evaluate using the paper's precomputed calibration/evaluation split."""
    cal_examples = load_examples(calibration_path)
    test_examples = load_examples(evaluation_path)
    return evaluate_with_split_examples(cal_examples, test_examples, controller)


def evaluate_with_split_examples(cal_examples, test_examples, controller: RCDPController) -> dict:
    fit_calibrator(controller, cal_examples)
    if controller.config.use_learned_transition_policy:
        controller.trajectory_policy = fit_trajectory_policy(controller, cal_examples)

    records: list[EvalRecord] = []
    for ex in test_examples:
        records.extend(controller.evaluate_example(ex))
    return summarize(records)


def summarize(records: list[EvalRecord]) -> dict:
    harmful = [r for r in records if r.label == "harmful"]
    benign = [r for r in records if r.label == "benign"]
    blocked_harmful = sum(r.decision == "block" for r in harmful)
    allowed_benign = sum(r.decision == "allow" for r in benign)
    avg_tokens = sum(r.prompt_tokens for r in records) / max(len(records), 1)
    state_counts: dict[str, int] = {}
    for r in records:
        state_counts[r.state] = state_counts.get(r.state, 0) + 1
    state_counts_harmful: dict[str, int] = {}
    state_counts_benign: dict[str, int] = {}
    for r in harmful:
        state_counts_harmful[r.state] = state_counts_harmful.get(r.state, 0) + 1
    for r in benign:
        state_counts_benign[r.state] = state_counts_benign.get(r.state, 0) + 1

    labels = [1 if r.label == "harmful" else 0 for r in records]
    scores = [r.calibrated_score for r in records]
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / max(len(records), 1)
    harmful_scores = [r.calibrated_score for r in harmful]
    benign_scores = [r.calibrated_score for r in benign]
    # A threshold-free diagnostic: how often a harmful item is ranked above
    # a benign item. It remains informative when all endpoint decisions agree.
    pair_count = len(harmful_scores) * len(benign_scores)
    concordant = sum(h > b for h in harmful_scores for b in benign_scores)
    tied = sum(h == b for h in harmful_scores for b in benign_scores)
    ranking_auc = (concordant + 0.5 * tied) / pair_count if pair_count else float("nan")
    mean_tokens_harmful = (
        sum(r.prompt_tokens for r in harmful) / max(len(harmful), 1)
    )
    mean_tokens_benign = (
        sum(r.prompt_tokens for r in benign) / max(len(benign), 1)
    )

    return {
        "num_records": len(records),
        "attack_block_rate": blocked_harmful / max(len(harmful), 1),
        "benign_allow_rate": allowed_benign / max(len(benign), 1),
        "false_positive_rate": 1.0 - (allowed_benign / max(len(benign), 1) if benign else 0.0),
        "mean_prompt_tokens": avg_tokens,
        "state_counts": state_counts,
        "state_counts_harmful": state_counts_harmful,
        "state_counts_benign": state_counts_benign,
        "mean_prompt_tokens_harmful": mean_tokens_harmful,
        "mean_prompt_tokens_benign": mean_tokens_benign,
        "mean_calibrated_harmful": sum(harmful_scores) / max(len(harmful_scores), 1),
        "mean_calibrated_benign": sum(benign_scores) / max(len(benign_scores), 1),
        "calibrated_score_gap": (
            sum(harmful_scores) / max(len(harmful_scores), 1)
            - sum(benign_scores) / max(len(benign_scores), 1)
        ),
        "brier_score": brier,
        "ranking_auc": ranking_auc,
        "records": [asdict(r) for r in records],
    }


def write_summary(summary: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
