from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .trajectory_features import build_transition_features


FEATURE_NAMES: tuple[str, ...] = (
    "raw_score",
    "calibrated_score",
    "raw_logit",
    "uncertainty",
    "raw_uncertainty",
    "dialogue_progress",
    "risk_memory",
    "prev_risk_level",
    "safe_streak",
    "risky_streak",
    "prior_high_risk_turns",
    "prior_cautious_turns",
    "prior_state_rank",
    "benchmark_harmbench",
)


@dataclass
class LearnedTrajectoryPolicy:
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def predict_proba(self, features: dict[str, float]) -> float:
        score = self.bias
        for name in FEATURE_NAMES:
            score += self.weights.get(name, 0.0) * features.get(name, 0.0)
        return 1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0)))


def fit_trajectory_policy(
    controller: Any,
    examples: list[Any],
    *,
    epochs: int = 300,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> LearnedTrajectoryPolicy:
    """Fit a lightweight learned transition gate from calibration trajectories."""

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for example in examples:
        history_texts: list[str] = []
        prior_state_rank = 0
        prior_high_risk_turns = 0
        prior_cautious_turns = 0
        risk_memory = 0.0
        safe_streak = 0
        risky_streak = 0
        prev_calibrated = 0.0
        total_user_turns = sum(1 for turn in example.turns if turn.role == "user") or 1
        user_turn_index = 0

        for turn in example.turns:
            if turn.role != "user":
                history_texts.append(turn.text)
                continue
            score_result = controller.scorer.score(turn.text, history_texts, example.metadata)
            calibrated = controller.calibrate(
                raw_score=score_result.raw_score,
                text=turn.text,
                benchmark=example.benchmark,
                prior_high_risk_turns=prior_high_risk_turns,
                prior_cautious_turns=prior_cautious_turns,
                turn_index=user_turn_index,
                total_user_turns=total_user_turns,
                risk_ema=risk_memory,
                prev_risk_level=prev_calibrated,
                safe_streak=safe_streak,
                risky_streak=risky_streak,
            )
            features = build_transition_features(
                raw_score=score_result.raw_score,
                calibrated_score=calibrated.probability,
                benchmark=example.benchmark,
                turn_index=user_turn_index,
                total_user_turns=total_user_turns,
                risk_memory=risk_memory,
                prev_risk_level=prev_calibrated,
                safe_streak=safe_streak,
                risky_streak=risky_streak,
                prior_high_risk_turns=prior_high_risk_turns,
                prior_cautious_turns=prior_cautious_turns,
                prior_state_rank=prior_state_rank,
                benchmark_aware=controller.config.benchmark_aware,
                history_enabled=controller.config.use_history,
            )
            rows.append(features)
            labels.append(1 if str(example.label.value) == "harmful" else 0)

            pressure = max(0.0, calibrated.probability - controller.config.thresholds["tau_c"])
            if controller.config.use_risk_memory:
                risk_memory = controller.config.risk_memory_decay * risk_memory + pressure
            elif controller.config.use_trajectory_features:
                risk_memory = controller.config.trajectory_decay * risk_memory + pressure
            else:
                risk_memory = 0.0

            if calibrated.probability < controller.config.thresholds["tau_c"]:
                safe_streak += 1
                risky_streak = 0
            else:
                risky_streak += 1
                safe_streak = 0

            if calibrated.probability >= controller.config.thresholds["tau_q"]:
                prior_state_rank = 3
                prior_high_risk_turns += 1
            elif calibrated.probability >= controller.config.thresholds["tau_h"]:
                prior_state_rank = max(prior_state_rank, 2)
            elif calibrated.probability >= controller.config.thresholds["tau_c"]:
                prior_state_rank = max(prior_state_rank, 1)
                prior_cautious_turns += 1
            else:
                prior_state_rank = max(0, prior_state_rank - 1)

            prev_calibrated = calibrated.probability
            user_turn_index += 1
            history_texts.append(turn.text)

    if not rows:
        return LearnedTrajectoryPolicy()

    weights = {name: 0.0 for name in FEATURE_NAMES}
    bias = 0.0
    pos_count = sum(labels)
    neg_count = max(len(labels) - pos_count, 1)
    pos_weight = neg_count / max(pos_count, 1)

    for _ in range(epochs):
        for features, label in zip(rows, labels, strict=False):
            score = bias
            for name in FEATURE_NAMES:
                score += weights[name] * features.get(name, 0.0)
            pred = 1.0 / (1.0 + math.exp(-max(min(score, 30.0), -30.0)))
            error = pred - label
            if label == 1:
                error *= pos_weight
            bias -= learning_rate * error
            for name in FEATURE_NAMES:
                value = features.get(name, 0.0)
                weights[name] -= learning_rate * (error * value + l2 * weights[name])

    return LearnedTrajectoryPolicy(weights=weights, bias=bias)

