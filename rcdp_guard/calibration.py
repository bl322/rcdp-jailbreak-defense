from __future__ import annotations

from dataclasses import dataclass

from .scorers import logit, sigmoid


FEATURE_GROUPS = {
    "raw": ("raw_score", "raw_logit"),
    "uncertainty": ("uncertainty_proxy",),
    "length": ("char_len", "word_len"),
    "lexical": ("contains_story_cover", "contains_research_cover"),
    "history": ("prior_high_risk_turns", "prior_cautious_turns"),
    "benchmark": ("benchmark_harmbench",),
    "trajectory": ("turn_index", "dialogue_progress", "risk_ema", "prev_risk_level", "safe_streak", "risky_streak"),
}


@dataclass
class FeatureConfig:
    enabled_groups: set[str]

    @classmethod
    def all(cls) -> "FeatureConfig":
        return cls(enabled_groups=set(FEATURE_GROUPS))

    def is_enabled(self, group: str) -> bool:
        return group in self.enabled_groups


class LogisticCalibrator:
    def __init__(self, lr: float = 0.1, epochs: int = 300):
        self.lr = lr
        self.epochs = epochs
        self.weights: dict[str, float] = {"bias": 0.0}

    def fit(self, rows: list[dict[str, float]], labels: list[int]) -> None:
        keys = sorted({k for row in rows for k in row})
        self.weights = {"bias": 0.0, **{k: 0.0 for k in keys}}
        if not rows:
            return

        for _ in range(self.epochs):
            grad = {k: 0.0 for k in self.weights}
            for row, label in zip(rows, labels):
                pred = self.predict_proba(row)
                err = pred - label
                grad["bias"] += err
                for key in keys:
                    grad[key] += err * row.get(key, 0.0)
            scale = 1.0 / len(rows)
            for key in self.weights:
                self.weights[key] -= self.lr * grad[key] * scale

    def predict_proba(self, row: dict[str, float]) -> float:
        z = self.weights.get("bias", 0.0)
        for key, value in row.items():
            z += self.weights.get(key, 0.0) * value
        return sigmoid(z)


def build_features(
    raw_score: float,
    text: str,
    benchmark: str,
    prior_high_risk_turns: int,
    prior_cautious_turns: int,
    feature_config: FeatureConfig,
    turn_index: int = 0,
    total_user_turns: int = 1,
    risk_ema: float = 0.0,
    prev_risk_level: float = 0.0,
    safe_streak: int = 0,
    risky_streak: int = 0,
) -> dict[str, float]:
    features: dict[str, float] = {}
    lower = text.lower()

    if feature_config.is_enabled("raw"):
        features["raw_score"] = raw_score
        features["raw_logit"] = logit(raw_score)

    if feature_config.is_enabled("uncertainty"):
        features["uncertainty_proxy"] = 1.0 - abs(2 * raw_score - 1.0)

    if feature_config.is_enabled("length"):
        features["char_len"] = min(len(text) / 1000.0, 1.0)
        features["word_len"] = min(len(text.split()) / 200.0, 1.0)

    if feature_config.is_enabled("lexical"):
        features["contains_story_cover"] = float("story" in lower or "fictional" in lower)
        features["contains_research_cover"] = float("research" in lower or "educational" in lower)

    if feature_config.is_enabled("history"):
        features["prior_high_risk_turns"] = float(prior_high_risk_turns)
        features["prior_cautious_turns"] = float(prior_cautious_turns)

    if feature_config.is_enabled("benchmark"):
        features["benchmark_harmbench"] = float(benchmark.lower().startswith("harm"))

    if feature_config.is_enabled("trajectory"):
        total_turns = max(total_user_turns - 1, 1)
        features["turn_index"] = min(max(turn_index / total_turns, 0.0), 1.0)
        features["dialogue_progress"] = min(max(turn_index / max(total_user_turns, 1), 0.0), 1.0)
        features["risk_ema"] = min(max(risk_ema, 0.0), 1.0)
        features["prev_risk_level"] = min(max(prev_risk_level, 0.0), 1.0)
        features["safe_streak"] = min(float(safe_streak), 8.0) / 8.0
        features["risky_streak"] = min(float(risky_streak), 8.0) / 8.0

    return features
