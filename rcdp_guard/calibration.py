from __future__ import annotations

import math
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
    """Logistic risk head fitted by full-batch gradient descent.

    ``l2`` is the ridge coefficient lambda of the per-example objective
    ``sum_i logloss_i + lambda * ||w||^2`` (bias not penalised), i.e. the
    scikit-learn convention with ``C = 1 / (2 * lambda)``. The paper uses
    ``C = 0.5`` (``lambda = 1.0``). ``l2 = 0`` reproduces the submitted,
    unpenalised head exactly.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 300, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
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
            if self.l2:
                for key in keys:
                    grad[key] += 2.0 * self.l2 * self.weights[key]
            scale = 1.0 / len(rows)
            for key in self.weights:
                self.weights[key] -= self.lr * grad[key] * scale

    def decision_function(self, row: dict[str, float]) -> float:
        """Uncalibrated risk logit s~_t of the head."""
        z = self.weights.get("bias", 0.0)
        for key, value in row.items():
            z += self.weights.get(key, 0.0) * value
        return z

    def predict_proba(self, row: dict[str, float]) -> float:
        return sigmoid(self.decision_function(row))


class PlattScaler:
    """Platt scaling r = sigmoid(a * s + b) on a held-apart subset.

    Fitted with Platt's smoothed targets t+ = (N+ + 1)/(N+ + 2), t- = 1/(N- + 2)
    and the Newton method with backtracking of Lin, Lin and Weng (2007).
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b
        self.n_fit = 0
        self.targets: tuple[float, float] | None = None

    @staticmethod
    def _nll(scores, targets, A, B):
        total = 0.0
        for f, t in zip(scores, targets):
            fApB = f * A + B
            if fApB >= 0:
                total += t * fApB + math.log1p(math.exp(-fApB))
            else:
                total += (t - 1.0) * fApB + math.log1p(math.exp(fApB))
        return total

    def fit(self, scores: list[float], labels: list[int], max_iter: int = 100,
            min_step: float = 1e-10, sigma: float = 1e-12) -> "PlattScaler":
        prior1 = sum(1 for y in labels if y == 1)
        prior0 = len(labels) - prior1
        if prior1 == 0 or prior0 == 0:
            raise ValueError("Platt scaling needs both classes in its fitting subset")
        hi = (prior1 + 1.0) / (prior1 + 2.0)
        lo = 1.0 / (prior0 + 2.0)
        targets = [hi if y == 1 else lo for y in labels]
        # Platt's parametrisation P(y=1|f) = 1 / (1 + exp(A f + B)).
        A, B = 0.0, math.log((prior0 + 1.0) / (prior1 + 1.0))
        fval = self._nll(scores, targets, A, B)
        for _ in range(max_iter):
            h11 = h22 = sigma
            h21 = g1 = g2 = 0.0
            for f, t in zip(scores, targets):
                fApB = f * A + B
                if fApB >= 0:
                    e = math.exp(-fApB)
                    p, q = e / (1.0 + e), 1.0 / (1.0 + e)
                else:
                    e = math.exp(fApB)
                    p, q = 1.0 / (1.0 + e), e / (1.0 + e)
                d2 = p * q
                h11 += f * f * d2
                h22 += d2
                h21 += f * d2
                d1 = t - p
                g1 += f * d1
                g2 += d1
            if abs(g1) < 1e-5 and abs(g2) < 1e-5:
                break
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB
            step = 1.0
            while step >= min_step:
                nA, nB = A + step * dA, B + step * dB
                nf = self._nll(scores, targets, nA, nB)
                if nf < fval + 1e-4 * step * gd:
                    A, B, fval = nA, nB, nf
                    break
                step /= 2.0
            if step < min_step:
                break
        self.a, self.b = -A, -B
        self.n_fit = len(scores)
        self.targets = (hi, lo)
        return self

    def __call__(self, score: float) -> float:
        return sigmoid(self.a * score + self.b)


class PlattCalibratedHead:
    """Risk head followed by an (optional) Platt layer: r = sigmoid(a * s~ + b_cal)."""

    def __init__(self, head: LogisticCalibrator, platt: PlattScaler | None = None):
        self.head = head
        self.platt = platt

    @property
    def weights(self) -> dict[str, float]:
        return self.head.weights

    def decision_function(self, row: dict[str, float]) -> float:
        return self.head.decision_function(row)

    def predict_proba(self, row: dict[str, float]) -> float:
        z = self.head.decision_function(row)
        return self.platt(z) if self.platt is not None else sigmoid(z)


class RawScoreCalibrator:
    """Ablation: route on the raw scorer output itself (no head, no Platt)."""

    weights: dict[str, float] = {}

    def decision_function(self, row: dict[str, float]) -> float:
        return logit(row["raw_score"])

    def predict_proba(self, row: dict[str, float]) -> float:
        return row["raw_score"]


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
