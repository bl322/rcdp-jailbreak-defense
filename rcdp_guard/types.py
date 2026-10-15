from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    HARMFUL = "harmful"
    BENIGN = "benign"


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Example:
    example_id: str
    benchmark: str
    label: Label
    turns: list[Turn]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    raw_score: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    probability: float
    features: dict[str, float]


@dataclass
class EvalRecord:
    example_id: str
    benchmark: str
    label: str
    decision: str
    state: str
    raw_score: float
    calibrated_score: float
    uncertainty: float
    prompt_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
