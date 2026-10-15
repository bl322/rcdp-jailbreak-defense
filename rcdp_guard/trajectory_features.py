from __future__ import annotations

import math
from typing import Mapping


def safe_logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def build_transition_features(
    *,
    raw_score: float,
    calibrated_score: float,
    benchmark: str,
    turn_index: int,
    total_user_turns: int,
    risk_memory: float,
    prev_risk_level: float,
    safe_streak: int,
    risky_streak: int,
    prior_high_risk_turns: int,
    prior_cautious_turns: int,
    prior_state_rank: int = 0,
    benchmark_aware: bool = True,
    history_enabled: bool = True,
) -> dict[str, float]:
    """Build compact trajectory features for learned transition control."""

    turn_scale = max(total_user_turns - 1, 1)
    progress = turn_index / turn_scale
    uncertainty = 1.0 - abs(2.0 * calibrated_score - 1.0)
    raw_uncertainty = 1.0 - abs(2.0 * raw_score - 1.0)
    label_source = benchmark.lower()
    benchmark_flag = 1.0 if benchmark_aware and "harmbench" in label_source else 0.0
    safe_len = max(len(str(benchmark)), 1)

    return {
        "raw_score": raw_score,
        "calibrated_score": calibrated_score,
        "raw_logit": safe_logit(raw_score),
        "uncertainty": uncertainty,
        "raw_uncertainty": raw_uncertainty,
        "dialogue_progress": progress,
        "risk_memory": risk_memory,
        "prev_risk_level": prev_risk_level,
        "safe_streak": float(safe_streak if history_enabled else 0),
        "risky_streak": float(risky_streak if history_enabled else 0),
        "prior_high_risk_turns": float(prior_high_risk_turns if history_enabled else 0),
        "prior_cautious_turns": float(prior_cautious_turns if history_enabled else 0),
        "prior_state_rank": float(prior_state_rank),
        "benchmark_harmbench": benchmark_flag,
        "benchmark_name_len": float(safe_len),
    }

