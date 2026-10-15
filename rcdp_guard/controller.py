from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .calibration import FeatureConfig, LogisticCalibrator, build_features
from .scorers import KeywordRiskScorer, RiskScorer
from .trajectory_features import build_transition_features
from .trajectory_policy import LearnedTrajectoryPolicy
from .transformation_views import analyze_transformation_text
from .types import CalibrationResult, EvalRecord, Example


class ControllerState(str, Enum):
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    HARDENED = "hardened"
    QUARANTINE = "quarantine"


@dataclass
class ControllerConfig:
    num_states: int = 4
    allow_state_downgrade: bool = False
    history_decay: int = 0
    benchmark_aware: bool = True
    use_uncertainty_gate: bool = True
    use_history: bool = True
    use_risk_memory: bool = False
    use_trajectory_features: bool = True
    use_learned_transition_policy: bool = False
    use_transformation_views: bool = False
    transformation_max_views: int = 3
    transformation_min_confidence: float = 0.80
    risk_memory_decay: float = 0.80
    risk_memory_gain: float = 0.35
    trajectory_decay: float = 0.72
    trajectory_gain: float = 0.45
    trajectory_trend_gain: float = 0.18
    trajectory_policy_strength: float = 0.16
    safe_streak_to_downgrade: int = 2
    risky_streak_to_escalate: int = 2
    use_hazard_latch: bool = False
    hazard_latch_min_state: ControllerState = ControllerState.HARDENED
    hazard_latch_uncertainty_threshold: float = 0.62
    hazard_latch_risk_threshold: float = 0.38
    use_terminal_semantic_review: bool = False
    terminal_review_min_turns: int = 2
    terminal_review_threshold: float = 0.40
    terminal_history_threshold: float = 0.32
    terminal_peak_threshold: float = 0.42
    terminal_risk_gain: float = 0.55
    terminal_peak_gain: float = 0.35
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "tau_c": 0.45,
            "tau_h": 0.68,
            "tau_q": 0.84,
            "tau_u": 0.28,
        }
    )
    state_prompt_tokens: dict[ControllerState, int] = field(
        default_factory=lambda: {
            ControllerState.NORMAL: 6,
            ControllerState.CAUTIOUS: 28,
            ControllerState.HARDENED: 52,
            ControllerState.QUARANTINE: 18,
        }
    )


class RCDPController:
    def __init__(
        self,
        scorer: RiskScorer | None = None,
        terminal_review_scorer: RiskScorer | None = None,
        calibrator: LogisticCalibrator | None = None,
        feature_config: FeatureConfig | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.scorer = scorer or KeywordRiskScorer()
        self.terminal_review_scorer = terminal_review_scorer
        self.calibrator = calibrator or LogisticCalibrator()
        self.feature_config = feature_config or FeatureConfig.all()
        self.config = config or ControllerConfig()
        self.trajectory_policy: LearnedTrajectoryPolicy | None = None

    def calibrate(
        self,
        raw_score: float,
        text: str,
        benchmark: str,
        prior_high_risk_turns: int,
        prior_cautious_turns: int,
        turn_index: int = 0,
        total_user_turns: int = 1,
        risk_ema: float = 0.0,
        prev_risk_level: float = 0.0,
        safe_streak: int = 0,
        risky_streak: int = 0,
    ) -> CalibrationResult:
        features = build_features(
            raw_score=raw_score,
            text=text,
            benchmark=benchmark if self.config.benchmark_aware else "unknown",
            prior_high_risk_turns=prior_high_risk_turns if self.config.use_history else 0,
            prior_cautious_turns=prior_cautious_turns if self.config.use_history else 0,
            turn_index=turn_index if self.config.use_trajectory_features else 0,
            total_user_turns=total_user_turns if self.config.use_trajectory_features else 1,
            risk_ema=risk_ema if self.config.use_trajectory_features else 0.0,
            prev_risk_level=prev_risk_level if self.config.use_trajectory_features else 0.0,
            safe_streak=safe_streak if self.config.use_trajectory_features else 0,
            risky_streak=risky_streak if self.config.use_trajectory_features else 0,
            feature_config=self.feature_config,
        )
        prob = self.calibrator.predict_proba(features)
        return CalibrationResult(probability=prob, features=features)

    def select_state(
        self,
        calibrated: float,
        uncertainty: float,
        prior_state: ControllerState | None,
        history_flag: bool,
        safe_streak: int,
        risky_streak: int,
        dialogue_progress: float = 0.0,
        risk_memory: float = 0.0,
        prev_risk_level: float = 0.0,
        transition_score: float = 0.5,
    ) -> ControllerState:
        return select_controller_state(
            calibrated=calibrated,
            uncertainty=uncertainty,
            prior_state=prior_state,
            history_flag=history_flag,
            safe_streak=safe_streak,
            risky_streak=risky_streak,
            config=self.config,
            dialogue_progress=dialogue_progress,
            risk_memory=risk_memory,
            prev_risk_level=prev_risk_level,
            transition_score=transition_score,
        )

    def evaluate_example(self, example: Example) -> list[EvalRecord]:
        records: list[EvalRecord] = []
        history_texts: list[str] = []
        user_texts: list[str] = []
        prior_state: ControllerState | None = None
        prior_high_risk_turns = 0
        prior_cautious_turns = 0
        risk_memory = 0.0
        safe_streak = 0
        risky_streak = 0
        hazard_latched = False
        prev_calibrated = 0.0
        peak_effective_risk = 0.0
        peak_raw_score = 0.0
        user_turn_index = 0
        total_user_turns = sum(1 for turn in example.turns if turn.role == "user") or 1

        for idx, turn in enumerate(example.turns):
            if turn.role != "user":
                history_texts.append(turn.text)
                continue

            if self.config.allow_state_downgrade and self.config.history_decay > 0:
                prior_high_risk_turns = max(0, prior_high_risk_turns - self.config.history_decay)
                prior_cautious_turns = max(0, prior_cautious_turns - self.config.history_decay)

            score_result, view_scores = self._score_turn(
                turn.text,
                history_texts,
                example.metadata,
            )
            terminal_review = None
            if (
                self.config.use_terminal_semantic_review
                and len(user_texts) + 1 >= self.config.terminal_review_min_turns
                and user_turn_index == total_user_turns - 1
                and self.terminal_review_scorer is not None
            ):
                terminal_review = self.terminal_review_scorer.score(
                    turn.text,
                    user_texts,
                    {**example.metadata, "task": "terminal-semantic-review"},
                )
            calibrated = self.calibrate(
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
            uncertainty = 1.0 - abs(2 * calibrated.probability - 1.0)
            trend = calibrated.probability - prev_calibrated
            effective_risk = calibrated.probability
            pressure = max(0.0, calibrated.probability - self.config.thresholds["tau_c"])
            transition_features = build_transition_features(
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
                prior_state_rank=_rank(prior_state) if prior_state is not None else 0,
                benchmark_aware=self.config.benchmark_aware,
                history_enabled=self.config.use_history,
            )
            transition_score = 0.5
            if self.config.use_learned_transition_policy and self.trajectory_policy is not None:
                transition_score = self.trajectory_policy.predict_proba(transition_features)

            if self.config.use_risk_memory:
                effective_risk = min(
                    1.0,
                    calibrated.probability
                    + self.config.risk_memory_gain * risk_memory,
                )

            if self.config.use_trajectory_features:
                effective_risk = min(
                    1.0,
                    effective_risk
                    + self.config.trajectory_gain * risk_memory
                    + self.config.trajectory_trend_gain * max(0.0, trend)
                    - 0.05 * max(0, safe_streak - 1),
                )

            original_effective_risk = effective_risk
            terminal_joint_risk = 0.0
            terminal_review_triggered = False
            if terminal_review is not None:
                # Review the final request in the context of preceding user
                # turns. The auxiliary score is only a multi-turn terminal
                # signal and never changes the single-turn operating point.
                terminal_joint_risk = min(
                    1.0,
                    terminal_review.raw_score
                    + self.config.terminal_risk_gain
                    * max(0.0, terminal_review.raw_score - self.config.terminal_review_threshold)
                    + self.config.terminal_peak_gain
                    * peak_effective_risk,
                )
                terminal_review_triggered = (
                    terminal_review.raw_score >= self.config.terminal_review_threshold
                    or (
                        peak_effective_risk >= self.config.terminal_peak_threshold
                        and terminal_review.raw_score >= self.config.terminal_history_threshold
                    )
                )
                if terminal_review_triggered:
                    effective_risk = max(effective_risk, terminal_joint_risk)
            state = self.select_state(
                calibrated=effective_risk,
                uncertainty=uncertainty,
                prior_state=prior_state,
                history_flag=prior_high_risk_turns > 0,
                safe_streak=safe_streak,
                risky_streak=risky_streak,
                dialogue_progress=(user_turn_index / max(total_user_turns - 1, 1)) if self.config.use_trajectory_features else 0.0,
                risk_memory=risk_memory,
                prev_risk_level=prev_calibrated,
                transition_score=transition_score,
            )
            if terminal_review_triggered:
                # A softened final turn must not erase harmful intent established
                # by the complete trajectory. This gate is multi-turn only.
                state = ControllerState.QUARANTINE
                effective_risk = 1.0
            # Once a dialogue reached a hardened state, later turns cannot
            # silently erase that evidence through benign-looking padding or
            # a semantically obfuscated final request. The stricter trigger
            # avoids latching ordinary cautious single-turn interactions.
            if (
                self.config.use_hazard_latch
                and hazard_latched
            ):
                state = ControllerState.QUARANTINE
            decision = "block" if state == ControllerState.QUARANTINE else "allow"
            prompt_tokens = self.config.state_prompt_tokens[state]

            records.append(
                EvalRecord(
                    example_id=f"{example.example_id}:turn-{idx}",
                    benchmark=example.benchmark,
                    label=example.label.value,
                    decision=decision,
                    state=state.value,
                    raw_score=score_result.raw_score,
                    calibrated_score=calibrated.probability,
                    uncertainty=uncertainty,
                    prompt_tokens=prompt_tokens,
                    metadata={
                        "reasons": score_result.reasons,
                        "risk_views": view_scores,
                        "features": calibrated.features,
                        "effective_risk": effective_risk,
                        "risk_memory": risk_memory,
                        "safe_streak": safe_streak,
                        "risky_streak": risky_streak,
                        "hazard_latched": hazard_latched,
                        "transition_score": transition_score,
                        "original_effective_risk": original_effective_risk,
                        "peak_effective_risk": peak_effective_risk,
                        "peak_raw_score": peak_raw_score,
                        "terminal_joint_risk": terminal_joint_risk,
                        "terminal_review_triggered": terminal_review_triggered,
                        "terminal_semantic_review": (
                            {
                                **terminal_review.metadata,
                                "raw_score": terminal_review.raw_score,
                                "threshold": self.config.terminal_review_threshold,
                            }
                            if terminal_review is not None
                            else None
                        ),
                    },
                )
            )

            if state in {ControllerState.HARDENED, ControllerState.QUARANTINE}:
                prior_high_risk_turns += 1
            if state == ControllerState.CAUTIOUS:
                prior_cautious_turns += 1
            if _rank(state) >= _rank(self.config.hazard_latch_min_state):
                hazard_latched = True

            if self.config.use_risk_memory or self.config.use_trajectory_features:
                decay = self.config.risk_memory_decay if self.config.use_risk_memory else self.config.trajectory_decay
                risk_memory = decay * risk_memory + pressure
            else:
                risk_memory = 0.0

            if calibrated.probability < self.config.thresholds["tau_c"]:
                safe_streak += 1
                risky_streak = 0
            else:
                risky_streak += 1
                safe_streak = 0

            prior_state = state
            prev_calibrated = calibrated.probability
            peak_effective_risk = max(peak_effective_risk, effective_risk)
            peak_raw_score = max(peak_raw_score, score_result.raw_score)
            user_turn_index += 1
            user_texts.append(turn.text)
            history_texts.append(turn.text)

        return records

    def _score_turn(
        self,
        text: str,
        history_texts: list[str],
        metadata: dict,
    ):
        if not self.config.use_transformation_views:
            result = self.scorer.score(text, history_texts, metadata)
            return result, [{"name": "raw", "raw_score": result.raw_score}]

        analysis = analyze_transformation_text(
            text,
            max_views=self.config.transformation_max_views,
        )
        if analysis.wrapper_confidence < self.config.transformation_min_confidence:
            result = self.scorer.score(text, history_texts, metadata)
            return (
                result,
                [
                    {
                        "name": "raw",
                        "raw_score": result.raw_score,
                        "wrapper_confidence": analysis.wrapper_confidence,
                    }
                ],
            )

        scored = []
        for view in analysis.views:
            result = self.scorer.score(
                view.text,
                history_texts if view.name == "raw" else None,
                metadata,
            )
            scored.append((view, result))

        raw_result = scored[0][1]
        payload_results = [result for view, result in scored[1:]]
        best_payload = max(payload_results, key=lambda item: item.raw_score) if payload_results else raw_result
        payload_delta = max(0.0, best_payload.raw_score - raw_result.raw_score)
        aggregated_raw_score = min(
            1.0,
            raw_result.raw_score + analysis.wrapper_confidence * payload_delta,
        )
        chosen_view = scored[0][0] if not payload_results or aggregated_raw_score <= raw_result.raw_score else max(
            scored[1:],
            key=lambda item: item[1].raw_score,
        )[0]
        view_scores = [
            {
                "name": view.name,
                "raw_score": result.raw_score,
                "wrapper_confidence": analysis.wrapper_confidence,
            }
            for view, result in scored
        ]
        base_result = scored[0][1]
        reasons = list(base_result.reasons)
        if payload_results:
            reasons.append(f"best_payload_risk={best_payload.raw_score:.4f}")
            reasons.append(f"risk_delta={payload_delta:.4f}")
            if chosen_view.name != "raw":
                reasons.append(f"risk_aggregation_view={chosen_view.name}")
        return (
            type(base_result)(
                raw_score=aggregated_raw_score,
                reasons=reasons,
                metadata={
                    **base_result.metadata,
                    "aggregation": "confidence-gated-risk-delta",
                    "selected_view": chosen_view.name,
                    "num_views": len(scored),
                    "wrapper_confidence": analysis.wrapper_confidence,
                    "payload_delta": payload_delta,
                    "best_payload_risk": best_payload.raw_score,
                    "raw_view_risk": raw_result.raw_score,
                },
            ),
            view_scores,
        )


def _rank(state: ControllerState) -> int:
    mapping = {
        ControllerState.NORMAL: 0,
        ControllerState.CAUTIOUS: 1,
        ControllerState.HARDENED: 2,
        ControllerState.QUARANTINE: 3,
    }
    return mapping[state]


def select_controller_state(
    *,
    calibrated: float,
    uncertainty: float,
    prior_state: ControllerState | None,
    history_flag: bool,
    safe_streak: int,
    risky_streak: int,
    config: ControllerConfig,
    dialogue_progress: float = 0.0,
    risk_memory: float = 0.0,
    prev_risk_level: float = 0.0,
    transition_score: float = 0.5,
) -> ControllerState:
    tau_c = config.thresholds["tau_c"]
    tau_h = config.thresholds["tau_h"]
    tau_q = config.thresholds["tau_q"]
    tau_u = config.thresholds["tau_u"]

    if config.use_trajectory_features:
        progress = min(max(dialogue_progress, 0.0), 1.0)
        memory = min(max(risk_memory, 0.0), 1.0)
        trend = max(0.0, calibrated - prev_risk_level)
        tau_c = max(0.15, tau_c - 0.03 * progress - 0.02 * memory - 0.01 * trend)
        tau_h = max(0.25, tau_h - 0.05 * progress - 0.03 * memory - 0.02 * trend)
        tau_q = max(0.35, tau_q - 0.07 * progress - 0.04 * memory - 0.03 * trend)

    if config.use_learned_transition_policy:
        policy_shift = transition_score - 0.5
        tau_c = max(0.12, tau_c - 0.06 * policy_shift * config.trajectory_policy_strength)
        tau_h = max(0.22, tau_h - 0.09 * policy_shift * config.trajectory_policy_strength)
        tau_q = max(0.34, tau_q - 0.12 * policy_shift * config.trajectory_policy_strength)

    if config.num_states == 2:
        state = ControllerState.QUARANTINE if calibrated >= tau_h else ControllerState.NORMAL
    elif config.num_states == 3:
        if calibrated >= tau_q:
            state = ControllerState.QUARANTINE
        elif calibrated >= tau_c or (config.use_uncertainty_gate and uncertainty >= tau_u):
            state = ControllerState.CAUTIOUS
        else:
            state = ControllerState.NORMAL
    else:
        if calibrated >= tau_q:
            state = ControllerState.QUARANTINE
        elif calibrated >= tau_h:
            state = ControllerState.HARDENED
        elif calibrated >= tau_c or (config.use_uncertainty_gate and uncertainty >= tau_u):
            state = ControllerState.CAUTIOUS
        else:
            state = ControllerState.NORMAL

    if risky_streak >= config.risky_streak_to_escalate:
        if state == ControllerState.NORMAL:
            state = ControllerState.CAUTIOUS
        elif state == ControllerState.CAUTIOUS and calibrated >= tau_h:
            state = ControllerState.HARDENED

    if config.use_trajectory_features and config.num_states == 4:
        if state == ControllerState.CAUTIOUS and risky_streak >= 1 and (
            dialogue_progress >= 0.33 or risk_memory >= 0.12 or calibrated >= tau_h
        ):
            state = ControllerState.HARDENED
        elif state == ControllerState.HARDENED and risky_streak >= 1 and (
            dialogue_progress >= 0.5 or risk_memory >= 0.18 or calibrated >= tau_q
        ):
            state = ControllerState.QUARANTINE

    if safe_streak >= config.safe_streak_to_downgrade and state != ControllerState.NORMAL:
        if state == ControllerState.QUARANTINE and calibrated < tau_q:
            state = ControllerState.HARDENED
        elif state == ControllerState.HARDENED and calibrated < tau_h:
            state = ControllerState.CAUTIOUS
        elif state == ControllerState.CAUTIOUS and calibrated < tau_c:
            state = ControllerState.NORMAL

    if history_flag and config.use_history and state == ControllerState.NORMAL and safe_streak == 0:
        state = ControllerState.CAUTIOUS

    if (
        not config.allow_state_downgrade
        and prior_state is not None
        and _rank(prior_state) > _rank(state)
    ):
        return prior_state

    return state
