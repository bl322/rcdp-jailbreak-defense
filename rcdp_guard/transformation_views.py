from __future__ import annotations

import re
from dataclasses import dataclass


_WRAPPER_PATTERN = re.compile(
    r"\b(?:rewrite|rephrase|transform|convert|translate|encode|decode|"
    r"summarize|extract|classify|format|structured|data-processing|"
    r"sequence-to-sequence|text-transformation)\b",
    flags=re.IGNORECASE,
)
_PAYLOAD_MARKER = re.compile(
    r"(?:^|\n)\s*(?:user\s+request|input\s+request|text\s+to\s+transform|"
    r"source\s+text|payload|instruction|input)\s*:\s*\n?",
    flags=re.IGNORECASE,
)
_QUOTED_PAYLOAD = re.compile(
    r"(?:following|provided|embedded|quoted)\s+(?:request|text|instruction)"
    r"\s*(?:is|:)\s*\"(.+?)\"",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class RiskView:
    """A deterministic interpretation of text embedded in a transformation task."""

    name: str
    text: str


@dataclass(frozen=True)
class TransformationAnalysis:
    """Result of a deterministic transformation-risk parse."""

    wrapper_confidence: float
    views: list[RiskView]

    @property
    def is_high_confidence(self) -> bool:
        return self.wrapper_confidence >= 0.80


def analyze_transformation_text(text: str, *, max_views: int = 3) -> TransformationAnalysis:
    """Recover embedded task payloads without rewriting or generating text."""

    normalized = text.strip()
    views = [RiskView(name="raw", text=normalized)]
    wrapper_confidence = 0.0
    if not normalized:
        return TransformationAnalysis(wrapper_confidence=wrapper_confidence, views=views)

    wrapper_match = _WRAPPER_PATTERN.search(normalized)
    if not wrapper_match:
        return TransformationAnalysis(wrapper_confidence=wrapper_confidence, views=views)

    wrapper_confidence += 0.45

    marker = _PAYLOAD_MARKER.search(normalized)
    if marker:
        wrapper_confidence += 0.30
        payload = normalized[marker.end() :].strip()
        if _valid_payload(payload):
            views.append(RiskView(name="embedded_payload", text=payload))

    if len(views) < max_views:
        quoted = _QUOTED_PAYLOAD.search(normalized)
        if quoted:
            wrapper_confidence += 0.15
            payload = quoted.group(1).strip()
            if _valid_payload(payload):
                views.append(RiskView(name="quoted_payload", text=payload))

    if len(views) > 1:
        wrapper_confidence += 0.10
    if len(normalized) > 160:
        wrapper_confidence += 0.05
    wrapper_confidence = min(wrapper_confidence, 0.99)

    deduplicated: list[RiskView] = []
    seen: set[str] = set()
    for view in views:
        key = re.sub(r"\s+", " ", view.text).strip().lower()
        if key and key not in seen:
            deduplicated.append(view)
            seen.add(key)
    return TransformationAnalysis(wrapper_confidence=wrapper_confidence, views=deduplicated[:max_views])


def _valid_payload(payload: str) -> bool:
    return 12 <= len(payload) <= 4096
