"""민감정보를 추론하지 않는 삭제 가능한 개인화 관찰값."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum


class TagScope(StrEnum):
    SESSION_OBSERVATION = "session_observation"
    PREFERENCE_CANDIDATE = "preference_candidate"
    JOURNEY_INTEREST = "journey_interest"


SESSION_TAGS = {
    "current_fatigue", "heat_discomfort", "technical_issue", "navigation_issue",
    "accessibility_request", "emotion_sadness", "frustration", "engagement_low",
}
PREFERENCE_TAGS = {
    "prefers_short", "prefers_very_short", "prefers_simple", "prefers_detailed",
    "prefers_summary", "prefers_fast_progress",
}
JOURNEY_TAGS = {
    "interest_architecture", "interest_people", "interest_daily_life",
    "interest_transport", "interest_commerce", "interest_urban_change",
    "interest_preservation", "interest_cross_cultural", "interest_historical_figures",
}


@dataclass(frozen=True, slots=True)
class TagObservation:
    tag: str
    scope: TagScope
    confidence: float
    evidence_turn_id: str
    observed_at: str
    original_user_message: str
    profile_candidate: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag, "scope": self.scope.value, "confidence": self.confidence,
            "evidence_turn_id": self.evidence_turn_id, "observed_at": self.observed_at,
            "original_user_message": self.original_user_message,
            "profile_candidate": self.profile_candidate,
        }


def classify_scope(tag: str) -> TagScope | None:
    normalized = tag.removesuffix("_temporarily").removesuffix("_possible")
    if normalized in SESSION_TAGS or tag.startswith("emotion_") or tag in {"service_dissatisfaction", "engagement_very_low"}:
        return TagScope.SESSION_OBSERVATION
    if normalized in PREFERENCE_TAGS:
        return TagScope.PREFERENCE_CANDIDATE
    if normalized in JOURNEY_TAGS or tag.startswith("interest_"):
        return TagScope.JOURNEY_INTEREST
    return None


def observations(tags: tuple[str, ...], *, turn_id: str, user_message: str, repeated_tags: frozenset[str] = frozenset()) -> tuple[TagObservation, ...]:
    now = datetime.now(timezone.utc).isoformat()
    result = []
    for tag in tags:
        scope = classify_scope(tag)
        if scope is None:
            continue
        result.append(TagObservation(
            tag, scope, 0.9, turn_id, now, user_message,
            profile_candidate=scope == TagScope.PREFERENCE_CANDIDATE and tag in repeated_tags,
        ))
    return tuple(result)
