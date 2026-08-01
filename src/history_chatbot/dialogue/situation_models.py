"""기록새 seed 및 런타임 분류 결과의 typed model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SituationId(StrEnum):
    INTRO_GIROKSAE = "INTRO_GIROKSAE"
    FREE_CHAT_GREETING = "FREE_CHAT_GREETING"
    REFLECTION_POSITIVE_GENERAL = "REFLECTION_POSITIVE_GENERAL"
    INTEREST_ARCHITECTURE = "INTEREST_ARCHITECTURE"
    INTEREST_PEOPLE = "INTEREST_PEOPLE"
    INTEREST_DAILY_CITY = "INTEREST_DAILY_CITY"
    COMPARISON_CONTEXT = "COMPARISON_CONTEXT"
    EMOTION_POSITIVE = "EMOTION_POSITIVE"
    EMOTION_NEGATIVE_HISTORY = "EMOTION_NEGATIVE_HISTORY"
    LOW_ENGAGEMENT = "LOW_ENGAGEMENT"
    HISTORY_FACT_QUESTION = "HISTORY_FACT_QUESTION"
    JOURNEY_CONTEXT_QUESTION = "JOURNEY_CONTEXT_QUESTION"
    RESPONSE_STYLE_REQUEST = "RESPONSE_STYLE_REQUEST"
    EVIDENCE_AND_CORRECTION = "EVIDENCE_AND_CORRECTION"
    STRONG_DISSATISFACTION = "STRONG_DISSATISFACTION"
    CROSS_CULTURAL_COMPARISON = "CROSS_CULTURAL_COMPARISON"
    PERSONAL_AND_LIGHT_CHAT = "PERSONAL_AND_LIGHT_CHAT"


class ScreenType(StrEnum):
    INTRO = "intro"
    PIECE_CHAT = "piece_chat"
    FREE_CHAT = "free_chat"
    PIECE_CHAT_OR_FREE_CHAT = "piece_chat_or_free_chat"


class ResponseLengthMode(StrEnum):
    DEFAULT = "default"
    SIMPLE = "simple"
    SHORT = "short"
    VERY_SHORT = "very_short"
    DETAILED = "detailed"
    SUMMARY = "summary"
    SOURCE_VIEW = "source_view"


@dataclass(frozen=True, slots=True)
class SituationExample:
    example_id: str
    situation_id: SituationId
    screen_type: ScreenType
    user_input: str
    response_goal: str
    response_draft: str
    next_action: str
    personalization_tags: tuple[str, ...]
    requires_rag: bool
    requires_clarification: bool
    allows_follow_up: bool
    response_length_mode: ResponseLengthMode
    next_action_code: str
    locale: str
    source_type: str
    review_status: str
    source_fields: dict[str, str]


@dataclass(frozen=True, slots=True)
class ClassificationInput:
    user_message: str
    conversation_mode: str = "free_chat"
    screen_type: ScreenType = ScreenType.FREE_CHAT
    locale: str = "ko"
    current_piece_id: str | None = None
    current_place_id: str | None = None
    recent_turns: tuple[str, ...] = ()
    visited_piece_ids: tuple[str, ...] = ()
    existing_style_preferences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    primary_situation_id: SituationId
    secondary_situation_ids: tuple[SituationId, ...]
    confidence: float
    detected_intent: str
    requires_rag: bool
    requires_clarification: bool
    response_length_mode: ResponseLengthMode
    personalization_tag_candidates: tuple[str, ...]
    next_action: str
    classification_reason_code: str
