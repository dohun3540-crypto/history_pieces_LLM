"""Typed contracts for Giroksae seed data and runtime dialogue policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


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
    TECHNICAL_HELP = "TECHNICAL_HELP"
    NAVIGATION_HELP = "NAVIGATION_HELP"
    SAFETY_ACCESSIBILITY = "SAFETY_ACCESSIBILITY"


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


class SourceType(StrEnum):
    HUMAN_AUTHORED_SEED = "human_authored_seed"
    HUMAN_PROPOSED_SEED = "human_proposed_seed"


class ReviewStatus(StrEnum):
    REVIEWED_SEED = "reviewed_seed"
    REVIEW_PENDING = "review_pending"


class RequiredContext(StrEnum):
    APP_STATE = "app_state"
    JOURNEY_STATE = "journey_state"
    CURRENT_LOCATION = "current_location"
    MAP_DATA = "map_data"
    VERIFIED_FACILITY_DATA = "verified_facility_data"
    STORAGE_CAPABILITY = "storage_capability"
    USER_CONSENT = "user_consent"


class ActionCode(StrEnum):
    OPEN_TECH_DIAGNOSTIC_OVERLAY = "OPEN_TECH_DIAGNOSTIC_OVERLAY"
    CHECK_MISSION_COMPLETION_STATE = "CHECK_MISSION_COMPLETION_STATE"
    OPEN_AUDIO_TROUBLESHOOTING = "OPEN_AUDIO_TROUBLESHOOTING"
    OPEN_ROUTE_TO_NEXT_PIECE = "OPEN_ROUTE_TO_NEXT_PIECE"
    CALCULATE_ROUTE_ETA = "CALCULATE_ROUTE_ETA"
    RECALCULATE_ROUTE_OR_SHOW_HELP = "RECALCULATE_ROUTE_OR_SHOW_HELP"
    CHECK_ACCESSIBLE_ROUTE = "CHECK_ACCESSIBLE_ROUTE"
    CHECK_WHEELCHAIR_ACCESS = "CHECK_WHEELCHAIR_ACCESS"
    SHOW_VERIFIED_REST_AREAS_OR_HELP = "SHOW_VERIFIED_REST_AREAS_OR_HELP"
    SAVE_SHORT_REFLECTION = "SAVE_SHORT_REFLECTION"
    SKIP_REFLECTION = "SKIP_REFLECTION"
    GO_NEXT_PIECE = "GO_NEXT_PIECE"
    PAUSE_JOURNEY = "PAUSE_JOURNEY"
    CONTINUE_WITH_SHORT_MODE = "CONTINUE_WITH_SHORT_MODE"
    OPEN_FREE_CHAT = "OPEN_FREE_CHAT"
    OFFER_MORE_HISTORY_IN_FREE_CHAT = "OFFER_MORE_HISTORY_IN_FREE_CHAT"
    CLOSE_FREE_CHAT = "CLOSE_FREE_CHAT"
    RETURN_TO_GAME = "RETURN_TO_GAME"
    OPEN_CITATION_PANEL = "OPEN_CITATION_PANEL"
    SHOW_SUGGESTED_QUESTIONS = "SHOW_SUGGESTED_QUESTIONS"
    SUMMARIZE_COMPLETED_PIECES = "SUMMARIZE_COMPLETED_PIECES"
    ANSWER_WITH_CITATIONS = "ANSWER_WITH_CITATIONS"


class FallbackBehavior(StrEnum):
    NONE = "none"
    GENERAL_TECHNICAL_GUIDANCE = "general_technical_guidance"
    REQUIRE_LOCATION_OR_OFFICIAL_GUIDANCE = "require_location_or_official_guidance"
    REQUIRE_VERIFIED_FACILITY_GUIDANCE = "require_verified_facility_guidance"


@dataclass(frozen=True, slots=True)
class SituationExample:
    example_id: str
    situation_id: SituationId
    screen_type: ScreenType
    user_input: str
    response_goal: str
    response_draft: str
    response_draft_original: str
    response_template: str
    next_action: str
    personalization_tags: tuple[str, ...]
    context_state: tuple[str, ...]
    policy_flags: tuple[str, ...]
    required_context: tuple[RequiredContext, ...]
    requires_rag: bool
    requires_clarification: bool
    allows_follow_up: bool
    response_length_mode: ResponseLengthMode
    next_action_code: str
    fallback_behavior: FallbackBehavior
    locale: str
    source_type: SourceType
    review_status: ReviewStatus
    source_fields: dict[str, object]


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
    app_state: Mapping[str, object] | None = None
    journey_state: Mapping[str, object] | None = None
    current_location: Mapping[str, object] | None = None
    map_data: Mapping[str, object] | None = None
    verified_facility_data: Mapping[str, object] | None = None
    storage_capability: bool = False
    user_consent: bool = False
    supported_action_codes: tuple[str, ...] = ()


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
    next_action_code: str | None = None
    required_context: tuple[RequiredContext, ...] = ()
    policy_flags: tuple[str, ...] = ()
    context_state: tuple[str, ...] = ()
