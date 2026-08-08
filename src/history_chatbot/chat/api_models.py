"""Typed request contracts for the History Pieces HTTP API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
)


SESSION_PATTERN = re.compile(r"^[a-f0-9]{32}$")
LOCALE_PATTERN = re.compile(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?")
CONTEXT_ID_PATTERN = re.compile(r"^[\w][\w.:-]{0,127}$")
MAX_MESSAGE_LENGTH = 2000


class ApiRequest(BaseModel):
    """Keep accepting additive client fields while validating known fields."""

    model_config = ConfigDict(extra="allow")


def validate_session_id(value: str) -> str:
    if not SESSION_PATTERN.fullmatch(value):
        raise ValueError("session_id 형식이 올바르지 않습니다.")
    return value


def validate_locale(value: str) -> str:
    if not LOCALE_PATTERN.fullmatch(value):
        raise ValueError("locale 형식이 올바르지 않습니다.")
    return value


def validate_context_id(value: str) -> str:
    if not CONTEXT_ID_PATTERN.fullmatch(value):
        raise ValueError("관광 문맥 ID 형식이 올바르지 않습니다.")
    return value


class SessionCreateRequest(ApiRequest):
    locale: StrictStr = "ko"

    def resolved_locale(self) -> str:
        return validate_locale(self.locale)


class SessionBoundRequest(ApiRequest):
    session_id: StrictStr | None = None

    def resolved_session_id(self) -> str:
        return validate_session_id(self.session_id or "")


class TrackChatRequest(SessionBoundRequest):
    user_message: StrictStr | None = None
    pending_user_question: StrictStr | None = None
    locale: StrictStr | None = None
    ui_state: StrictStr | None = None
    return_target: StrictStr = "journey"

    def resolved_message(self) -> str:
        message = self.user_message
        if message is None:
            message = self.pending_user_question
        if message is None or not message.strip():
            raise ValueError("user_message 또는 pending_user_question이 필요합니다.")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"사용자 메시지는 {MAX_MESSAGE_LENGTH:,}자 이하여야 합니다.")
        return message.strip()

    def resolved_locale(self, default: str) -> str:
        return validate_locale(self.locale if self.locale is not None else default)


class PieceChatRequest(TrackChatRequest):
    pass


class FreeChatRequest(TrackChatRequest):
    pass


class TransitionRequest(SessionBoundRequest):
    from_mode: StrictStr | None = None
    to_mode: StrictStr | None = None
    mode_transition: dict[str, object] | None = None


class JourneyActionRequest(SessionBoundRequest):
    action_code: StrictStr | None = None


class GenericChatRequest(ApiRequest):
    user_query: StrictStr | None = None
    session_id: StrictStr | None = None
    locale: StrictStr = "ko"
    top_k: StrictInt = 3
    conversation_mode: StrictStr | None = None
    screen_type: StrictStr | None = None
    current_piece_id: StrictStr | None = None
    current_place_id: StrictStr | None = None
    visited_piece_ids: tuple[StrictStr, ...] = ()
    existing_style_preferences: tuple[StrictStr, ...] = ()
    current_journey_step: StrictStr | None = None
    piece_follow_up_count: StrictInt | None = None
    return_target: StrictStr = "game"
    available_capabilities: tuple[StrictStr, ...] = ()
    storage_capability: StrictBool = False
    user_consent: StrictBool = False
    mode_transition: dict[str, object] | None = None

    def service_payload(self) -> dict[str, object]:
        value = self.model_dump(exclude_none=True)
        if self.session_id is not None:
            value["session_id"] = validate_session_id(self.session_id)
        value["locale"] = validate_locale(self.locale)
        if self.user_query is not None and len(self.user_query) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"사용자 메시지는 {MAX_MESSAGE_LENGTH:,}자 이하여야 합니다.")
        return value


class ApiV1Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(ApiV1Model):
    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content는 비어 있을 수 없습니다.")
        return value.strip()


class SearchRequest(ApiV1Model):
    query: StrictStr = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    top_k: StrictInt = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query는 비어 있을 수 없습니다.")
        return value.strip()


class ChatRequest(ApiV1Model):
    message: StrictStr = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: tuple[HistoryMessage, ...] = Field(default=(), max_length=20)
    session_id: StrictStr | None = None
    locale: StrictStr = "ko"
    current_place_id: StrictStr | None = None
    current_piece_id: StrictStr | None = None
    completed_place_ids: tuple[StrictStr, ...] = Field(default=(), max_length=20)
    completed_piece_ids: tuple[StrictStr, ...] = Field(default=(), max_length=50)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message는 비어 있을 수 없습니다.")
        return value.strip()

    def resolved_session_id(self) -> str | None:
        return validate_session_id(self.session_id) if self.session_id is not None else None

    def resolved_locale(self) -> str:
        return validate_locale(self.locale)

    @field_validator("current_place_id", "current_piece_id")
    @classmethod
    def context_id_must_be_safe(cls, value: str | None) -> str | None:
        return validate_context_id(value) if value is not None else None

    @field_validator("completed_place_ids", "completed_piece_ids")
    @classmethod
    def context_ids_must_be_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("관광 문맥 ID에는 중복이 없어야 합니다.")
        return tuple(validate_context_id(value) for value in values)


class SearchResultResponse(ApiV1Model):
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float
    source_name: str
    source_url: str


class SearchResponse(ApiV1Model):
    query: str
    results: tuple[SearchResultResponse, ...]


class SourceResponse(ApiV1Model):
    document_id: str
    chunk_id: str
    title: str
    source_name: str
    source_url: str
    score: float


class ChatResponse(ApiV1Model):
    answer: str


class HealthResponse(ApiV1Model):
    status: Literal["ok"]


class ReadyResponse(ApiV1Model):
    ready: bool
    index_loaded: bool
    retriever: bool
    llm: bool
    backend: str
    llm_status: str


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "FreeChatRequest",
    "GenericChatRequest",
    "HealthResponse",
    "HistoryMessage",
    "JourneyActionRequest",
    "MAX_MESSAGE_LENGTH",
    "PieceChatRequest",
    "ReadyResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResultResponse",
    "SessionCreateRequest",
    "SourceResponse",
    "TrackChatRequest",
    "TransitionRequest",
    "validate_session_id",
]
