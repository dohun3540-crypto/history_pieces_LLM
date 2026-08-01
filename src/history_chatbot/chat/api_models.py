"""Typed request contracts for the History Pieces HTTP API."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr


SESSION_PATTERN = re.compile(r"^[a-f0-9]{32}$")
LOCALE_PATTERN = re.compile(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?")
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


__all__ = [
    "FreeChatRequest",
    "GenericChatRequest",
    "JourneyActionRequest",
    "MAX_MESSAGE_LENGTH",
    "PieceChatRequest",
    "SessionCreateRequest",
    "TrackChatRequest",
    "TransitionRequest",
    "validate_session_id",
]
