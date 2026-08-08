"""Backend contracts shared by the embedded and independent chat tracks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import uuid

from history_chatbot.dialogue.modes import ConversationMode


class PieceChatUiState(StrEnum):
    HIDDEN = "hidden"
    SHOWING_PROMPT = "showing_prompt"
    AWAITING_REFLECTION = "awaiting_reflection"
    RESPONDING = "responding"
    READY_FOR_NEXT_PIECE = "ready_for_next_piece"
    SKIPPED = "skipped"
    PAUSED = "paused"


class FreeChatUiState(StrEnum):
    CLOSED = "closed"
    OPENING = "opening"
    ACTIVE = "active"
    LOADING = "loading"
    SHOWING_CITATIONS = "showing_citations"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"
    RETURNING_TO_GAME = "returning_to_game"


class RequestState(StrEnum):
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    MISSING_CONTEXT = "missing_context"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SharedSessionContext:
    session_id: str | None = None
    locale: str = "ko"
    current_place_id: str | None = None
    current_piece_id: str | None = None
    completed_place_ids: tuple[str, ...] = ()
    completed_piece_ids: tuple[str, ...] = ()
    current_journey_step: str | None = None
    temporary_response_length_preference: str | None = None
    temporary_fatigue_state: str | None = None
    temporary_frustration_state: str | None = None
    active_persona: str = "giroksae"
    available_capabilities: tuple[str, ...] = ()
    storage_capability: bool = False
    user_consent: bool = False

    def __post_init__(self) -> None:
        if len(self.completed_place_ids) != len(set(self.completed_place_ids)):
            raise ValueError("completed_place_ids에는 중복이 없어야 합니다.")
        if any(not value.strip() for value in self.completed_place_ids):
            raise ValueError("completed_place_ids는 비어 있지 않은 문자열이어야 합니다.")
        if len(self.completed_piece_ids) != len(set(self.completed_piece_ids)):
            raise ValueError("completed_piece_ids에는 중복이 없어야 합니다.")
        if any(not value.strip() for value in self.completed_piece_ids):
            raise ValueError("completed_piece_ids는 비어 있지 않은 문자열이어야 합니다.")

    def completed_only(self, requested_piece_ids: tuple[str, ...]) -> tuple[str, ...]:
        completed = set(self.completed_piece_ids)
        return tuple(piece_id for piece_id in requested_piece_ids if piece_id in completed)


@dataclass(frozen=True, slots=True)
class ModeTransition:
    transition_id: str
    from_mode: ConversationMode
    to_mode: ConversationMode
    reason: str
    pending_user_question: str
    return_target: str
    preserve_game_state: bool
    source_session_id: str
    created_at: str
    current_place_id: str | None = None
    current_piece_id: str | None = None
    completed_piece_ids: tuple[str, ...] = ()

    @classmethod
    def open_free_chat(
        cls, *, question: str, context: SharedSessionContext, return_target: str,
    ) -> "ModeTransition":
        return cls(
            transition_id=uuid.uuid4().hex,
            from_mode=ConversationMode.PIECE_CHAT,
            to_mode=ConversationMode.FREE_CHAT,
            reason="detailed_history_requires_free_chat",
            pending_user_question=question,
            return_target=return_target,
            preserve_game_state=True,
            source_session_id=context.session_id or "",
            created_at=datetime.now(timezone.utc).isoformat(),
            current_place_id=context.current_place_id,
            current_piece_id=context.current_piece_id,
            completed_piece_ids=context.completed_piece_ids,
        )

    @classmethod
    def return_to_game(
        cls, *, context: SharedSessionContext, return_target: str,
    ) -> "ModeTransition":
        return cls(
            transition_id=uuid.uuid4().hex,
            from_mode=ConversationMode.FREE_CHAT,
            to_mode=ConversationMode.PIECE_CHAT,
            reason="user_closed_free_chat",
            pending_user_question="",
            return_target=return_target,
            preserve_game_state=True,
            source_session_id=context.session_id or "",
            created_at=datetime.now(timezone.utc).isoformat(),
            current_place_id=context.current_place_id,
            current_piece_id=context.current_piece_id,
            completed_piece_ids=context.completed_piece_ids,
        )


@dataclass(frozen=True, slots=True)
class PieceChatUiContract:
    state: PieceChatUiState
    reflection_input_enabled: bool = True
    skip_available: bool = True
    pause_available: bool = True
    next_piece_available: bool = False
    free_chat_available: bool = False
    storage_capability: bool = False


@dataclass(frozen=True, slots=True)
class FreeChatUiContract:
    state: FreeChatUiState
    suggested_questions: tuple[str, ...] = ()
    return_action: str = "RETURN_TO_GAME"
    transition_pending: bool = False
