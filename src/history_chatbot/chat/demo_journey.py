"""Provider contract and in-memory journey state for the reference web demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Protocol

from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.situation_models import ActionCode
from history_chatbot.dialogue.track_models import FreeChatUiState, PieceChatUiState


DEMO_PIECES = (
    ("demo-piece-1", "조각 1"),
    ("demo-piece-2", "조각 2"),
    ("demo-piece-3", "조각 3"),
)
DEMO_CAPABILITIES = (
    ActionCode.SKIP_REFLECTION.value,
    ActionCode.GO_NEXT_PIECE.value,
    ActionCode.PAUSE_JOURNEY.value,
    ActionCode.CONTINUE_WITH_SHORT_MODE.value,
    ActionCode.OPEN_FREE_CHAT.value,
    ActionCode.CLOSE_FREE_CHAT.value,
    ActionCode.RETURN_TO_GAME.value,
)


class JourneyProviderError(Exception):
    def __init__(self, error_code: str, message: str, *, status_code: int, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


@dataclass(slots=True)
class DemoJourneyState:
    session_id: str
    locale: str = "ko"
    current_place_id: str = "demo-place"
    current_piece_id: str | None = DEMO_PIECES[0][0]
    completed_piece_ids: list[str] = field(default_factory=list)
    current_journey_step: str = "piece_1_active"
    chat_mode: str = ConversationMode.PIECE_CHAT.value
    piece_ui_state: str = PieceChatUiState.SHOWING_PROMPT.value
    free_ui_state: str = FreeChatUiState.CLOSED.value
    active_transition: dict[str, object] | None = None
    temporary_context_state: list[str] = field(default_factory=list)
    available_capabilities: tuple[str, ...] = DEMO_CAPABILITIES

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["completed_piece_ids"] = tuple(self.completed_piece_ids)
        result["demo_pieces"] = tuple(
            {"piece_id": piece_id, "label": label} for piece_id, label in DEMO_PIECES
        )
        result["current_piece_label"] = next(
            (label for piece_id, label in DEMO_PIECES if piece_id == self.current_piece_id),
            "여정 완료",
        )
        result["ephemeral"] = True
        return result


class JourneyProvider(Protocol):
    def create(self, session_id: str, locale: str = "ko") -> DemoJourneyState: ...
    def get(self, session_id: str) -> DemoJourneyState: ...
    def apply_action(self, session_id: str, action_code: str, payload: dict[str, object]) -> DemoJourneyState: ...


class InMemoryDemoJourneyProvider:
    """Ephemeral provider; state is intentionally lost when the server restarts."""

    def __init__(self) -> None:
        self._states: dict[str, DemoJourneyState] = {}
        self._lock = RLock()

    def create(self, session_id: str, locale: str = "ko") -> DemoJourneyState:
        with self._lock:
            state = DemoJourneyState(session_id=session_id, locale=locale)
            self._states[session_id] = state
            return state

    def get(self, session_id: str) -> DemoJourneyState:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise JourneyProviderError("session_not_found", "데모 세션을 찾을 수 없습니다.", status_code=404)
            return state

    def apply_action(self, session_id: str, action_code: str, payload: dict[str, object]) -> DemoJourneyState:
        try:
            action = ActionCode(action_code)
        except ValueError as error:
            raise JourneyProviderError("unsupported_action", "지원하지 않는 action입니다.", status_code=409) from error
        state = self.get(session_id)
        if action.value not in state.available_capabilities:
            raise JourneyProviderError("capability_unavailable", "현재 demo provider가 지원하지 않는 action입니다.", status_code=409)
        with self._lock:
            if action == ActionCode.GO_NEXT_PIECE:
                self._go_next(state)
            elif action == ActionCode.SKIP_REFLECTION:
                state.piece_ui_state = PieceChatUiState.SKIPPED.value
            elif action == ActionCode.PAUSE_JOURNEY:
                state.piece_ui_state = PieceChatUiState.PAUSED.value
                self._add_context(state, "current_fatigue")
            elif action == ActionCode.CONTINUE_WITH_SHORT_MODE:
                state.piece_ui_state = PieceChatUiState.RESPONDING.value
                self._add_context(state, "prefers_short_current")
            elif action == ActionCode.OPEN_FREE_CHAT:
                state.chat_mode = ConversationMode.FREE_CHAT.value
                state.free_ui_state = FreeChatUiState.ACTIVE.value
                transition = payload.get("mode_transition")
                state.active_transition = dict(transition) if type(transition) is dict else None
            elif action in {ActionCode.CLOSE_FREE_CHAT, ActionCode.RETURN_TO_GAME}:
                state.chat_mode = ConversationMode.PIECE_CHAT.value
                state.free_ui_state = FreeChatUiState.CLOSED.value
                state.active_transition = None
            return state

    @staticmethod
    def _add_context(state: DemoJourneyState, value: str) -> None:
        if value not in state.temporary_context_state:
            state.temporary_context_state.append(value)

    @staticmethod
    def _go_next(state: DemoJourneyState) -> None:
        if state.current_piece_id is None:
            return
        if state.current_piece_id not in state.completed_piece_ids:
            state.completed_piece_ids.append(state.current_piece_id)
        current_index = next(
            index for index, item in enumerate(DEMO_PIECES) if item[0] == state.current_piece_id
        )
        if current_index + 1 >= len(DEMO_PIECES):
            state.current_piece_id = None
            state.current_journey_step = "journey_complete"
            state.piece_ui_state = PieceChatUiState.HIDDEN.value
        else:
            state.current_piece_id = DEMO_PIECES[current_index + 1][0]
            state.current_journey_step = f"piece_{current_index + 2}_active"
            state.piece_ui_state = PieceChatUiState.SHOWING_PROMPT.value
