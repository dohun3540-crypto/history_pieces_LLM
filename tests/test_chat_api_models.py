import pytest
from pydantic import ValidationError

from history_chatbot.chat.api_models import (
    FreeChatRequest,
    GenericChatRequest,
    JourneyActionRequest,
    MAX_MESSAGE_LENGTH,
    PieceChatRequest,
    SessionCreateRequest,
    TransitionRequest,
)
from history_chatbot.chat.api import create_app
from history_chatbot.chat.demo_journey import InMemoryDemoJourneyProvider
from history_chatbot.chat.session import SessionStore
from history_chatbot.runtime import RuntimeMode


SESSION_ID = "a" * 32


class _Sessions:
    def create(self, locale: str):
        return type("Session", (), {"session_id": SESSION_ID, "locale": locale})()


class _Service:
    orchestrator = type("Orchestrator", (), {"sessions": _Sessions()})()

    def chat(self, _payload: dict[str, object]) -> dict[str, object]:
        return {
            "primary_situation_id": "FREE_CHAT_GREETING",
            "ui_state": "responding",
            "context_state": [],
        }


def _client():
    fastapi = pytest.importorskip("fastapi")
    assert fastapi
    from fastapi.testclient import TestClient

    return TestClient(create_app(_Service(), InMemoryDemoJourneyProvider()))


def test_normal_piece_chat_request() -> None:
    request = PieceChatRequest.model_validate({
        "session_id": SESSION_ID,
        "user_message": "  인상 깊었어요.  ",
        "ui_state": "awaiting_reflection",
    })
    assert request.resolved_session_id() == SESSION_ID
    assert request.resolved_message() == "인상 깊었어요."


def test_normal_free_chat_request() -> None:
    request = FreeChatRequest.model_validate({
        "session_id": SESSION_ID,
        "user_message": "목포역의 역사를 알려줘",
        "locale": "zh-CN",
    })
    assert request.resolved_message() == "목포역의 역사를 알려줘"
    assert request.resolved_locale("ko") == "zh-CN"


@pytest.mark.parametrize("payload", [
    {"session_id": SESSION_ID},
    {"session_id": SESSION_ID, "user_message": "   "},
])
def test_missing_or_blank_message_is_a_domain_error(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="user_message"):
        PieceChatRequest.model_validate(payload).resolved_message()


def test_non_string_message_is_rejected_as_a_type_error() -> None:
    with pytest.raises(ValidationError):
        PieceChatRequest.model_validate({"session_id": SESSION_ID, "user_message": 123})


def test_too_long_message_is_a_domain_error() -> None:
    request = PieceChatRequest(session_id=SESSION_ID, user_message="가" * (MAX_MESSAGE_LENGTH + 1))
    with pytest.raises(ValueError, match="2,000"):
        request.resolved_message()


def test_session_id_type_and_format_are_distinct() -> None:
    with pytest.raises(ValidationError):
        PieceChatRequest.model_validate({"session_id": 123, "user_message": "질문"})
    with pytest.raises(ValueError, match="session_id"):
        PieceChatRequest(session_id="invalid", user_message="질문").resolved_session_id()


def test_actual_server_generated_session_id_is_accepted() -> None:
    session_id = SessionStore(RuntimeMode.TEST).create().session_id
    assert len(session_id) == 32
    assert session_id == session_id.lower()
    assert PieceChatRequest(
        session_id=session_id, user_message="질문",
    ).resolved_session_id() == session_id


def test_locale_default_and_invalid_format() -> None:
    assert SessionCreateRequest().resolved_locale() == "ko"
    with pytest.raises(ValueError, match="locale"):
        SessionCreateRequest(locale="unsupported").resolved_locale()


def test_optional_fields_have_frontend_compatible_defaults() -> None:
    request = PieceChatRequest(session_id=SESSION_ID, user_message="질문")
    assert request.locale is None
    assert request.ui_state is None
    assert request.return_target == "journey"


def test_existing_frontend_payloads_are_accepted() -> None:
    piece = PieceChatRequest.model_validate({
        "session_id": SESSION_ID,
        "user_message": "인상 깊었어요.",
        "ui_state": "awaiting_reflection",
    })
    free = FreeChatRequest.model_validate({
        "session_id": SESSION_ID,
        "user_message": "안녕하세요",
        "ui_state": "active",
    })
    transition = TransitionRequest.model_validate({
        "session_id": SESSION_ID,
        "from_mode": "piece_chat",
        "to_mode": "free_chat",
        "mode_transition": {"pending_user_question": "질문"},
    })
    action = JourneyActionRequest.model_validate({
        "session_id": SESSION_ID,
        "action_code": "GO_NEXT_PIECE",
    })
    assert piece.resolved_message() and free.resolved_message()
    assert transition.mode_transition == {"pending_user_question": "질문"}
    assert action.action_code == "GO_NEXT_PIECE"


def test_pending_question_only_and_message_precedence_match_existing_behavior() -> None:
    pending_only = PieceChatRequest(
        session_id=SESSION_ID, pending_user_question="복원할 질문",
    )
    both = PieceChatRequest(
        session_id=SESSION_ID,
        user_message="현재 질문",
        pending_user_question="이전 질문",
    )
    assert pending_only.resolved_message() == "복원할 질문"
    assert both.resolved_message() == "현재 질문"


def test_additive_client_fields_remain_compatible() -> None:
    request = PieceChatRequest.model_validate({
        "session_id": SESSION_ID,
        "user_message": "질문",
        "future_client_field": "preserved",
    })
    assert request.model_dump()["future_client_field"] == "preserved"


def test_generic_chat_validates_shared_session_locale_and_length() -> None:
    request = GenericChatRequest(
        session_id=SESSION_ID, user_query="질문", locale="ko",
    )
    assert request.service_payload()["session_id"] == SESSION_ID
    with pytest.raises(ValueError, match="locale"):
        GenericChatRequest(locale="unsupported").service_payload()
    with pytest.raises(ValueError, match="2,000"):
        GenericChatRequest(user_query="가" * (MAX_MESSAGE_LENGTH + 1)).service_payload()


def test_http_validation_preserves_400_domain_and_422_type_errors() -> None:
    client = _client()
    session = client.post("/api/session", json={"locale": "ko"})
    assert session.status_code == 200

    missing = client.post("/api/chat/piece", json={"session_id": SESSION_ID})
    wrong_type = client.post(
        "/api/chat/piece", json={"session_id": SESSION_ID, "user_message": 123},
    )
    invalid_locale = client.post("/api/session", json={"locale": "unsupported"})
    missing_session = client.post(
        "/api/chat/piece", json={"user_message": "질문"},
    )
    invalid_session_format = client.post(
        "/api/chat/piece", json={"session_id": "invalid", "user_message": "질문"},
    )
    invalid_session_type = client.post(
        "/api/chat/piece", json={"session_id": 123, "user_message": "질문"},
    )

    assert missing.status_code == 400
    assert missing.json()["error_code"] == "invalid_request"
    assert wrong_type.status_code == 422
    assert wrong_type.json()["error_code"] == "invalid_payload"
    assert invalid_locale.status_code == 400
    assert missing_session.status_code == 400
    assert missing_session.json()["error_code"] == "invalid_request"
    assert invalid_session_format.status_code == 400
    assert invalid_session_type.status_code == 422


def test_http_frontend_piece_payload_remains_compatible() -> None:
    client = _client()
    client.post("/api/session", json={"locale": "ko"})
    response = client.post("/api/chat/piece", json={
        "session_id": SESSION_ID,
        "user_message": "인상 깊었어요.",
        "ui_state": "awaiting_reflection",
    })
    assert response.status_code == 200
    assert response.json()["situation_id"] == "FREE_CHAT_GREETING"
