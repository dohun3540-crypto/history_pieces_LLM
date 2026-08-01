from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from history_chatbot.chat.api import create_app
from history_chatbot.chat.demo_journey import InMemoryDemoJourneyProvider
from history_chatbot.chat.service import ChatApplicationService, create_development_orchestrator


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    orchestrator = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "chat-sessions.json",
    )
    app = create_app(ChatApplicationService(orchestrator), InMemoryDemoJourneyProvider())
    return TestClient(app)


def new_session(client: TestClient) -> dict[str, object]:
    response = client.post("/api/session", json={"locale": "ko"})
    assert response.status_code == 200
    return response.json()


def test_health_and_main_static_assets(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok", "service": "history-pieces",
        "chat_modes": ["piece_chat", "free_chat"],
    }
    html = client.get("/")
    assert html.status_code == 200 and "History Pieces" in html.text
    assert "IMAGE ASSET PLACEHOLDER" in html.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_session_create_lookup_and_missing(client: TestClient) -> None:
    session = new_session(client)
    assert session["current_piece_id"] == "demo-piece-1"
    assert session["completed_piece_ids"] == []
    assert session["ephemeral"] is True
    found = client.get(f"/api/session/{session['session_id']}")
    assert found.status_code == 200
    missing = client.get("/api/session/00000000000000000000000000000000")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "session_not_found"


def test_piece_emotion_and_fatigue_are_no_rag(client: TestClient) -> None:
    session = new_session(client)
    emotion = client.post("/api/chat/piece", json={
        "session_id": session["session_id"], "user_message": "인상 깊었어요.",
        "ui_state": "awaiting_reflection",
    })
    assert emotion.status_code == 200
    assert emotion.json()["chat_mode"] == "piece_chat"
    assert "piece_ui_state" in emotion.json()
    assert emotion.json()["rag_used"] is False
    assert emotion.json()["game_state_mutation"] is False
    assert emotion.json()["storage_permitted"] is False
    fatigue = client.post("/api/chat/piece", json={
        "session_id": session["session_id"], "user_message": "조금 피곤해요.",
    })
    assert fatigue.status_code == 200
    assert fatigue.json()["next_action_code"] == "PAUSE_JOURNEY"
    assert fatigue.json()["rag_used"] is False


def test_piece_detail_returns_preserved_free_chat_transition(client: TestClient) -> None:
    session = new_session(client)
    question = "이 장소의 역사를 출처와 함께 자세히 알려주세요."
    response = client.post("/api/chat/piece", json={
        "session_id": session["session_id"], "user_message": question,
        "return_target": "journey-card",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["next_action_code"] == "OPEN_FREE_CHAT"
    assert body["mode_transition"]["pending_user_question"] == question
    assert body["mode_transition"]["current_piece_id"] == "demo-piece-1"
    assert body["mode_transition"]["return_target"] == "journey-card"
    assert body["game_state_mutation"] is False


def test_free_chat_rag_citations_greeting_and_insufficient_evidence(client: TestClient) -> None:
    session = new_session(client)
    session_id = session["session_id"]
    factual = client.post("/api/chat/free", json={
        "session_id": session_id, "user_message": "붉은 등대 전시관은 언제 만들어졌어요?",
    })
    assert factual.status_code == 200
    assert factual.json()["rag_used"] is True
    assert "free_ui_state" in factual.json()
    assert factual.json()["citations"]
    greeting = client.post("/api/chat/free", json={"session_id": session_id, "user_message": "안녕하세요"})
    assert greeting.status_code == 200
    assert greeting.json()["rag_used"] is False and greeting.json()["citations"] == []
    missing = client.post("/api/chat/free", json={
        "session_id": session_id, "user_message": "서울 궁궐의 왕은 누구야?",
    })
    assert missing.status_code == 200
    assert missing.json()["request_state"] == "insufficient_evidence"
    assert missing.json()["citations"] == []


def test_free_chat_does_not_complete_piece_and_return_preserves_state(client: TestClient) -> None:
    session = new_session(client)
    session_id = session["session_id"]
    opened = client.post("/api/chat/transition", json={
        "session_id": session_id, "from_mode": "piece_chat", "to_mode": "free_chat",
        "mode_transition": {"pending_user_question": "원 질문"},
    })
    assert opened.status_code == 200
    client.post("/api/chat/free", json={"session_id": session_id, "user_message": "안녕하세요"})
    returned = client.post("/api/chat/transition", json={
        "session_id": session_id, "from_mode": "free_chat", "to_mode": "game",
    })
    body = returned.json()
    assert body["action_code"] == "RETURN_TO_GAME"
    assert body["game_state_mutation"] is False
    assert body["session"]["current_piece_id"] == "demo-piece-1"
    assert body["session"]["completed_piece_ids"] == []


def test_go_next_requires_explicit_action(client: TestClient) -> None:
    session = new_session(client)
    session_id = session["session_id"]
    client.post("/api/chat/piece", json={"session_id": session_id, "user_message": "다음으로 갈까요?"})
    unchanged = client.get(f"/api/session/{session_id}").json()
    assert unchanged["current_piece_id"] == "demo-piece-1"
    moved = client.post("/api/journey/action", json={
        "session_id": session_id, "action_code": "GO_NEXT_PIECE",
    })
    assert moved.status_code == 200
    assert moved.json()["game_state_mutation"] is True
    assert moved.json()["session"]["current_piece_id"] == "demo-piece-2"
    assert moved.json()["session"]["completed_piece_ids"] == ["demo-piece-1"]


def test_unsupported_action_and_invalid_payload_are_structured(client: TestClient) -> None:
    session = new_session(client)
    unsupported = client.post("/api/journey/action", json={
        "session_id": session["session_id"], "action_code": "SAVE_SHORT_REFLECTION",
    })
    assert unsupported.status_code == 409
    assert unsupported.json()["error_code"] == "capability_unavailable"
    assert unsupported.json()["request_state"] == "error"
    invalid = client.post("/api/chat/piece", json={"session_id": session["session_id"]})
    assert invalid.status_code == 400
    assert invalid.json()["error_code"] == "invalid_request"
    malformed = client.post("/api/chat/piece", json=["not", "an", "object"])
    assert malformed.status_code == 422
    assert malformed.json()["error_code"] == "invalid_payload"


def test_client_state_uses_safe_dom_and_guarded_ui_logic(client: TestClient) -> None:
    script = client.get("/static/app.js").text
    assert ".innerHTML" not in script and "textContent" in script
    assert 'state.request==="loading"' in script
    assert "renderCitations(result.citations)" in script
    assert 'toggle.hidden = valid.length === 0' in script
    assert 'action_code:action' in script
    assert 'to_mode:"game"' in script
    assert "IMAGE ASSET" not in script
