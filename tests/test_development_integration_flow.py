import shutil
import uuid
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from history_chatbot.chat.api import create_app
from history_chatbot.chat.demo_journey import InMemoryDemoJourneyProvider
from history_chatbot.chat.integration_app import create_integration_app
from history_chatbot.chat.service import create_development_integration_service
from history_chatbot.models.mock_llm import MockLLM


class CountingMockLLM(MockLLM):
    def __init__(self) -> None:
        super().__init__("확인 가능한 자료가 부족합니다.")
        self.completions = 0

    def complete(self, request):
        self.completions += 1
        return super().complete(request)


def test_fixture_session_rag_llm_citation_and_insufficient_flow() -> None:
    runtime_dir = Path(".runtime/integration-tests") / uuid.uuid4().hex
    llm = CountingMockLLM()
    try:
        service = create_development_integration_service(
            runtime_dir=runtime_dir,
            llm=llm,
        )
        client = TestClient(
            create_app(service=service, journey_provider=InMemoryDemoJourneyProvider())
        )

        session_response = client.post("/api/session", json={"locale": "ko"})
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]
        assert len(session_id) == 32

        piece = client.post(
            "/api/chat/piece",
            json={"session_id": session_id, "user_message": "조금 피곤해"},
        )
        assert piece.status_code == 200
        assert piece.json()["rag_used"] is False
        assert piece.json()["citations"] == []

        factual = client.post(
            "/api/chat/free",
            json={
                "session_id": session_id,
                "user_message": "가상 청해항의 붉은 등대 전시관을 알려줘",
                "pending_user_question": "서울 궁궐의 정확한 개수는?",
            },
        )
        assert factual.status_code == 200
        body = factual.json()
        assert body["request_state"] == "success"
        assert body["rag_used"] is True
        assert body["grounded"] is True
        assert body["retrieved_chunk_ids"]
        assert body["citations"]
        assert body["citations"][0]["chunk_id"] in body["retrieved_chunk_ids"]
        assert body["citations"][0]["source_url"].startswith("https://example.invalid/")
        assert body["citations"][0]["is_fixture"] is True
        assert llm.completions == 1

        insufficient = client.post(
            "/api/chat/free",
            json={
                "session_id": session_id,
                "user_message": "서울 궁궐의 정확한 개수는?",
            },
        )
        assert insufficient.status_code == 200
        missing = insufficient.json()
        assert missing["request_state"] == "insufficient_evidence"
        assert missing["source_sufficiency"] == "insufficient"
        assert missing["citations"] == []
        assert missing["retrieved_chunk_ids"] == []
        assert llm.completions == 1

        invalid = client.post(
            "/api/chat/free",
            json={"session_id": "not-a-session", "user_message": "질문"},
        )
        assert invalid.status_code in {400, 422}
        assert invalid.json()["request_state"] == "error"
    finally:
        if runtime_dir.is_dir():
            shutil.rmtree(runtime_dir)


def test_frontend_contract_matches_integrated_api_response() -> None:
    script = Path("src/history_chatbot/web/static/app.js").read_text(encoding="utf-8")

    for endpoint in ("/api/session", "/api/chat/piece", "/api/chat/free"):
        assert endpoint in script
    assert "user_message" in script
    assert "renderCitations(result.citations)" in script
    assert "insufficient_evidence" in script
    assert ".innerHTML" not in script
    assert "textContent" in script
    assert callable(create_integration_app)
