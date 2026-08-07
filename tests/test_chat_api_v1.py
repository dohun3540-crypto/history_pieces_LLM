from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from history_chatbot.chat.api import create_app
from history_chatbot.chat.demo_journey import InMemoryDemoJourneyProvider
from history_chatbot.chat.service import create_development_integration_service
from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.mock_llm import MockLLM


class CapturingMockLLM(MockLLM):
    def __init__(self) -> None:
        super().__init__("확인 가능한 자료가 부족합니다.")
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest):
        self.requests.append(request)
        return super().complete(request)


def client(tmp_path: Path) -> tuple[TestClient, CapturingMockLLM]:
    llm = CapturingMockLLM()
    service = create_development_integration_service(
        runtime_dir=tmp_path / "runtime",
        llm=llm,
    )
    app = create_app(service, InMemoryDemoJourneyProvider())
    return TestClient(app), llm


def test_v1_chat_uses_history_for_interpretation_and_retrieved_chunks_for_grounding(
    tmp_path: Path,
) -> None:
    api, llm = client(tmp_path)

    response = api.post(
        "/api/v1/chat",
        json={
            "message": "그 건물은 무엇인가요?",
            "history": [
                {"role": "user", "content": "가상 해솔관을 알려줘"},
                {"role": "assistant", "content": "자료를 확인해 볼게요."},
            ],
        },
    )

    assert response.status_code == 200
    assert set(response.json()) == {"answer"}
    assert response.json()["answer"]
    assert len(llm.requests) == 1
    request = llm.requests[0]
    assert request.metadata["evidence"]
    assert "가상 해솔관" in " ".join(request.metadata["evidence"])
    assert [(item.role, item.content) for item in request.messages] == [
        ("user", "가상 해솔관을 알려줘"),
        ("assistant", "자료를 확인해 볼게요."),
    ]
    assert "이전 대화는 후속 질문 해석에만 사용" in request.system_prompt


def test_v1_chat_falls_back_without_calling_llm_when_evidence_is_missing(
    tmp_path: Path,
) -> None:
    api, llm = client(tmp_path)

    response = api.post(
        "/api/v1/chat",
        json={"message": "서울 권궐의 정확한 개수는?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "제공된 역사 자료에서 충분한 근거를 찾지 못했습니다."
    }
    assert llm.requests == []


def test_v1_search_and_readiness_keep_generation_separate(tmp_path: Path) -> None:
    api, llm = client(tmp_path)

    search = api.post(
        "/api/v1/search",
        json={"query": "가상 청해항 붉은 등대 전시관", "top_k": 1},
    )
    ready = api.get("/ready")
    ready_alias = api.get("/api/v1/ready")

    assert search.status_code == 200
    assert len(search.json()["results"]) == 1
    assert "붉은 등대 전시관" in search.json()["results"][0]["text"]
    assert llm.requests == []
    assert ready.status_code == 200
    assert ready.json() == ready_alias.json()
    assert ready.json()["ready"] is True
    assert ready.json()["index_loaded"] is True
    assert ready.json()["retriever"] is True
    assert ready.json()["llm"] is True
    assert ready.json()["backend"] == "mock"


def test_v1_contract_rejects_blank_message_and_extra_fields(tmp_path: Path) -> None:
    api, _llm = client(tmp_path)

    blank = api.post("/api/v1/chat", json={"message": "   "})
    extra = api.post(
        "/api/v1/chat",
        json={"message": "질문", "unexpected": True},
    )

    assert blank.status_code == 422
    assert extra.status_code == 422
