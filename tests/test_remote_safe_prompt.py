from __future__ import annotations

import json

import pytest

from history_chatbot.chat import orchestrator as orchestrator_module
from history_chatbot.chat.remote_safe import (
    RemotePromptPolicy,
    sanitize_remote_text,
    serialize_remote_prompt,
)
from history_chatbot.chat.service import create_development_real_service
from history_chatbot.models.remote import (
    HttpResponse,
    OpenAICompatibleAdapter,
    RemoteLLMBackend,
    RemoteLLMConfig,
)
from history_chatbot.models.contract import LLMRequest
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.runtime import RuntimeMode


class CaptureTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], bytes | None, float]] = []

    def request(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        return HttpResponse(
            200,
            json.dumps(
                {
                    "id": "anonymous-response",
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "확인된 근거에 따른 답변이야."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

    def stream(self, method, url, headers, body, timeout):  # pragma: no cover
        raise AssertionError("이 테스트에서는 stream을 열지 않습니다.")


def ranked(
    document_id: str,
    chunk_id: str,
    text: str,
    *,
    title: str = "목포진",
    publisher: str = "목포시",
) -> RankedChunk:
    return RankedChunk(
        RetrievalChunk(
            document_id,
            chunk_id,
            text,
            title,
            publisher,
            "https://internal.example/citation",
            {
                "production_approved": False,
                "allowed_for_rag": True,
                "license_review_status": "pending_review",
                "reviewed_by": "reviewer",
            },
        ),
        0.9,
        ("sparse", "dense"),
    )


def remote_backend(transport: CaptureTransport, **overrides) -> RemoteLLMBackend:
    values = {
        "api_format": "openai",
        "base_url": "http://localhost:8001",
        "model": "test-model",
        "readiness_probe_enabled": False,
        "max_retries": 0,
    }
    values.update(overrides)
    return RemoteLLMBackend(
        RemoteLLMConfig(**values),
        mode=RuntimeMode.DEVELOPMENT,
        transport=transport,
    )


def serialized_body(transport: CaptureTransport) -> str:
    assert len(transport.calls) == 1
    body = transport.calls[0][3]
    assert body is not None
    return body.decode("utf-8")


def test_serializer_anonymizes_evidence_and_sanitizes_private_values() -> None:
    session_id = "0123456789abcdef0123456789abcdef"
    result = serialize_remote_prompt(
        system_prompt="근거 안에서만 답하세요.",
        user_query=(
            f"연락처 user@example.com, 010-1234-5678, {session_id}, "
            "C:\\private\\records\\item.json, http://127.0.0.1:8000"
        ),
        chunks=(
            ranked("mokpo_hist_0005", "mokpo_hist_0005::0000", "첫 번째 근거"),
            ranked("mokpo_hist_0006", "mokpo_hist_0006::0000", "두 번째 근거"),
        ),
        policy=RemotePromptPolicy(context_max_chars=4_000),
    )
    combined = result.system_prompt + result.user_prompt

    assert "[자료1]" in combined and "[자료2]" in combined
    assert combined.index("첫 번째 근거") < combined.index("두 번째 근거")
    for forbidden in (
        "mokpo_hist_0005",
        "mokpo_hist_0006",
        "document_id",
        "chunk_id",
        "internal.example",
        session_id,
        "user@example.com",
        "010-1234-5678",
        "C:\\private",
        "127.0.0.1",
    ):
        assert forbidden not in combined


def test_history_is_bounded_and_not_duplicated_in_user_prompt() -> None:
    result = serialize_remote_prompt(
        system_prompt="시스템",
        user_query="현재 질문",
        chunks=(ranked("doc-1", "chunk-1", "근거"),),
        history=(("오래된 질문", "오래된 답변"), ("최근 질문", "최근 답변")),
        policy=RemotePromptPolicy(
            history_enabled=True, history_max_turns=1, context_max_chars=2_000
        ),
    )

    assert [message.content for message in result.messages] == ["최근 질문", "최근 답변"]
    assert "최근 질문" not in result.user_prompt
    assert "오래된 질문" not in result.user_prompt


def test_context_and_evidence_limits_are_deterministic() -> None:
    result = serialize_remote_prompt(
        system_prompt="시스템 지침",
        user_query="현재 질문",
        chunks=tuple(
            ranked(f"doc-{index}", f"chunk-{index}", str(index) * 500)
            for index in range(1, 5)
        ),
        policy=RemotePromptPolicy(
            context_max_chars=1_024,
            chunk_max_chars=180,
            max_evidence_items=2,
        ),
    )

    assert result.evidence_items == 2
    assert "[자료3]" not in result.user_prompt
    assert result.total_chars <= 1_024


@pytest.mark.parametrize(
    "overrides",
    [
        {"remote_history_max_turns": -1},
        {"remote_context_max_chars": 500},
        {"remote_chunk_max_chars": 20},
        {"remote_max_evidence_items": 0},
    ],
)
def test_remote_prompt_configuration_rejects_unsafe_values(overrides) -> None:
    with pytest.raises(ValueError):
        RemoteLLMConfig(
            api_format="openai",
            base_url="http://localhost:8001",
            model="test-model",
            **overrides,
        ).validate(RuntimeMode.DEVELOPMENT)


def test_remote_prompt_environment_defaults_are_safe() -> None:
    config = RemoteLLMConfig.from_environment(
        RuntimeMode.DEVELOPMENT,
        {
            "LLM_BASE_URL": "http://localhost:8001",
            "LLM_MODEL": "test-model",
        },
    )

    assert config.remote_history_enabled is False
    assert config.remote_history_max_turns == 1
    assert config.remote_context_max_chars == 12_000
    assert config.remote_chunk_max_chars == 1_600
    assert config.remote_max_evidence_items == 4
    assert config.remote_sanitize_enabled is True


def test_streaming_adapter_uses_the_same_safe_serialized_content() -> None:
    safe = serialize_remote_prompt(
        system_prompt="시스템",
        user_query="질문",
        chunks=(ranked("mokpo_hist_0005", "mokpo_hist_0005::0000", "근거"),),
    )
    config = RemoteLLMConfig(
        api_format="openai",
        base_url="http://localhost:8001",
        model="test-model",
    )
    payload = OpenAICompatibleAdapter().payload(
        LLMRequest(
            safe.system_prompt,
            safe.user_prompt,
            messages=safe.messages,
            stream=True,
        ),
        config,
    )
    body = json.dumps(payload, ensure_ascii=False)

    assert payload["stream"] is True
    assert "[자료1]" in body
    assert "mokpo_hist_0005" not in body


def test_final_openai_body_excludes_local_metadata_and_session_id() -> None:
    transport = CaptureTransport()
    service = create_development_real_service(llm=remote_backend(transport))
    session = service.orchestrator.sessions.create()
    response = service.chat(
        {
            "session_id": session.session_id,
            "user_query": "목포진은 언제 설치되고 폐지되었나요?",
            "conversation_mode": "free_chat",
        }
    )
    body = serialized_body(transport)

    assert response["request_state"] == "success"
    assert {item["document_id"] for item in response["citations"]} == {"mokpo_hist_0005"}
    assert "[PRIMARY FACT]" in body
    for forbidden in (
        "document_id",
        "chunk_id",
        "mokpo_hist_0005",
        "citation_url",
        "production_approved",
        "allowed_for_rag",
        "license_review_status",
        session.session_id,
        "biz.mokpo.go.kr",
    ):
        assert forbidden not in body


def test_insufficient_evidence_never_serializes_or_calls_remote(monkeypatch) -> None:
    transport = CaptureTransport()
    service = create_development_real_service(llm=remote_backend(transport))
    session = service.orchestrator.sessions.create()

    def unexpected_serializer(**_kwargs):
        raise AssertionError("근거 부족 요청은 remote-safe prompt를 만들면 안 됩니다.")

    monkeypatch.setattr(orchestrator_module, "serialize_remote_prompt", unexpected_serializer)
    response = service.chat(
        {
            "session_id": session.session_id,
            "user_query": "경동성당은 언제 건립되었나요?",
            "conversation_mode": "free_chat",
        }
    )

    assert response["request_state"] == "insufficient_evidence"
    assert response["citations"] == ()
    assert transport.calls == []


def test_sanitizer_does_not_expose_secret_assignment() -> None:
    cleaned = sanitize_remote_text("API_KEY=not-a-real-secret 일반 역사 문장")
    assert "not-a-real-secret" not in cleaned
    assert "일반 역사 문장" in cleaned


def test_entity_only_prefers_identity_over_neighbor_context() -> None:
    photo = ranked(
        "photo", "photo-1",
        "목포근대역사관 1관 앞 항공사진입니다. 정면 사진입니다. 좌측면 사진입니다.",
        title="현대사 사진 아카이브",
    )
    identity = ranked(
        "official", "official-1",
        "목포근대역사관 1관은 구 목포 일본영사관 건물을 활용한 역사 전시 시설입니다.",
        title="목포근대역사관 1관",
    )

    prompt = serialize_remote_prompt(
        system_prompt="기록으로 답하세요.",
        user_query="목포근대역사관 1관은 어떤 곳이야?",
        chunks=(photo, identity),
        question_subject="목포근대역사관 1관",
        question_intent="overview",
    )

    assert "역사 전시 시설" in prompt.user_prompt
    assert "좌측면 사진" not in prompt.user_prompt
