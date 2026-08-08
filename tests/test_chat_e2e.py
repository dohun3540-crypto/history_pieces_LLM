from pathlib import Path

import pytest

from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.prompt_builder import SYSTEM_INSTRUCTIONS, build_prompt
from history_chatbot.chat.service import (
    ChatApplicationService,
    create_development_orchestrator,
)
from history_chatbot.chat.session import SessionStore
from history_chatbot.models.factory import build_llm_backend
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.runtime import RuntimeMode


def orchestrator(tmp_path: Path) -> ConversationalRagOrchestrator:
    return create_development_orchestrator(
        runtime_dir=tmp_path / "runtime",
        session_path=tmp_path / "sessions.json",
    )


def test_single_question_end_to_end_with_fixture_source(tmp_path) -> None:
    response = orchestrator(tmp_path).ask("붉은 등대 전시관을 알려줘")
    assert response.status == "ok"
    assert response.used_chunks >= 1
    assert response.answer.startswith("[테스트용 응답]")
    assert response.sources
    source = response.sources[0]
    assert source.is_fixture
    assert source.source_id
    assert source.document_id
    assert source.title
    assert source.institution
    assert source.source_url
    assert source.chunk_id
    assert source.excerpt
    assert isinstance(source.retrieval_score, float)
    assert source.license_status == "open_license"


def test_followup_question_keeps_session_context(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    first = chat.ask("가상 해솔관을 알려줘")
    second = chat.ask("그 건물은 어떤 설정이야?", session_id=first.session_id)
    session = chat.sessions.get(first.session_id)
    assert second.session_id == first.session_id
    assert second.status == "ok"
    assert session is not None and len(session.turns) == 2


def test_explicit_people_followup_reuses_previous_retrieval_query(tmp_path, monkeypatch) -> None:
    chat = orchestrator(tmp_path)
    queries: list[str] = []
    original_search = chat.retrieval.search

    def capture_search(query: str):
        queries.append(query)
        return original_search(query)

    monkeypatch.setattr(chat.retrieval, "search", capture_search)
    first = chat.ask("붉은 등대 전시관은 언제 만들어졌어요?")
    chat.ask("관련 인물은 누구인가요?", session_id=first.session_id)

    assert queries[-1] == "붉은 등대 전시관은 언제 만들어졌어요? 관련 인물은 누구인가요?"


def test_independent_question_does_not_reuse_previous_retrieval_query(tmp_path, monkeypatch) -> None:
    chat = orchestrator(tmp_path)
    queries: list[str] = []
    original_search = chat.retrieval.search

    def capture_search(query: str):
        queries.append(query)
        return original_search(query)

    monkeypatch.setattr(chat.retrieval, "search", capture_search)
    first = chat.ask("붉은 등대 전시관은 언제 만들어졌어요?")
    chat.ask("가상 해솔관은 어떤 건물인가요?", session_id=first.session_id)

    assert queries[-1] == "가상 해솔관은 어떤 건물인가요?"


def test_missing_evidence_never_calls_grounded_generation(tmp_path, monkeypatch) -> None:
    chat = orchestrator(tmp_path)

    def forbidden(**kwargs):
        raise AssertionError("근거 없이 LLM을 호출하면 안 됩니다.")

    monkeypatch.setattr(chat.llm, "generate_grounded", forbidden)
    response = chat.ask("서울 궁궐의 왕은 누구야?")
    assert response.answer == "현재 검수된 자료만으로는 확인할 수 없습니다."
    assert response.status == "insufficient_evidence"
    assert response.sources == ()
    assert response.used_chunks == 0


@pytest.mark.parametrize(
    "followup",
    ("관련 인물은 누구인가요?", "관련된 사람은?", "누가 참여했나요?"),
)
def test_explicit_people_followup_forms_are_bounded(followup: str) -> None:
    previous = "목포역 학생운동은 어떻게 전개되었나요?"
    assert ConversationalRagOrchestrator._rewrite_followup(followup, previous) == (
        f"{previous} {followup}"
    )


def test_session_create_lookup_limit_reset_and_unknown_reset(tmp_path) -> None:
    store = SessionStore(
        RuntimeMode.DEVELOPMENT,
        path=tmp_path / "sessions.json",
        max_turns=2,
    )
    session = store.create()
    for index in range(3):
        store.add_turn(session.session_id, f"질문 {index}", f"응답 {index}")
    assert store.get(session.session_id) is not None
    assert len(store.get(session.session_id).turns) == 2  # type: ignore[union-attr]
    assert store.get(session.session_id).summary  # type: ignore[union-attr]
    assert store.reset(session.session_id)
    assert store.get(session.session_id) is None
    assert not store.reset("missing-session")


def test_stream_has_complete_event_with_sources(tmp_path) -> None:
    events = list(orchestrator(tmp_path).stream("붉은 등대 전시관"))
    assert any(event.event == "token" for event in events)
    assert events[-1].event == "completed"
    assert events[-1].data["status"] == "ok"
    assert events[-1].data["sources"]


def ranked(document_id: str, chunk_id: str, score: float = 0.9) -> RankedChunk:
    chunk = RetrievalChunk(
        document_id,
        chunk_id,
        "테스트용 가상 자료이며 실제 역사 사실이 아님",
        "테스트용 가상 자료",
        "테스트 기관",
        "https://example.invalid/source",
        {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "text": "테스트용 가상 자료이며 실제 역사 사실이 아님",
            "title": "테스트용 가상 자료",
            "publisher": "테스트 기관",
            "source_url": "https://example.invalid/source",
            "data_classification": "fictional_fixture",
            "copyright_status": "open_license",
        },
    )
    return RankedChunk(chunk, score, ("dense", "sparse"), score, score)


def test_duplicate_chunks_and_per_document_limit_are_enforced(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    results = [
        ranked("doc-1", "chunk-1"),
        ranked("doc-1", "chunk-1"),
        ranked("doc-1", "chunk-2"),
        ranked("doc-1", "chunk-3"),
        ranked("doc-2", "chunk-4"),
    ]
    selected = chat._select(results, top_k=5)
    assert [item.chunk.chunk_id for item in selected] == ["chunk-1", "chunk-2", "chunk-4"]
    from history_chatbot.chat.citation_builder import build_citations

    assert len(build_citations(selected)) == 2


def test_prompt_has_all_boundaries_and_no_guessing_rule() -> None:
    prompt = build_prompt(
        user_query="질문",
        conversation_summary="이전 대화",
        chunks=[ranked("doc", "chunk")],
        locale="ko",
    )
    assert "[시스템 지침" in prompt
    assert "[이전 대화 요약]" in prompt
    assert "[검색 근거]" in prompt
    assert "[사용자 질문]" in prompt
    assert "근거에 없는 내용은 추측" in SYSTEM_INSTRUCTIONS
    assert "개발 fixture는 실제 역사 사실이 아니다" in prompt


def test_production_blocks_fixture_and_mock(tmp_path) -> None:
    with pytest.raises(ValueError, match="production"):
        build_llm_backend(
            "mock",
            runtime_mode=RuntimeMode.PRODUCTION,
            fallback_message="fallback",
        )
    with pytest.raises(ValueError, match="production"):
        ConversationalRagOrchestrator(
            retrieval=object(),  # type: ignore[arg-type]
            llm=MockLLM("fallback"),
            sessions=SessionStore(RuntimeMode.PRODUCTION),
            mode=RuntimeMode.PRODUCTION,
        )


def test_application_service_readiness_and_reset(tmp_path) -> None:
    app_service = ChatApplicationService(orchestrator(tmp_path))
    response = app_service.chat({"user_query": "붉은 등대 전시관"})
    assert app_service.health() == {"status": "ok"}
    assert app_service.readiness()["status"] == "development_ready"
    reset = app_service.reset(str(response["session_id"]))
    assert reset["reset"] is True


def test_non_rag_greeting_returns_dialogue_contract_without_llm_or_sources(tmp_path) -> None:
    response = orchestrator(tmp_path).ask("안녕하세요", conversation_mode="free_chat")
    payload = response.to_dict()
    assert payload["primary_situation_id"] == "FREE_CHAT_GREETING"
    assert payload["conversation_mode"] == "free_chat"
    assert payload["grounded"] is False
    assert payload["retrieved_chunk_ids"] == ()
    assert payload["citations"] == ()
    assert payload["model_backend"] == "mock"
    assert payload["embedding_backend"] == "hashing-v1"


def test_journey_prompt_contains_only_completed_piece_ids(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    response = chat.ask(
        "방금 본 조각이랑 이전 조각은 무슨 관계예요?",
        conversation_mode="piece_chat",
        screen_type="piece_chat",
        visited_piece_ids=("piece-1", "piece-2"),
    )
    assert "piece-3" not in response.answer
