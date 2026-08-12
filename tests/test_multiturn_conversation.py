from __future__ import annotations

from pathlib import Path

import pytest

from history_chatbot.chat.context_resolver import (
    ConversationContextResolver,
    ConversationRequestKind,
)
from history_chatbot.chat.service import create_development_orchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.chat.session import SessionTurn
from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.runtime import RuntimeMode


class CapturingLLM(MockLLM):
    def __init__(self) -> None:
        super().__init__("근거 부족")
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest):
        self.requests.append(request)
        return super().complete(request)


def _ranked(chunk_id: str, text: str, *, title: str = "목포역 행사 기록") -> RankedChunk:
    payload = {
        "document_id": f"doc-{chunk_id}",
        "chunk_id": chunk_id,
        "text": text,
        "title": title,
        "publisher": "테스트 기관",
        "source_url": f"https://example.invalid/{chunk_id}",
        "data_classification": "fictional_fixture",
        "copyright_status": "open_license",
    }
    chunk = RetrievalChunk(
        payload["document_id"], chunk_id, text, title, "테스트 기관",
        payload["source_url"], payload,
    )
    return RankedChunk(chunk, 0.9, ("test",), 0.9, 0.9)


RICH_EVIDENCE = _ranked(
    "rich",
    "1949년 6월 27일 목포역에서 행사가 열렸다. 이범석과 안호상이 참석했다. "
    "이 행사는 지역 교통망을 알리고 시민을 격려하기 위한 배경에서 개최되었다. "
    "참석자들은 목포역에 도착해 환영 행사와 연설을 진행했다. 행사 이후 관련 활동이 "
    "이어졌고 지역 사회에 영향을 남겼다. 당시 과정과 결과는 여러 기록에 남아 있다. "
    "행사의 시점, 장소, 주요 인물, 배경, 진행 과정과 결과를 함께 설명하는 상세 기록이다.",
)
SHORT_EVIDENCE = _ranked(
    "short", "1949년 6월 27일 목포역에서 이범석 등이 환영을 받았다."
)
NEW_BACKGROUND = _ranked(
    "background",
    "행사의 배경은 지역 교통망을 알리고 시민을 격려하려는 데 있었다. "
    "행사 이후 관련 활동이 이어졌고 지역 사회에 영향을 남겼다.",
)


def _store_with_turn(
    user: str,
    assistant: str = "검색 근거에 따른 답변",
    *,
    people: tuple[str, ...] = (),
):
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, user, assistant)
    store.update_context(session.session_id, recent_people=people)
    return store, session


def _resolve(user: str, followup: str, *, people: tuple[str, ...] = ()):
    _store, session = _store_with_turn(user, people=people)
    return ConversationContextResolver().resolve(
        followup, session, current_place_id=None, current_piece_id=None
    )


@pytest.mark.parametrize(
    "followup",
    (
        "좀 더 쉽게 설명해줘", "한 문장으로 말해줘", "정리해서 알려줘",
        "짧게 설명해줘", "초등학생도 이해할 수 있게 설명해줘",
    ),
)
def test_reformulation_and_detail_expansion_are_distinct(followup: str) -> None:
    resolved = _resolve("목포역 행사를 알려줘", followup)
    assert resolved.request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER


@pytest.mark.parametrize(
    "followup",
    (
        "좀 더 자세히 알려줘", "조금 더 설명해줘", "그 사건에 대해 더 알려줘",
        "관련 내용도 더 알려줘", "배경까지 자세히 알려줘",
    ),
)
def test_detail_expansion_forms_are_classified_for_sufficiency_check(
    followup: str,
) -> None:
    resolved = _resolve("목포역 행사를 알려줘", followup)
    assert resolved.request_kind == ConversationRequestKind.EXPAND_PREVIOUS_ANSWER


def test_pronoun_resolution_uses_recent_focused_person() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    resolver = ConversationContextResolver()
    first = resolver.resolve(
        "이범석이 목포에 온 사건을 알려줘.", session,
        current_place_id=None, current_piece_id=None,
    )
    assert first.recent_people == ("이범석",)
    store.update_context(session.session_id, recent_people=first.recent_people)
    store.add_turn(session.session_id, first.current_user_query, "근거 기반 응답")
    resolved = resolver.resolve(
        "그 사람은 왜 왔어?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert resolved.request_kind == ConversationRequestKind.FACTUAL_FOLLOWUP
    assert "이범석" in resolved.resolved_question
    assert resolved.needs_new_evidence is True


def test_temporal_reference_keeps_recent_event_and_period() -> None:
    store, session = _store_with_turn("1949년 목포역에서 무슨 일이 있었어?")
    store.update_context(
        session.session_id,
        active_place="목포역",
        active_topic="목포역",
        recent_period="1949년",
    )
    resolved = ConversationContextResolver().resolve(
        "그때 참석한 사람들은?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert resolved.followup_resolved is True
    assert "1949년" in resolved.search_query
    assert "목포역" in resolved.search_query


def test_implicit_comparison_carries_previous_predicate() -> None:
    resolved = _resolve(
        "안호상은 왜 목포에 왔어?",
        "이범석은?",
        people=("안호상",),
    )
    assert resolved.resolved_question == "이범석은 왜 목포에 왔어?"
    assert "안호상" not in resolved.search_query
    assert "이범석" in resolved.search_query


def test_answer_reformulation_reuses_verified_chunks_without_search(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime",
        session_path=tmp_path / "sessions.json",
    )
    calls: list[str] = []
    search = chat.retrieval.search

    def capture(query: str):
        calls.append(query)
        return search(query)

    monkeypatch.setattr(chat.retrieval, "search", capture)
    first = chat.ask("붉은 등대 전시관을 알려줘")
    easy = chat.ask("좀 더 쉽게 설명해줘", session_id=first.session_id)

    assert len(calls) == 1
    assert easy.context_metadata["request_kind"] == "transform_previous_answer"
    assert easy.context_metadata["retrieval_performed"] is False
    assert easy.retrieved_chunk_ids == first.retrieved_chunk_ids


def test_sufficient_detail_expansion_reuses_existing_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    session = chat.sessions.create()
    chat.sessions.add_turn(session.session_id, "목포역 행사를 알려줘", "짧은 답변")
    chat.sessions.add_evidence_turn(
        session.session_id, user="목포역 행사를 알려줘",
        active_place="목포역", active_topic="행사", chunk_ids=("rich",),
    )
    monkeypatch.setattr(chat, "_remembered_evidence", lambda *args, **kwargs: [RICH_EVIDENCE])
    calls: list[str] = []
    monkeypatch.setattr(chat.retrieval, "search", lambda query: calls.append(query) or [])

    response = chat.ask("좀 더 자세히 알려줘", session_id=session.session_id)

    assert calls == []
    assert response.context_metadata["detail_evidence_sufficient"] is True
    assert response.context_metadata["needs_new_evidence"] is False
    assert response.context_metadata["retrieval_performed"] is False
    assert response.retrieved_chunk_ids == ("rich",)


def test_insufficient_detail_expansion_retrieves_and_composes_partial_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    session = chat.sessions.create()
    chat.sessions.add_turn(session.session_id, "목포역 행사를 알려줘", "짧은 답변")
    chat.sessions.add_evidence_turn(
        session.session_id, user="목포역 행사를 알려줘",
        active_place="목포역", active_topic="행사", chunk_ids=("short",),
    )
    monkeypatch.setattr(chat, "_remembered_evidence", lambda *args, **kwargs: [SHORT_EVIDENCE])
    calls: list[str] = []
    monkeypatch.setattr(
        chat.retrieval, "search",
        lambda query: calls.append(query) or [SHORT_EVIDENCE, NEW_BACKGROUND],
    )

    response = chat.ask(
        "그 행사의 배경과 결과까지 자세히 알려줘",
        session_id=session.session_id,
    )

    assert len(calls) == 1
    assert response.context_metadata["detail_evidence_sufficient"] is False
    assert response.context_metadata["needs_new_evidence"] is True
    assert response.context_metadata["retrieval_performed"] is True
    assert response.context_metadata["partial_evidence_used"] is True
    assert response.retrieved_chunk_ids == ("short", "background")
    assert len(response.retrieved_chunk_ids) == len(set(response.retrieved_chunk_ids))

    chat.ask("그 이후 이범석은 어떻게 됐어?", session_id=session.session_id)
    assert len(calls) == 2


def test_sync_and_stream_share_pre_generation_evidence_decision(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    other_place = _ranked(
        "other-place",
        "유달산의 지형과 경관을 설명하는 별도 기록이다. " * 20,
        title="유달산 기록",
    )
    monkeypatch.setattr(
        chat, "_remembered_evidence",
        lambda *args, **kwargs: [other_place, SHORT_EVIDENCE],
    )
    searches: list[str] = []
    monkeypatch.setattr(
        chat.retrieval, "search",
        lambda query: searches.append(query) or [SHORT_EVIDENCE, NEW_BACKGROUND],
    )

    session_ids: list[str] = []
    for _ in range(2):
        session = chat.sessions.create()
        chat.sessions.add_turn(session.session_id, "목포역 행사를 알려줘", "짧은 답변")
        chat.sessions.update_context(
            session.session_id, active_place="목포역", active_topic="행사"
        )
        chat.sessions.add_evidence_turn(
            session.session_id, user="목포역 행사를 알려줘",
            active_place="목포역", active_topic="행사", chunk_ids=("short",),
        )
        session_ids.append(session.session_id)

    sync = chat.ask("좀 더 자세히 알려줘", session_id=session_ids[0])
    completed = list(
        chat.stream("좀 더 자세히 알려줘", session_id=session_ids[1])
    )[-1]
    stream_metadata = completed.data["context_metadata"]
    sync_metadata = sync.context_metadata
    assert sync_metadata is not None
    compared = (
        "resolved_question", "request_kind", "needs_new_evidence",
        "selected_evidence_ids", "retrieval_performed", "active_place",
        "detail_evidence_sufficient", "partial_evidence_used",
    )
    assert {key: sync_metadata[key] for key in compared} == {
        key: stream_metadata[key] for key in compared
    }
    assert sync_metadata["selected_evidence_ids"] == ("short", "background")
    assert "other-place" not in sync_metadata["selected_evidence_ids"]
    assert len(searches) == 2


def test_explicit_topic_switch_does_not_inherit_station() -> None:
    resolved = _resolve("목포역은 언제 만들어졌어?", "유달산은 왜 유명해?")
    assert resolved.request_kind == ConversationRequestKind.INDEPENDENT
    assert "목포역" not in resolved.search_query
    assert resolved.search_query == "유달산은 왜 유명해?"


def test_user_correction_has_priority_over_previous_interpretation() -> None:
    resolved = _resolve(
        "그 행사에 누가 왔어?",
        "아니, 내가 물어본 건 그 사람이 목포에 온 이유야.",
        people=("이범석",),
    )
    assert resolved.request_kind == ConversationRequestKind.CORRECTION
    assert "이범석" in resolved.resolved_question
    assert "목포에 온 이유" in resolved.resolved_question


def test_assistant_claim_is_context_but_never_llm_evidence(
    tmp_path: Path,
) -> None:
    llm = CapturingLLM()
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", llm=llm, in_memory_sessions=True
    )
    false_claim = "목포역은 9999년에 세워졌다고 단정한다."
    first = chat.ask("붉은 등대 전시관은 왜 만들었어?")
    session = chat.sessions.get(first.session_id)
    assert session is not None
    session.turns[-1] = SessionTurn(session.turns[-1].user, false_claim)

    response = chat.ask(
        "그 행사의 배경과 결과까지 자세히 알려줘",
        session_id=first.session_id,
    )

    assert response.status == "ok"
    request = llm.requests[-1]
    assert false_claim in tuple(message.content for message in request.messages)
    assert false_claim not in request.user_prompt
    assert false_claim not in request.metadata["evidence"]
    assert all(false_claim != item for item in response.evidence)
    assert "대화 문맥 | 역사적 사실의 근거가 아님" in request.user_prompt


def test_long_conversation_prefers_recent_topic() -> None:
    store = SessionStore(RuntimeMode.HACKATHON, max_turns=4)
    session = store.create()
    for index, topic in enumerate(("목포역", "유달산", "삼학도", "목포항", "학생운동")):
        store.add_turn(session.session_id, f"{topic} 질문 {index}", f"{topic} 답변 {index}")
        store.update_context(session.session_id, active_topic=topic, recent_event=topic)
    resolved = ConversationContextResolver().resolve(
        "그때 결과는?", session, current_place_id=None, current_piece_id=None
    )
    assert "학생운동" in resolved.search_query
    assert "목포역" not in resolved.search_query


def test_ambiguous_pronoun_does_not_always_choose_first_person() -> None:
    resolved = _resolve(
        "행사 참석자를 알려줘.",
        "그 사람은 왜 왔어?",
        people=("이범석", "안호상"),
    )
    assert "이범석" not in resolved.resolved_question
    assert "안호상" not in resolved.resolved_question
    assert "그 사람" in resolved.resolved_question
