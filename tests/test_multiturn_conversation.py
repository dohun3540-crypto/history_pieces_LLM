from __future__ import annotations

from pathlib import Path

import pytest

from history_chatbot.chat.context_resolver import (
    ConversationContextResolver,
    ConversationRequestKind,
)
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.remote_safe import (
    EvidenceSupport,
    GroundedFact,
    GroundedFactPacket,
)
from history_chatbot.chat.service import create_development_orchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.chat.session import SessionTurn
from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.retrieval.query_normalizer import explicit_subject_words
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
    ("return_query", "expected", "intervening"),
    (
        ("다시 목포역으로 돌아가서 개통 시기를 알려 줘.", "목포역", "광주학생운동"),
        ("다시 고하도로 돌아가서 관련 인물을 알려 줘.", "고하도", "고인돌"),
    ),
)
def test_explicit_topic_return_drops_intervening_context(
    return_query: str, expected: str, intervening: str,
) -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, f"{expected}에 대해 알려 줘.", "첫 답변")
    store.add_evidence_turn(
        session.session_id, user=f"{expected}에 대해 알려 줘.",
        active_place="", active_topic=expected, chunk_ids=(f"{expected}-1",),
    )
    store.add_turn(session.session_id, f"{intervening}을 알려 줘.", "둘째 답변")
    store.add_evidence_turn(
        session.session_id, user=f"{intervening}을 알려 줘.",
        active_place="", active_topic=intervening, chunk_ids=(f"{intervening}-1",),
    )
    store.update_context(
        session.session_id, active_topic=intervening, recent_event=intervening,
    )

    resolved = ConversationContextResolver().resolve(
        return_query, session, current_place_id=None, current_piece_id=None,
    )

    assert expected in resolved.search_query
    assert intervening not in resolved.search_query


def test_return_to_first_event_uses_first_retrieved_user_turn_only() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "세월호 참사를 설명해 줘.", "첫 답변")
    store.add_evidence_turn(
        session.session_id, user="세월호 참사를 설명해 줘.",
        active_place="", active_topic="세월호 참사", chunk_ids=("sewol-1",),
    )
    store.add_turn(session.session_id, "목포대학교를 설명해 줘.", "둘째 답변")
    store.update_context(session.session_id, active_topic="목포대학교")

    resolved = ConversationContextResolver().resolve(
        "다시 첫 사건으로 돌아가 장소를 알려 줘.", session,
        current_place_id=None, current_piece_id=None,
    )

    assert "세월호 참사" in resolved.search_query
    assert "장소" in resolved.search_query
    assert "목포대학교" not in resolved.search_query
    assert resolved.active_topic == "세월호 참사"


def test_generic_place_return_uses_validated_subject_history() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    for topic in ("가람도", "누리도"):
        user = f"{topic}의 역사를 알려 줘."
        store.add_turn(session.session_id, user, "근거 기반 답변")
        store.add_evidence_turn(
            session.session_id, user=user, active_place="", active_topic=topic,
            chunk_ids=(f"{topic}-1",),
        )
        store.update_context(session.session_id, active_topic=topic)

    resolved = ConversationContextResolver().resolve(
        "다시 가람도로 돌아가 장소 특징을 알려 줘.", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_topic == "가람도"
    assert "가람도" in resolved.search_query
    assert "누리도" not in resolved.search_query


def test_unvalidated_false_premise_topic_is_not_inherited() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "가람회가 2099년에 생겼지?", "확인할 수 없습니다.")
    store.update_context(session.session_id, active_topic="가람회")

    resolved = ConversationContextResolver().resolve(
        "그럼 언제였어?", session, current_place_id=None, current_piece_id=None,
    )

    assert "가람회" not in resolved.search_query


def test_false_premise_keeps_verified_subject_but_discards_claimed_period(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    evidence = _ranked(
        "organization",
        "신간회는 1927년에 창립된 민족 협동 전선 단체이다.",
        title="신간회 - 검증 사전",
    )
    monkeypatch.setattr(chat.retrieval, "search", lambda query: [evidence])

    first = chat.ask("신간회가 2001년에 생겼다는 말이 맞아?")
    session = chat.sessions.get(first.session_id)
    assert session is not None
    assert first.status == "insufficient_evidence"
    assert session.stable_evidence_anchor == "신간회"
    assert session.recent_period == ""

    temporal = chat.context_resolver.resolve(
        "그럼 실제 시기는 언제야?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert temporal.active_subject == "신간회"
    assert temporal.current_intent == "time"
    assert temporal.search_query.startswith("신간회 ")
    assert "2001" not in temporal.search_query


def test_false_premise_followup_keeps_current_people_intent(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    evidence = _ranked(
        "organization",
        "신간회는 1927년에 창립되었고 이상재가 초대 회장을 맡았다.",
        title="신간회 - 검증 사전",
    )
    monkeypatch.setattr(chat.retrieval, "search", lambda query: [evidence])
    first = chat.ask("신간회가 2001년에 생겼다는 말이 맞아?")
    session = chat.sessions.get(first.session_id)
    assert session is not None

    people = chat.context_resolver.resolve(
        "관련 인물은?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert people.active_subject == "신간회"
    assert people.current_intent == "people"
    assert people.search_query == "신간회 관련 인물은?"
    assert "2001" not in people.search_query


def test_current_turn_facet_overrides_previous_facet_with_subject_continuity() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "가람회의 역사를 알려 줘.", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id,
        user="가람회의 역사를 알려 줘.",
        active_place="",
        active_topic="가람회",
        active_subject="가람회",
        answered_intent="overview",
        chunk_ids=("organization",),
    )
    store.update_context(
        session.session_id,
        active_topic="가람회",
        active_subject="가람회",
        stable_evidence_anchor="가람회",
    )
    resolver = ConversationContextResolver()

    temporal = resolver.resolve(
        "언제였어?", session, current_place_id=None, current_piece_id=None
    )
    assert temporal.active_subject == "가람회"
    assert temporal.current_intent == "time"

    store.add_turn(session.session_id, "언제였어?", "1900년에 창립되었다.")
    store.add_evidence_turn(
        session.session_id,
        user="언제였어?",
        active_place="",
        active_topic="가람회",
        active_subject="가람회",
        answered_intent="time",
        chunk_ids=("organization",),
    )
    people = resolver.resolve(
        "관련 인물은?", session, current_place_id=None, current_piece_id=None
    )
    assert people.active_subject == "가람회"
    assert people.current_intent == "people"
    assert people.search_query == "가람회 관련 인물은?"

    store.add_turn(session.session_id, "관련 인물은?", "이기록이 참여하였다.")
    store.add_evidence_turn(
        session.session_id,
        user="관련 인물은?",
        active_place="",
        active_topic="가람회",
        active_subject="가람회",
        answered_intent="people",
        chunk_ids=("organization",),
    )
    place = resolver.resolve(
        "관련 장소는?", session, current_place_id=None, current_piece_id=None
    )
    assert place.active_subject == "가람회"
    assert place.current_intent == "place"

    store.add_turn(session.session_id, "관련 장소는?", "한빛관에서 활동하였다.")
    store.add_evidence_turn(
        session.session_id,
        user="관련 장소는?",
        active_place="",
        active_topic="가람회",
        active_subject="가람회",
        answered_intent="place",
        chunk_ids=("organization",),
    )
    temporal_again = resolver.resolve(
        "그럼 언제였어?", session, current_place_id=None, current_piece_id=None
    )
    assert temporal_again.active_subject == "가람회"
    assert temporal_again.current_intent == "time"


def test_false_premise_temporal_followup_retries_with_subject_and_intent(
    tmp_path: Path, monkeypatch,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    evidence = _ranked(
        "organization",
        "신간회는 1927년에 창립된 민족 협동 전선 단체이다.",
        title="신간회 - 검증 사전",
    )
    session = chat.sessions.create()
    chat.sessions.add_turn(
        session.session_id,
        "신간회가 잘못된 연도에 생겼다는 전제가 맞아?",
        "그 전제는 기록과 맞지 않아요.",
    )
    chat.sessions.update_context(
        session.session_id,
        active_subject="신간회",
        active_topic="신간회",
        stable_evidence_anchor="신간회",
    )
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        return [evidence] if query == "신간회 창립 시기" else []

    monkeypatch.setattr(chat.retrieval, "search", search)
    prepared = chat._prepare_turn_evidence(
        "그럼 실제 시기는 언제야?", session,
        top_k=3, current_place_id=None, current_piece_id=None,
    )

    assert calls[-1] == "신간회 창립 시기"
    assert prepared.retrieval_retry_performed is True
    assert prepared.assessed_intent == "time"
    assert prepared.chunks == (evidence,)


def test_current_explicit_subject_beats_validated_previous_topic() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "가람도의 역사를 알려 줘.", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="가람도의 역사를 알려 줘.", active_place="",
        active_topic="가람도", chunk_ids=("island-1",),
    )
    store.update_context(session.session_id, active_topic="가람도")

    resolved = ConversationContextResolver().resolve(
        "이번에는 누리도의 역사를 알려 줘.", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_topic == "누리도"
    assert "누리도" in resolved.search_query
    assert "가람도" not in resolved.search_query


def test_chained_short_followups_keep_original_evidence_subject() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포역에 대해 알려 줘.", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="목포역에 대해 알려 줘.", active_place="목포역",
        active_topic="목포역", chunk_ids=("station-1",),
    )
    store.add_turn(session.session_id, "언제?", "1913년이라는 근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="언제?", active_place="목포역",
        active_topic="목포역", chunk_ids=("station-1",),
    )
    store.update_context(session.session_id, active_place="목포역", active_topic="목포역")

    resolved = ConversationContextResolver().resolve(
        "왜?", session, current_place_id=None, current_piece_id=None,
    )

    assert resolved.search_query.startswith("목포역")
    assert "언제? 왜?" not in resolved.search_query


def test_afterward_followup_keeps_original_evidence_subject() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포진은 어떤 곳이었어?", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="목포진은 어떤 곳이었어?", active_place="목포진",
        active_topic="목포진", chunk_ids=("fort-1",),
    )
    store.update_context(session.session_id, active_place="목포진", active_topic="목포진")

    resolved = ConversationContextResolver().resolve(
        "그 이후에는?", session, current_place_id=None, current_piece_id=None,
    )

    assert resolved.followup_resolved
    assert resolved.search_query.startswith("목포진")


def test_explicit_new_person_clears_stale_place_context() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포역을 알려 줘.", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="목포역을 알려 줘.", active_place="목포역",
        active_topic="목포역", chunk_ids=("station-1",),
    )
    store.update_context(session.session_id, active_place="목포역", active_topic="목포역")

    resolved = ConversationContextResolver().resolve(
        "이범석은 누구야?", session, current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_topic == "이범석"
    assert resolved.active_place == ""
    assert "목포역" not in resolved.search_query


@pytest.mark.parametrize(
    ("followup", "expected"),
    (("첫 단체의 인물은?", "가람회"), ("두 번째 단체의 시기는?", "누리회")),
)
def test_ordinal_group_reference_uses_validated_subjects(
    followup: str, expected: str,
) -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    user = "가람회와 누리회를 구분해 줘."
    store.add_turn(session.session_id, user, "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user=user, active_place="", active_topic="가람회",
        chunk_ids=("group-1",),
    )
    store.update_context(session.session_id, active_topic="가람회")

    resolved = ConversationContextResolver().resolve(
        followup, session, current_place_id=None, current_piece_id=None,
    )

    assert expected in resolved.search_query


@pytest.mark.parametrize("followup", ("그 노선과 관련된 역은?", "관련 시기는?"))
def test_relational_ellipsis_inherits_validated_topic(followup: str) -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "가람선을 알려 줘.", "근거 기반 답변")
    store.add_evidence_turn(
        session.session_id, user="가람선을 알려 줘.", active_place="",
        active_topic="가람선", chunk_ids=("rail-1",),
    )
    store.update_context(session.session_id, active_topic="가람선")

    resolved = ConversationContextResolver().resolve(
        followup, session, current_place_id=None, current_piece_id=None,
    )

    assert "가람선" in resolved.search_query


def test_subject_aware_comparison_keeps_one_result_per_subject(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "sessions.json",
    )
    first = _ranked("first-group", "가람회 활동 기록", title="가람회")
    second = _ranked("second-group", "누리회 활동 기록", title="누리회")
    monkeypatch.setattr(
        engine.retrieval, "search",
        lambda query: [first] if query == "가람회" else [second] if query == "누리회" else [],
    )

    results = engine._subject_aware_search("가람회와 누리회를 구분해 줘.", 3)

    assert [item.chunk.title for item in results] == ["가람회", "누리회"]


def test_subject_aware_relationship_keeps_person_and_place_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "sessions.json",
    )
    person = _ranked("person", "이범석이 행사에 참석했다.", title="행사 기록")
    station = _ranked("station", "목포역에서 행사가 열렸다.", title="목포역")
    monkeypatch.setattr(
        engine.retrieval, "search",
        lambda query: [person] if query == "이범석" else [station] if query == "목포역" else [],
    )

    results = engine._subject_aware_search("이범석은 목포역에서 무슨 일을 했어?", 3)

    assert {item.chunk.chunk_id for item in results} == {"person", "station"}


def test_weak_full_query_retries_with_only_confirmed_subject(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "sessions.json",
    )
    station = _ranked("station", "목포역은 1913년에 영업을 시작했다.", title="목포역")
    queries: list[str] = []

    def search(query: str):
        queries.append(query)
        return [station] if query == "목포역" else []

    monkeypatch.setattr(engine.retrieval, "search", search)
    response = engine.ask("목포역 언제 만들어졌어?")

    assert queries == ["목포역 언제 만들어졌어?", "목포역"]
    assert response.used_chunks == 1
    assert response.context_metadata["retrieval_retry_performed"] is True
    assert response.context_metadata["retrieval_retry_query"] == "목포역"


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


def test_person_reference_keeps_active_person() -> None:
    store, session = _store_with_turn(
        "이범석은 누구야?", people=("이범석", "안호상")
    )
    store.update_context(
        session.session_id, active_topic="목포역", active_subject="이범석",
        active_person="이범석", stable_evidence_anchor="이범석",
    )
    store.add_evidence_turn(
        session.session_id, user="이범석은 누구야?", active_place="",
        active_topic="이범석", active_subject="이범석", active_person="이범석",
        answered_intent="people", chunk_ids=("person",),
    )

    resolved = ConversationContextResolver().resolve(
        "그 사람은 목포역에서 뭘 했어?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_person == "이범석"
    assert "이범석" in resolved.resolved_question


def test_person_reference_does_not_inherit_wrong_place() -> None:
    store, session = _store_with_turn("이범석은 누구야?", people=("이범석",))
    store.update_context(
        session.session_id, active_place="목포역", active_topic="목포역",
        active_subject="이범석", active_person="이범석",
        stable_evidence_anchor="이범석",
    )
    resolved = ConversationContextResolver().resolve(
        "그 사람은 누구야?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert resolved.active_subject == "이범석"
    assert resolved.search_query.startswith("이범석")


def test_place_adverb_is_not_promoted_to_active_person() -> None:
    store, session = _store_with_turn("동양척식주식회사 목포지점")
    store.update_context(
        session.session_id,
        active_topic="동양척식주식회사 목포지점",
        active_subject="동양척식주식회사 목포지점",
        stable_evidence_anchor="동양척식주식회사 목포지점",
    )

    resolved = ConversationContextResolver().resolve(
        "거기서는 무슨 일을 했어?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_person == ""
    assert resolved.active_subject == "동양척식주식회사 목포지점"
    assert resolved.search_query.startswith("동양척식주식회사 목포지점")


def test_explicit_place_in_anchored_relation_does_not_reset_subject() -> None:
    store, session = _store_with_turn("동양척식주식회사 목포지점")
    store.update_context(
        session.session_id,
        active_topic="동양척식주식회사 목포지점",
        active_subject="동양척식주식회사 목포지점",
        stable_evidence_anchor="동양척식주식회사 목포지점",
    )

    resolved = ConversationContextResolver().resolve(
        "왜 목포에 있었어?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_subject == "동양척식주식회사 목포지점"
    assert "동양척식주식회사 목포지점" in resolved.search_query
    assert "목포" in resolved.search_query


@pytest.mark.parametrize("query", ("그거", "그게 무슨 뜻이야?", "그 말이 뭐야?", "무슨 뜻이야?"))
def test_that_means_uses_verified_previous_evidence(query: str) -> None:
    store, session = _store_with_turn("동양척식주식회사 목포지점은 어떤 곳이야?")
    store.update_context(
        session.session_id,
        active_topic="동양척식주식회사 목포지점",
        active_subject="동양척식주식회사 목포지점",
        stable_evidence_anchor="동양척식주식회사 목포지점",
        last_answered_intent="role",
    )
    store.add_evidence_turn(
        session.session_id,
        user="동양척식주식회사 목포지점은 어떤 곳이야?",
        active_place="동양척식주식회사 목포지점",
        active_topic="동양척식주식회사 목포지점",
        active_subject="동양척식주식회사 목포지점",
        answered_intent="role",
        chunk_ids=("company",),
    )

    resolved = ConversationContextResolver().resolve(
        query, session, current_place_id=None, current_piece_id=None
    )

    assert resolved.request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
    assert resolved.active_subject == "동양척식주식회사 목포지점"
    assert resolved.search_query.startswith("동양척식주식회사 목포지점")
    assert "그거" not in resolved.recent_entities


def test_three_turn_followup_keeps_evidence_anchor() -> None:
    store, session = _store_with_turn("목포역은 언제 만들어졌어?")
    store.add_evidence_turn(
        session.session_id, user="목포역은 언제 만들어졌어?",
        active_place="목포역", active_topic="목포역", active_subject="목포역",
        answered_intent="time", chunk_ids=("station",),
    )
    store.update_context(
        session.session_id, active_place="목포역", active_topic="목포역",
        active_subject="목포역", stable_evidence_anchor="목포역",
    )
    resolver = ConversationContextResolver()
    for query in ("왜 만들었어?", "그 뒤에는?", "관련 인물은?"):
        resolved = resolver.resolve(
            query, session, current_place_id=None, current_piece_id=None
        )
        assert "목포역" in resolved.search_query
        store.add_turn(session.session_id, query, "근거 기반 답변")


def test_four_turn_followup_does_not_collapse_to_unanchored_query() -> None:
    store, session = _store_with_turn("구 목포 일본영사관은 어떤 건물이야?")
    store.add_evidence_turn(
        session.session_id, user="구 목포 일본영사관은 어떤 건물이야?",
        active_place="구 목포 일본영사관", active_topic="구 목포 일본영사관",
        active_subject="구 목포 일본영사관", answered_intent="overview",
        chunk_ids=("consulate",),
    )
    store.update_context(
        session.session_id, active_place="구 목포 일본영사관",
        active_topic="구 목포 일본영사관", active_subject="구 목포 일본영사관",
        stable_evidence_anchor="구 목포 일본영사관",
    )
    resolver = ConversationContextResolver()
    for query in ("언제 지어졌어?", "왜 지었어?", "그 다음에는?", "지금은 뭐야?"):
        resolved = resolver.resolve(query, session, current_place_id=None, current_piece_id=None)
        assert "일본영사관" in resolved.search_query
        store.add_turn(session.session_id, query, "근거 기반 답변")


def test_temporal_reference_keeps_recent_event_and_period() -> None:
    store, session = _store_with_turn("1949년 목포역에서 무슨 일이 있었어?")
    store.update_context(
        session.session_id,
        active_place="목포역",
        active_topic="목포역",
        recent_period="1949년",
    )
    store.add_evidence_turn(
        session.session_id, user="1949년 목포역에서 무슨 일이 있었어?",
        active_place="목포역", active_topic="목포역", chunk_ids=("station-1",),
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


@pytest.mark.parametrize("style_request", ("쉽게 설명해줘", "핵심만", "다시 설명해줘"))
def test_transform_reuses_verified_evidence_without_retrieval(
    tmp_path: Path, monkeypatch, style_request: str,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    first = chat.ask("붉은 등대 전시관을 알려줘")
    calls: list[str] = []
    monkeypatch.setattr(chat.retrieval, "search", lambda query: calls.append(query) or [])

    transformed = chat.ask(style_request, session_id=first.session_id)

    assert calls == []
    assert transformed.context_metadata["request_kind"] == "transform_previous_answer"
    assert transformed.context_metadata["memory_evidence_used"] is True
    assert transformed.retrieved_chunk_ids == first.retrieved_chunk_ids


@pytest.mark.parametrize("reference", ("그거", "그게 무슨 뜻이야?", "무슨 뜻이야?"))
def test_usable_context_prevents_generic_fallback(
    tmp_path: Path, monkeypatch, reference: str,
) -> None:
    chat = create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", in_memory_sessions=True
    )
    first = chat.ask("붉은 등대 전시관을 알려줘")
    calls: list[str] = []
    monkeypatch.setattr(chat.retrieval, "search", lambda query: calls.append(query) or [])

    response = chat.ask(reference, session_id=first.session_id)

    assert calls == []
    assert response.status != "insufficient_evidence"
    assert response.context_metadata["request_kind"] == "transform_previous_answer"
    assert response.context_metadata["memory_evidence_used"] is True


def test_transform_explains_limitation_instead_of_repeating_it() -> None:
    answer = ConversationalRagOrchestrator._explain_evidence_boundary(
        "목포역", "cause"
    )

    assert "뜻이에요" in answer
    assert "직접적인 이유" in answer
    assert "확인하기 어려워요" not in answer


def test_nearby_supported_fact_preferred_over_bare_limitation() -> None:
    packet = GroundedFactPacket(
        "목포역", "time",
        (GroundedFact("목포역", "time", "목포역은 1913년에 영업을 시작했다.", "source"),),
        EvidenceSupport.NEARBY_SUPPORTED,
    )

    answer = ConversationalRagOrchestrator._nearby_supported_answer(
        "목포역", "cause", packet
    )

    assert "이유 자체" in answer
    assert "1913년에 영업을 시작" in answer


def test_comparison_builds_answer_for_each_subject() -> None:
    packets = (
        GroundedFactPacket(
            "1관", "overview",
            (GroundedFact("1관", "overview", "1관은 옛 영사관 건물이다.", "a"),),
            EvidenceSupport.DIRECT,
        ),
        GroundedFactPacket(
            "2관", "overview",
            (GroundedFact("2관", "overview", "2관은 옛 회사 지점 건물이다.", "b"),),
            EvidenceSupport.DIRECT,
        ),
    )

    answer, complete = ConversationalRagOrchestrator._comparison_answer(packets)

    assert complete is True
    assert "1관은 옛 영사관" in answer
    assert "2관은 옛 회사" in answer


def test_numbered_place_comparison_resolves_both_subjects() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.update_context(
        session.session_id,
        active_subject="목포근대역사관 1관",
        active_place="목포근대역사관 1관",
        stable_evidence_anchor="목포근대역사관 1관",
    )
    store.add_turn(session.session_id, "1관은 어떤 곳이야?", "근거 기반 응답")

    resolved = ConversationContextResolver().resolve(
        "2관과 뭐가 달라?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert "목포근대역사관 1관" in resolved.search_query
    assert "목포근대역사관 2관" in resolved.search_query


def test_role_followup_does_not_promote_predicate_to_subject() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.update_context(
        session.session_id,
        active_subject="동양척식주식회사 목포지점",
        stable_evidence_anchor="동양척식주식회사 목포지점",
    )
    store.add_turn(session.session_id, "동양척식주식회사 목포지점", "근거 기반 응답")

    resolved = ConversationContextResolver().resolve(
        "뭐 하는 곳이었어?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert resolved.active_subject == "동양척식주식회사 목포지점"


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

    assert response.status in {"ok", "partial_evidence"}
    assert all(false_claim != item for item in response.evidence)
    if response.status == "ok":
        assert llm.requests
        request = llm.requests[-1]
        assert false_claim in {message.content for message in request.messages}
        assert false_claim not in request.user_prompt
        assert false_claim not in request.metadata["evidence"]
    else:
        assert response.context_metadata["evidence_support"] in {
            "related_only",
            "none",
        }


def test_long_conversation_prefers_recent_topic() -> None:
    store = SessionStore(RuntimeMode.HACKATHON, max_turns=4)
    session = store.create()
    for index, topic in enumerate(("목포역", "유달산", "삼학도", "목포항", "학생운동")):
        store.add_turn(session.session_id, f"{topic} 질문 {index}", f"{topic} 답변 {index}")
        store.add_evidence_turn(
            session.session_id, user=f"{topic} 질문 {index}", active_place="",
            active_topic=topic, chunk_ids=(f"topic-{index}",),
        )
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


def test_active_person_topic_wins_for_pronoun_when_multiple_people_exist() -> None:
    store, session = _store_with_turn(
        "이범석은 누구야?", people=("이범석", "안호상")
    )
    store.update_context(session.session_id, active_topic="이범석")

    resolved = ConversationContextResolver().resolve(
        "그 사람은 무슨 일을 했어?", session,
        current_place_id=None, current_piece_id=None,
    )

    assert "이범석" in resolved.resolved_question


@pytest.mark.parametrize(
    "followup",
    ("그 일이 왜 중요했는데?", "그게 정확히 무슨 뜻이야?", "그 다음에는?"),
)
def test_natural_followup_variants_keep_topic(followup: str) -> None:
    store, session = _store_with_turn("목포역에 대해 알려줘")
    store.add_evidence_turn(
        session.session_id, user="목포역에 대해 알려줘", active_place="목포역",
        active_topic="목포역", chunk_ids=("station",),
    )
    store.update_context(session.session_id, active_place="목포역", active_topic="목포역")
    resolved = ConversationContextResolver().resolve(
        followup, session, current_place_id=None, current_piece_id=None
    )
    assert resolved.request_kind != ConversationRequestKind.INDEPENDENT
    assert "목포역" in resolved.search_query


@pytest.mark.parametrize(
    "followup",
    ("아주 쉽게 설명해줘", "간단하게 정리해줘", "쉽게 풀어줘"),
)
def test_natural_transformation_variants_reuse_verified_evidence(followup: str) -> None:
    resolved = _resolve("목포역에 대해 알려줘", followup)
    assert resolved.request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
    assert resolved.needs_new_evidence is False


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("해양 사고와 관련된 인물이나 장소를 알려 줘.", "해양 사고"),
        ("지역 학생운동을 설명해 줘.", "지역 학생운동"),
        ("도시 양동교회의 역사를 알려 줘.", "도시 양동교회"),
        ("목포 양동교회의 역사를 알려 줘.", "목포 양동교회"),
    ),
)
def test_independent_query_keeps_specific_compound_subject(
    query: str, expected: str,
) -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()

    resolved = ConversationContextResolver().resolve(
        query, session, current_place_id=None, current_piece_id=None
    )

    assert resolved.active_subject == expected
    assert resolved.active_person == ""


def test_coordinated_entities_are_both_preserved_for_coverage() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()

    resolved = ConversationContextResolver().resolve(
        "북부선 개통과 지역 학생운동의 날짜와 인물을 구분해 줘.",
        session,
        current_place_id=None,
        current_piece_id=None,
    )

    assert "북부선" in resolved.recent_entities
    assert "지역 학생운동" in resolved.recent_entities


@pytest.mark.parametrize(
    "query",
    (
        "이기록과 김자료를 각각 설명해 줘.",
        "북부선과 지역 학생운동을 구분해서 설명해 줘.",
    ),
)
def test_general_comparison_preserves_both_explicit_subjects(query: str) -> None:
    subjects = explicit_subject_words(query)

    assert len(subjects) == 2


@pytest.mark.parametrize(
    ("query", "subject"),
    (
        ("먼저 북부선 시기만 알려 줘.", "북부선"),
        ("이제 지역 학생운동 시기만 알려 줘.", "지역 학생운동"),
    ),
)
def test_named_facet_followup_keeps_its_explicit_subject(
    query: str, subject: str,
) -> None:
    assert explicit_subject_words(query) == (subject,)


def test_decimal_numbered_event_keeps_complete_explicit_subject() -> None:
    assert explicit_subject_words(
        "3.1운동과 관련된 인물이나 장소를 알려 줘."
    ) == ("3.1운동",)


def test_person_activity_reference_uses_role_facet_not_people_facet() -> None:
    store, session = _store_with_turn("이기록을 설명해 줘.", people=("이기록",))
    store.update_context(
        session.session_id,
        active_person="이기록",
        active_subject="이기록",
        stable_evidence_anchor="이기록",
    )

    resolved = ConversationContextResolver().resolve(
        "그 사람은 어떤 활동을 했어?",
        session,
        current_place_id=None,
        current_piece_id=None,
    )

    assert resolved.active_person == "이기록"
    assert resolved.current_intent == "role"
    assert "이기록" in resolved.search_query


def test_comparison_certainty_followups_do_not_promote_discourse_words() -> None:
    assert explicit_subject_words(
        "둘 중 자료로 더 명확히 확인되는 사람을 구분해 줘."
    ) == ()
    assert explicit_subject_words("확실하지 않은 부분은 뭐야?") == ()


def test_comparison_uncertainty_followup_keeps_both_evidence_subjects() -> None:
    store, session = _store_with_turn("박용희와 송내호를 각각 설명해 줘.")
    store.add_evidence_turn(
        session.session_id,
        user="박용희와 송내호를 각각 설명해 줘.",
        active_place="",
        active_topic="박용희",
        active_subject="박용희",
        answered_intent="overview",
        chunk_ids=("park", "song"),
    )

    resolved = ConversationContextResolver().resolve(
        "확실하지 않은 부분은 뭐야?",
        session,
        current_place_id=None,
        current_piece_id=None,
    )

    assert "박용희" in resolved.recent_entities
    assert "송내호" in resolved.recent_entities
