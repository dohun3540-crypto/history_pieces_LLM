from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from history_chatbot.chat.context_resolver import (
    ConversationContextResolver,
    is_placeholder_context,
)
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.dialogue.persona import DOCENT_PROMPT, OutputDomain
from history_chatbot.chat.service import create_hackathon_orchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.history_collection.verified_corpus import build_verified_corpus
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.runtime import RuntimeMode


def _candidate(root: Path, index: int) -> dict[str, object]:
    document_id = f"candidate-{index:03d}"
    text = f"제목: 목포역 역사 자료 {index}\n기관: 공공 역사 기관\n\n" + " ".join(
        "목포역은 근대 목포의 철도와 항만 교통을 연결한 역사적 장소입니다. "
        "일제강점기 당시 목포의 도시 형성과 상업 활동, 학생운동의 이동 경로를 "
        f"이해하는 데 필요한 기록 {part}입니다. 목포항과 호남선의 변천도 함께 설명합니다."
        for part in range(7)
    )
    raw = f"<main>{text}</main>".encode()
    extracted = text.encode()
    raw_path = root / f"raw/{document_id}.html"
    extracted_path = root / f"extracted/{document_id}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    extracted_path.write_bytes(extracted)
    url = f"https://history.example.org/item/{index}"
    return {
        "candidate_id": document_id,
        "document_id": document_id,
        "source_id": "official_test",
        "source_tier": "tier_1",
        "institution": "공공 역사 기관",
        "publisher": "공공 역사 기관",
        "publisher_family": "official_test",
        "source_title": f"목포역 역사 자료 {index}",
        "source_url": url,
        "canonical_url": url,
        "raw_path": str(raw_path.relative_to(root)),
        "extracted_path": str(extracted_path.relative_to(root)),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "extracted_sha256": hashlib.sha256(extracted).hexdigest(),
        "extraction_status": "success",
        "duplicate_status": "new_unique",
        "rights_status": "unknown",
        "provenance": {"new_unique_increment": 1},
    }


def test_verified_builder_selects_only_valid_and_preserves_rights(tmp_path: Path) -> None:
    rows = [_candidate(tmp_path, index) for index in range(100)]
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = build_verified_corpus(root=tmp_path, candidate_manifest=manifest, output_root=tmp_path / "verified")
    assert report["document_count"] == 100
    chunks = [json.loads(line) for line in (tmp_path / "verified/index_ready/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks
    assert all(item["verification_status"] == "VALID" for item in chunks)
    assert all(item["rights_status"] == "unknown" for item in chunks)
    assert all(item["human_review_required"] is True for item in chunks)
    assert all(item["production_approved"] is False for item in chunks)


def test_context_resolver_uses_place_and_recent_user_topic_only() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포 학생운동은 어떻게 진행됐어?", "근거 기반 응답")
    store.update_context(session.session_id, recent_event="학생운동", recent_entities=("목포 학생운동",))
    resolved = ConversationContextResolver().resolve(
        "그때 여기서는 무슨 일이 있었어?", session,
        current_place_id="mokpo-station", current_piece_id=None,
    )
    assert resolved.followup_resolved
    assert "목포역" in resolved.search_query
    assert "학생운동" in resolved.search_query
    assert "근거 기반 응답" not in resolved.search_query


def test_hackathon_factory_uses_verified_lane_and_multiturn(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = [_candidate(tmp_path, index) for index in range(100)]
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "verified"
    build_verified_corpus(root=tmp_path, candidate_manifest=manifest, output_root=output)
    chat = create_hackathon_orchestrator(
        runtime_dir=tmp_path / "runtime",
        chunks_path=output / "index_ready/chunks.jsonl",
        session_path=tmp_path / "sessions.json",
        llm=MockLLM("목포역 관련 기록을 근거로 설명합니다."),
    )
    search_queries: list[str] = []
    real_search = chat.retrieval.search

    def capture_search(query: str):
        search_queries.append(query)
        return real_search(query)

    monkeypatch.setattr(chat.retrieval, "search", capture_search)
    first = chat.ask(
        "목포역에 대해 설명해줘",
        current_place_id="demo-place",
        current_piece_id="demo-piece-1",
    )
    second = chat.ask(
        "그 당시에는 어떤 사람들이 이용했어?",
        session_id=first.session_id,
        current_place_id="demo-place",
        current_piece_id="demo-piece-1",
    )
    # The verified fixture describes the station, but contains no evidence
    # identifying its users.  Preserve the grounded limitation instead of
    # presenting a nearby station fact as an answer about people.
    assert second.status == "partial_evidence"
    assert second.context_metadata["grounded_fact_count"] > 0
    assert second.context_metadata["followup_resolved"] is True
    assert "목포역" in second.context_metadata["search_query"]
    assert "demo" not in second.context_metadata["search_query"]
    assert second.grounded is True
    assert second.source_sufficiency == "partial"
    assert second.context_metadata["evidence_support"] == "partial"
    assert second.context_metadata["retrieval_performed"] is True
    assert second.sources
    assert len(search_queries) == 2
    assert chat.retrieval.store.metadata()["data_lane"] == "verified_hackathon"

    shortage = chat.ask(
        "당시 목포역 내부 모습은 어땠어?",
        session_id=first.session_id,
        current_place_id="demo-place",
        current_piece_id="demo-piece-1",
    )
    assert shortage.status == "ok"
    assert shortage.source_sufficiency == "partial"
    assert shortage.context_metadata["requested_detail_supported"] is False
    assert shortage.sources
    assert len(search_queries) == 3
    assert shortage.context_metadata["retrieval_retry_performed"] is False


def test_place_change_replaces_active_place() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    resolver = ConversationContextResolver()
    first = resolver.resolve("여기는 왜 중요해?", session, current_place_id="mokpo-station", current_piece_id=None)
    store.update_context(session.session_id, active_place=first.active_place)
    second = resolver.resolve("여기는 왜 중요해?", session, current_place_id="mokpo-port", current_piece_id=None)
    assert second.active_place == "목포항"
    assert second.search_query.startswith("목포항")
    assert "목포역" not in second.search_query


def test_placeholder_context_never_overrides_explicit_or_conversational_place() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    resolver = ConversationContextResolver()
    store.update_context(
        session.session_id,
        active_place="demo-place",
        active_piece="demo-piece-1",
        active_topic="test-place",
        recent_entities=("demo-place",),
        recent_event="placeholder",
        recent_period="unknown",
    )

    first = resolver.resolve(
        "목포역에 대해 설명해줘", session,
        current_place_id="demo-place", current_piece_id="demo-piece-1",
    )
    store.update_context(
        session.session_id,
        active_place=first.active_place,
        active_piece=first.active_piece,
        active_topic=first.active_topic,
        recent_entities=first.recent_entities,
    )
    store.add_turn(session.session_id, "목포역에 대해 설명해줘", "검색 근거 기반 응답")
    second = resolver.resolve(
        "그 당시에는 어떤 사람들이 이용했어?", session,
        current_place_id="demo-place", current_piece_id="demo-piece-1",
    )

    assert is_placeholder_context("demo-place")
    assert is_placeholder_context("demo-piece-12")
    assert is_placeholder_context("test-place")
    assert is_placeholder_context("placeholder")
    assert is_placeholder_context("unknown")
    assert second.active_place == "목포역"
    assert second.active_piece == ""
    assert second.active_topic == "목포역"
    assert second.recent_event == ""
    assert second.recent_period == ""
    assert "목포역" in second.search_query
    assert "어떤 사람들이 이용" in second.search_query
    assert "demo" not in second.search_query


def test_real_journey_place_resolves_here_but_placeholder_uses_conversation() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포 학생운동에 대해 알려줘", "근거 기반 응답")
    store.update_context(
        session.session_id,
        active_place="목포역",
        active_topic="학생운동",
        recent_event="학생운동",
        recent_entities=("목포 학생운동",),
    )
    resolver = ConversationContextResolver()

    real = resolver.resolve(
        "아까 말한 학생운동은 여기에서도 일어났어?", session,
        current_place_id="mokpo-port", current_piece_id=None,
    )
    placeholder = resolver.resolve(
        "아까 말한 학생운동은 여기에서도 일어났어?", session,
        current_place_id="demo-place", current_piece_id="demo-piece-1",
    )

    assert real.active_place == "목포항"
    assert "목포항" in real.search_query
    assert "학생운동" in real.search_query
    assert placeholder.active_place == "목포역"
    assert "목포역" in placeholder.search_query
    assert "학생운동" in placeholder.search_query
    assert "demo" not in placeholder.search_query


def test_recent_event_survives_place_question_and_resolves_then() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    resolver = ConversationContextResolver()
    first = resolver.resolve(
        "목포 학생운동에 대해 알려줘", session,
        current_place_id=None, current_piece_id=None,
    )
    store.update_context(
        session.session_id,
        active_place=first.active_place,
        active_topic=first.active_topic,
        recent_entities=first.recent_entities,
        recent_event=first.recent_event,
    )
    store.add_turn(session.session_id, "목포 학생운동에 대해 알려줘", "근거 기반 응답")
    second = resolver.resolve(
        "목포역과도 관련이 있어?", session,
        current_place_id="demo-place", current_piece_id=None,
    )
    store.update_context(
        session.session_id,
        active_place=second.active_place,
        active_topic=second.active_topic,
        recent_entities=second.recent_entities,
        recent_event=second.recent_event,
    )
    store.add_turn(session.session_id, "목포역과도 관련이 있어?", "근거 기반 응답")
    third = resolver.resolve(
        "그때 누가 참여했어?", session,
        current_place_id="demo-place", current_piece_id=None,
    )

    assert third.active_place == "목포역"
    assert third.recent_event == "학생운동"
    assert "목포역" in third.search_query
    assert "학생운동" in third.search_query
    assert "누가 참여" in third.search_query
    assert "demo" not in third.search_query


def test_explicit_place_and_person_override_prior_context() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.update_context(
        session.session_id,
        active_place="목포역",
        active_topic="목포역",
        recent_entities=("목포역",),
    )
    resolver = ConversationContextResolver()

    place = resolver.resolve(
        "구 일본영사관은 어떤 곳이야?", session,
        current_place_id="demo-place", current_piece_id=None,
    )
    assert place.active_place == "구 일본영사관"
    assert "목포역" not in place.search_query

    person = resolver.resolve(
        "오상록은 어떤 사람이야?", session,
        current_place_id="demo-place", current_piece_id=None,
    )
    store.update_context(
        session.session_id,
        active_topic=person.active_topic,
        recent_entities=person.recent_entities,
    )
    store.add_turn(session.session_id, "오상록은 어떤 사람이야?", "검증 근거 기반 응답")
    followup = resolver.resolve(
        "그 사람은 이후 어떻게 됐어?", session,
        current_place_id="demo-place", current_piece_id=None,
    )
    assert followup.recent_entities[0] == "오상록"
    assert "오상록" in followup.search_query
    assert "검증 근거 기반 응답" not in followup.search_query


def test_placeholder_is_removed_from_historical_prompt_context() -> None:
    contextual = ConversationalRagOrchestrator._contextualize_query(
        "목포역에 대해 설명해줘",
        current_place_id="demo-place",
        current_piece_id="demo-piece-1",
        completed_place_ids=("test-place",),
        completed_piece_ids=("demo-piece-1",),
    )
    scoped = ConversationalRagOrchestrator._journey_scoped_query(
        "여기에서도 일어났어?", "JOURNEY_CONTEXT_QUESTION", ("demo-piece-1",)
    )
    assert "demo" not in contextual
    assert "test-place" not in contextual
    assert "demo" not in scoped


def test_narrow_interior_question_requires_matching_evidence() -> None:
    chunk = type("Ranked", (), {
        "chunk": type("Chunk", (), {"text": "목포역은 철도와 항만을 연결했다."})()
    })()
    assert not ConversationalRagOrchestrator._supports_requested_detail(
        "당시 목포역 내부 모습은 어땠어?", [chunk]
    )
    assert ConversationalRagOrchestrator._supports_requested_detail(
        "목포역의 역사적 역할은 무엇이야?", [chunk]
    )


def test_docent_prompt_prioritizes_relevance_over_unsolicited_chronology() -> None:
    assert "역사적 역할과 장소의 의미를 먼저" in DOCENT_PROMPT
    assert "연혁·연도별 정리·시기별 변화" in DOCENT_PROMPT
    assert "연표식 답변을 피한다" in DOCENT_PROMPT


@pytest.mark.parametrize(
    "followup",
    ("왜?", "언제였어?", "그럼 결과는?"),
)
def test_elliptical_followup_reuses_last_user_question(followup: str) -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(
        session.session_id,
        "목포 학생운동은 어떻게 전개됐어?",
        "검색 근거에 따른 답변",
    )
    resolved = ConversationContextResolver().resolve(
        followup, session,
        current_place_id="demo-place", current_piece_id="demo-piece-1",
    )
    assert resolved.followup_resolved is True
    assert resolved.search_query == f"목포 학생운동은 어떻게 전개됐어? {followup}"
    assert "검색 근거에 따른 답변" not in resolved.search_query
    assert "demo" not in resolved.search_query


def test_answer_transformation_is_not_rewritten_as_a_new_search_question() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(
        session.session_id,
        "목포 학생운동은 어떻게 전개됐어?",
        "검색 근거에 따른 답변",
    )
    resolved = ConversationContextResolver().resolve(
        "좀 더 쉽게 설명해줘", session,
        current_place_id="demo-place", current_piece_id="demo-piece-1",
    )
    assert resolved.followup_resolved is True
    assert resolved.search_query == "좀 더 쉽게 설명해줘"
    assert resolved.needs_new_evidence is False


def test_explicit_short_question_does_not_inherit_previous_subject() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포역은 언제 생겼어?", "근거 기반 응답")
    resolved = ConversationContextResolver().resolve(
        "유달산은 왜 유명해?", session,
        current_place_id=None, current_piece_id=None,
    )
    assert resolved.followup_resolved is False
    assert resolved.search_query == "유달산은 왜 유명해?"
    assert "목포역" not in resolved.search_query


def test_resolved_followup_is_labeled_as_non_evidence_for_generation() -> None:
    interpreted = ConversationalRagOrchestrator._conversation_scoped_query(
        "왜?",
        search_query="목포 학생운동은 어떻게 전개됐어? 왜?",
        followup_resolved=True,
    )
    assert interpreted.startswith("왜?\n")
    assert "대화 문맥 해석 | 역사적 사실의 근거가 아님" in interpreted
    assert "목포 학생운동" in interpreted
    assert ConversationalRagOrchestrator._conversation_scoped_query(
        "유달산은 왜 유명해?",
        search_query="유달산 유달산은 왜 유명해?",
        followup_resolved=False,
    ) == "유달산은 왜 유명해?"


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("이 장소는 언제 만들어졌나요?", "정확한 시점"),
        ("관련 인물은 누구인가요?", "직접 연결되는 인물"),
    ),
)
def test_insufficient_guidance_uses_place_and_question_intent(
    query: str, expected: str,
) -> None:
    answer, suggestions = ConversationalRagOrchestrator._insufficient_guidance(
        query,
        OutputDomain.HISTORICAL_DOCENT,
        "ko",
        active_place="목포역",
    )
    assert expected in answer
    assert "현재 확보된" not in answer
    assert "추측하지" not in answer
    assert "목포역" in answer
    assert len(suggestions) == 1
    assert all("목포역" in item for item in suggestions)
