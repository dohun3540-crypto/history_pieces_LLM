from pathlib import Path

import pytest

from history_chatbot.chat.context_resolver import ConversationContextResolver
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.prompt_builder import SYSTEM_INSTRUCTIONS
from history_chatbot.chat.service import create_development_orchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.dialogue.persona import OutputDomain
from history_chatbot.evaluation.conversation_quality import (
    answer_is_complete,
    load_dataset_directory,
    validate_splits,
)
from history_chatbot.models.remote import RemoteLLMError
from history_chatbot.runtime import RuntimeMode


DATASET = Path("evaluation/conversation_quality")


def seeded_session(user: str = "목포역에서 있었던 사건을 알려줘"):
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, user, "assistant 자유 생성 문장")
    store.update_context(
        session.session_id,
        active_place="목포역",
        active_topic="목포역",
        recent_entities=("목포역",),
        recent_event="사건",
    )
    return store, session


def resolved(query: str):
    _, session = seeded_session()
    return ConversationContextResolver().resolve(
        query, session, current_place_id=None, current_piece_id=None
    )


def test_followup_who() -> None:
    value = resolved("관련 인물은?")
    assert value.followup_resolved
    assert "목포역" in value.search_query


def test_followup_when() -> None:
    value = resolved("언제 지었어?")
    assert value.followup_resolved
    assert value.search_query.startswith("목포역 사건")


def test_followup_construction_period_keeps_evidence_topic() -> None:
    store, session = seeded_session("근대역사관1관의 역사적 성격을 알려 줘")
    store.update_context(session.session_id, active_topic="근대역사관1관")

    value = ConversationContextResolver().resolve(
        "건립 시기는?", session, current_place_id=None, current_piece_id=None
    )

    assert value.followup_resolved
    assert value.search_query.startswith("목포역 사건 근대역사관1관")


def test_followup_why() -> None:
    value = resolved("왜 왔던 거야?")
    assert value.followup_resolved
    assert "목포역" in value.search_query


def test_pronoun_reference() -> None:
    value = resolved("그 사람은 이후 어떻게 됐어?")
    assert value.followup_resolved
    assert "목포역" in value.search_query


def test_topic_switch() -> None:
    value = resolved("됐고 동양척식주식회사 이야기해줘.")
    assert value.followup_resolved is False
    assert value.active_place == "동양척식주식회사"
    assert "목포역" not in value.search_query


def test_return_to_previous_topic() -> None:
    _, session = seeded_session("동양척식주식회사 이야기해줘")
    value = ConversationContextResolver().resolve(
        "아까 목포역 이야기로 돌아가자.", session,
        current_place_id=None, current_piece_id=None,
    )
    assert value.active_place == "목포역"
    assert value.search_query.startswith("목포역")


def test_partial_answer_fallback() -> None:
    chunk = type("Ranked", (), {
        "chunk": type("Chunk", (), {"text": "목포역은 철도 교통의 장소였다."})()
    })()
    assert not ConversationalRagOrchestrator._supports_requested_detail(
        "목포역 내부 천장 색깔은 뭐였어?", [chunk]
    )


def test_unanswerable_fallback() -> None:
    answer, suggestions = ConversationalRagOrchestrator._insufficient_guidance(
        "책임자는 누구였어?", OutputDomain.HISTORICAL_DOCENT, "ko",
        active_place="목포진",
    )
    assert "직접 연결되는 인물" in answer
    assert "추측" not in answer
    assert len(suggestions) == 1


def test_out_of_scope(tmp_path: Path) -> None:
    chat = create_development_orchestrator(runtime_dir=tmp_path)
    response = chat.ask("목포에 양자컴퓨터 공장이 있었어?")
    assert response.status == "ok"
    assert "역사 이야기를 중심" in response.answer
    assert not response.grounded and not response.sources


def test_false_premise() -> None:
    assert "전제가 검색 근거와 다르거나 근거에 없으면 동조하지" in SYSTEM_INSTRUCTIONS


def test_answer_completeness() -> None:
    assert answer_is_complete("완결된 답변입니다.")
    assert not answer_is_complete("주한 미국 대사 존 무초(John J.")
    with pytest.raises(RemoteLLMError, match="열린 괄호"):
        ConversationalRagOrchestrator._completion_text_values(
            "[답변] 주한 미국 대사 존 무초(John J.", "stop"
        )
    answer, warnings = ConversationalRagOrchestrator._completion_text_values(
        "이범석은 독립운동가였습니다. 행사에는 존 무초(John J.", "length"
    )
    assert answer == "이범석은 독립운동가였습니다."
    assert warnings == ("generation_truncated_at_sentence_boundary",)


def test_no_repeated_hard_fallback() -> None:
    people, _ = ConversationalRagOrchestrator._insufficient_guidance(
        "누구였어?", OutputDomain.HISTORICAL_DOCENT, "ko", active_place="목포역"
    )
    date, _ = ConversationalRagOrchestrator._insufficient_guidance(
        "언제였어?", OutputDomain.HISTORICAL_DOCENT, "ko", active_place="목포역"
    )
    assert people != date


def test_no_assistant_text_as_evidence() -> None:
    store, session = seeded_session()
    value = ConversationContextResolver().resolve(
        "왜 왔던 거야?", session, current_place_id=None, current_piece_id=None
    )
    assert "assistant 자유 생성 문장" not in value.search_query
    assert store.get(session.session_id).evidence_turns == []


def test_dataset_splits_and_permissions() -> None:
    scenarios = load_dataset_directory(DATASET)
    counts = validate_splits(scenarios)
    assert counts == {
        "holdout_test_scenarios": 3,
        "train_dev_scenarios": 4,
        "validation_scenarios": 3,
        "holdout_test_turns": 8,
        "train_dev_turns": 11,
        "validation_turns": 7,
    }
    assert all(item["training_eligible"] is False for item in scenarios)
