from pathlib import Path
from types import SimpleNamespace

import pytest

from history_chatbot.chat.service import create_development_orchestrator
from history_chatbot.chat.service import ChatApplicationService
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.persona import OutputDomain
from history_chatbot.dialogue.track_models import (
    FreeChatUiState, ModeTransition, PieceChatUiState, RequestState,
    SharedSessionContext,
)


def chat(tmp_path: Path):
    return create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "sessions.json",
    )


def test_chat_mode_is_strict() -> None:
    assert {value.value for value in ConversationMode} == {"piece_chat", "free_chat"}
    with pytest.raises(ValueError):
        ConversationMode("embedded_chat")


def test_reference_ui_and_request_states_are_strict() -> None:
    assert PieceChatUiState.PAUSED.value == "paused"
    assert FreeChatUiState.RETURNING_TO_GAME.value == "returning_to_game"
    assert RequestState.CAPABILITY_UNAVAILABLE.value == "capability_unavailable"
    with pytest.raises(ValueError):
        PieceChatUiState("saving")


def test_shared_context_filters_to_completed_pieces() -> None:
    context = SharedSessionContext(completed_piece_ids=("piece-1", "piece-2"))
    assert context.completed_only(("piece-2", "piece-3")) == ("piece-2",)
    with pytest.raises(ValueError, match="중복"):
        SharedSessionContext(completed_piece_ids=("piece-1", "piece-1"))


def test_piece_emotion_is_short_no_rag_and_only_one_follow_up(tmp_path) -> None:
    engine = chat(tmp_path)
    first = engine.ask("인상 깊었어요.", conversation_mode="piece_chat")
    second = engine.ask(
        "인상 깊었어요.", session_id=first.session_id,
        conversation_mode="piece_chat", piece_follow_up_count=1,
    )
    assert not first.rag_used and first.citations == ()
    assert first.follow_up_question
    assert second.follow_up_question is None
    assert first.game_state_mutation is False
    assert first.chat_mode == "piece_chat"


def test_piece_follow_up_is_bounded_by_server_session(tmp_path) -> None:
    engine = chat(tmp_path)
    first = engine.ask("인상 깊었어요.", conversation_mode="piece_chat")
    second = engine.ask(
        "또 기억에 남아요.", session_id=first.session_id,
        conversation_mode="piece_chat",
    )
    assert first.follow_up_question
    assert second.follow_up_question is None


def test_piece_low_engagement_can_skip_without_pressure(tmp_path) -> None:
    response = chat(tmp_path).ask("딱히 없어요.", conversation_mode="piece_chat")
    assert response.next_action_code == "SKIP_REFLECTION"
    assert "건너뛰" in response.answer
    assert not response.rag_used


def test_piece_fatigue_offers_pause_without_persisting(tmp_path) -> None:
    response = chat(tmp_path).ask("여행 와서 좀 지쳤어요.", conversation_mode="piece_chat")
    assert response.next_action_code == "PAUSE_JOURNEY"
    assert "current_fatigue" in response.context_state or any(
        item["tag"] == "current_fatigue" for item in response.personalization_tag_candidates
    )
    assert not response.storage_permitted and not response.game_state_mutation
    assert not response.rag_used


def test_detailed_piece_question_preserves_transition_context(tmp_path) -> None:
    response = chat(tmp_path).ask(
        "여기서 일했던 사람들은 누구예요? 출처도 자세히 알려주세요.",
        conversation_mode="piece_chat", current_place_id="place-1",
        current_piece_id="piece-2", visited_piece_ids=("piece-1",),
        return_target="piece-overlay",
    )
    transition = response.mode_transition
    assert response.next_action_code == "OPEN_FREE_CHAT"
    assert transition is not None
    assert transition["pending_user_question"].startswith("여기서 일했던")
    assert transition["current_place_id"] == "place-1"
    assert transition["completed_piece_ids"] == ("piece-1",)
    assert transition["return_target"] == "piece-overlay"
    assert transition["preserve_game_state"] is True
    assert not response.rag_used and not response.game_state_mutation

    restored = ChatApplicationService(chat(tmp_path)).chat({
        "conversation_mode": "free_chat", "mode_transition": transition,
        "visited_piece_ids": transition["completed_piece_ids"],
        "current_place_id": transition["current_place_id"],
        "current_piece_id": transition["current_piece_id"],
    })
    assert restored["chat_mode"] == "free_chat"
    assert restored["rag_used"] is True


def test_free_chat_fact_uses_rag_and_citations(tmp_path) -> None:
    response = chat(tmp_path).ask(
        "붉은 등대 전시관은 언제 만들어졌어요?", conversation_mode="free_chat",
        current_place_id="place-1", visited_piece_ids=("piece-1",),
    )
    assert response.rag_used and response.grounded
    assert response.citations
    assert response.next_action_code == "ANSWER_WITH_CITATIONS"
    assert response.ui_state == "showing_citations"
    assert not response.game_state_mutation
    assert response.response_text == response.answer


def test_docent_repetition_guard_removes_exact_duplicates() -> None:
    answer = "A입니다. A입니다. A입니다."
    assert ConversationalRagOrchestrator._apply_repetition_guard(
        answer, output_domain=OutputDomain.HISTORICAL_DOCENT,
    ) == "A입니다."


def test_docent_repetition_guard_removes_only_very_close_variants() -> None:
    answer = "1929년 학생들은 시위를 준비했습니다. 1929년, 학생들은 시위를 준비했습니다."
    assert ConversationalRagOrchestrator._apply_repetition_guard(
        answer, output_domain=OutputDomain.HISTORICAL_DOCENT,
    ) == "1929년 학생들은 시위를 준비했습니다."


@pytest.mark.parametrize(
    "answer",
    (
        "11월 16일 전단을 작성했습니다. 11월 19일 전단 1,500매를 인쇄했습니다.",
        "전단 1,000매를 인쇄했습니다. 전단 1,500매를 인쇄했습니다.",
        "박종식은 시위를 준비했습니다. 오상록은 전단 배포를 준비했습니다.",
        "학생들은 전단을 작성했습니다. 학생들은 거리 시위를 전개했습니다.",
    ),
)
def test_docent_repetition_guard_preserves_distinct_facts(answer: str) -> None:
    assert ConversationalRagOrchestrator._apply_repetition_guard(
        answer, output_domain=OutputDomain.HISTORICAL_DOCENT,
    ) == answer


@pytest.mark.parametrize(
    "domain",
    (OutputDomain.CHARACTER_DIALOGUE, OutputDomain.SYSTEM_UI),
)
def test_repetition_guard_does_not_change_other_domains(domain: OutputDomain) -> None:
    answer = "같은 문장이야.  같은 문장이야."
    assert ConversationalRagOrchestrator._apply_repetition_guard(
        answer, output_domain=domain,
    ) == answer


def test_historical_docent_generation_path_applies_repetition_guard(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = chat(tmp_path)
    monkeypatch.setattr(
        engine.llm,
        "complete",
        lambda request: SimpleNamespace(
            generated_text="목포의 역사 설명입니다. 목포의 역사 설명입니다."
        ),
    )
    response = engine.ask(
        "붉은 등대 전시관은 언제 만들어졌어요?", conversation_mode="free_chat",
    )
    assert response.output_domain == "historical_docent"
    assert response.answer == "목포의 역사 설명입니다."


def test_free_chat_source_request_exposes_citation_panel_action(tmp_path) -> None:
    response = chat(tmp_path).ask("붉은 등대 전시관 출처가 어디예요?", conversation_mode="free_chat")
    assert response.rag_used and response.citations
    assert response.next_action_code == "OPEN_CITATION_PANEL"


def test_free_chat_greeting_is_no_rag_with_suggestions(tmp_path) -> None:
    response = chat(tmp_path).ask("안녕하세요", conversation_mode="free_chat")
    assert not response.rag_used and response.citations == ()
    assert response.suggested_questions
    assert response.chat_mode == "free_chat"


def test_track_aware_stream_returns_common_contract(tmp_path) -> None:
    service = ChatApplicationService(chat(tmp_path))
    events = list(service.stream({"user_query": "안녕하세요", "conversation_mode": "free_chat"}))
    assert len(events) == 1 and events[0].event == "completed"
    assert events[0].data["chat_mode"] == "free_chat"
    assert events[0].data["game_state_mutation"] is False


def test_free_chat_insufficient_evidence_has_no_fake_citation(tmp_path) -> None:
    response = chat(tmp_path).ask("서울 궁궐의 왕은 누구야?", conversation_mode="free_chat")
    assert response.status == "insufficient_evidence"
    assert response.request_state == "insufficient_evidence"
    assert response.citations == () and response.refusal_reason == "insufficient_evidence"
    assert not response.game_state_mutation


def test_storage_needs_capability_and_consent(tmp_path) -> None:
    engine = chat(tmp_path)
    base = engine.ask("감상을 저장해 주세요", conversation_mode="piece_chat")
    assert not base.storage_permitted
    assert "저장했다" not in base.answer and "기록했다" not in base.answer
    permitted = engine.ask(
        "감상을 저장해 주세요", conversation_mode="piece_chat",
        storage_capability=True, user_consent=True,
        available_capabilities=("SAVE_SHORT_REFLECTION",),
    )
    assert permitted.storage_permitted
    assert permitted.next_action_code == "SAVE_SHORT_REFLECTION"


def test_v03_fallback_remains_no_rag_in_both_modes(tmp_path) -> None:
    engine = chat(tmp_path)
    for mode in ConversationMode:
        response = engine.ask("휠체어로 갈 수 있어요?", conversation_mode=mode.value)
        assert not response.rag_used and response.citations == ()
        assert not response.capability_supported and response.fallback_used
        assert response.game_state_mutation is False


def test_transition_factory_preserves_return_position() -> None:
    context = SharedSessionContext(
        session_id="session-1", current_place_id="place-1",
        current_piece_id="piece-1", completed_piece_ids=("piece-0",),
    )
    transition = ModeTransition.open_free_chat(
        question="원래 질문", context=context, return_target="overlay",
    )
    assert transition.pending_user_question == "원래 질문"
    assert transition.source_session_id == "session-1"
    assert transition.return_target == "overlay"
    assert transition.preserve_game_state
    returning = ModeTransition.return_to_game(context=context, return_target="overlay")
    assert returning.from_mode == ConversationMode.FREE_CHAT
    assert returning.to_mode == ConversationMode.PIECE_CHAT
    assert returning.current_piece_id == "piece-1"
    assert returning.preserve_game_state
