from history_chatbot.dialogue.response_policy import GiroksaeDialogueEngine
from history_chatbot.dialogue.situation_models import ClassificationInput, ScreenType, SituationId


def decide(message, **kwargs):
    return GiroksaeDialogueEngine().decide(ClassificationInput(message, **kwargs))


def test_low_engagement_does_not_force_reflection_or_rag() -> None:
    result = decide("딱히 없어요.", screen_type=ScreenType.PIECE_CHAT)
    assert not result.should_retrieve
    assert result.follow_up_question is None
    assert "다음 단계" in result.answer


def test_dissatisfaction_does_not_echo_abuse_or_lecture() -> None:
    result = decide("ㅋㅋ 지랄.")
    assert "지랄" not in result.answer
    assert not result.should_retrieve
    assert len(result.answer) < 100


def test_fatigue_gets_short_session_response() -> None:
    result = decide("여행 와서 좀 지쳤어요.")
    assert "한두 문장" in result.answer
    assert not result.should_retrieve


def test_fact_question_routes_to_retrieval_without_lookup_placeholder() -> None:
    result = decide("이 건물은 언제 만들어졌어요?")
    assert result.should_retrieve and result.should_call_llm
    assert result.answer == ""


def test_clarification_blocks_retrieval_and_llm() -> None:
    result = decide("그게 왜 그런 거예요?")
    assert result.classification.requires_clarification
    assert not result.should_retrieve and not result.should_call_llm
    assert result.follow_up_question


def test_korean_non_rag_answers_keep_polite_style() -> None:
    result = decide("안녕하세요")
    assert "예요" in result.answer or "요" in result.answer
    assert "야." not in result.answer


def test_zh_cn_locale_hook_is_accepted_without_translation_generation() -> None:
    result = decide("你好", locale="zh-CN")
    assert result.classification.primary_situation_id == SituationId.PERSONAL_AND_LIGHT_CHAT
