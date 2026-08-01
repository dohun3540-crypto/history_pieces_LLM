import pytest

from history_chatbot.dialogue.situation_classifier import SituationClassifier
from history_chatbot.dialogue.situation_models import ClassificationInput, ResponseLengthMode, ScreenType, SituationId


def classify(message, **kwargs):
    return SituationClassifier().classify(ClassificationInput(message, **kwargs))


def test_simple_greeting_does_not_use_rag() -> None:
    result = classify("안녕하세요")
    assert result.primary_situation_id == SituationId.FREE_CHAT_GREETING
    assert not result.requires_rag


@pytest.mark.parametrize(
    ("message", "situation"),
    [
        ("이 건물 구조는 왜 이렇게 생겼어요?", SituationId.INTEREST_ARCHITECTURE),
        ("여기서 일했던 사람들은 누구예요?", SituationId.INTEREST_PEOPLE),
    ],
)
def test_historical_architecture_and_people_questions_require_rag(message, situation) -> None:
    result = classify(message)
    assert result.primary_situation_id == situation
    assert result.requires_rag


def test_positive_reflection_does_not_use_rag() -> None:
    assert not classify("인상 깊었어요.", screen_type=ScreenType.PIECE_CHAT).requires_rag


def test_abuse_overrides_other_topics_and_applies_very_short() -> None:
    result = classify("역사 설명 말 ㅈㄴ 많네")
    assert result.primary_situation_id == SituationId.STRONG_DISSATISFACTION
    assert result.response_length_mode == ResponseLengthMode.VERY_SHORT
    assert "prefers_very_short" in result.personalization_tag_candidates


def test_ambiguous_reference_without_context_clarifies() -> None:
    result = classify("그게 왜 그런 거예요?")
    assert result.requires_clarification
    assert not result.requires_rag


def test_source_request_is_style_plus_evidence_and_rag() -> None:
    result = classify("출처가 어디예요?")
    assert result.primary_situation_id == SituationId.RESPONSE_STYLE_REQUEST
    assert SituationId.EVIDENCE_AND_CORRECTION in result.secondary_situation_ids
    assert result.requires_rag
    assert result.response_length_mode == ResponseLengthMode.SOURCE_VIEW


def test_cross_cultural_does_not_infer_country() -> None:
    result = classify("우리나라 역사와 비슷한 점이 있나요?")
    assert result.primary_situation_id == SituationId.CROSS_CULTURAL_COMPARISON
    assert result.requires_clarification
    assert not result.requires_rag


def test_screen_context_selects_intro() -> None:
    result = classify("뭘 하면 돼?", screen_type=ScreenType.INTRO)
    assert result.primary_situation_id == SituationId.INTRO_GIROKSAE


def test_journey_summary_is_not_misclassified_as_style_only() -> None:
    result = classify("지금까지 본 거 정리해 줘.")
    assert result.primary_situation_id == SituationId.JOURNEY_CONTEXT_QUESTION


def test_chinese_question_punctuation_routes_to_fact_rag() -> None:
    result = classify("木浦港是哪一年开放的？", locale="zh-CN")
    assert result.primary_situation_id == SituationId.HISTORY_FACT_QUESTION
    assert result.requires_rag


@pytest.mark.parametrize(
    ("message", "situation"),
    [
        ("다음 버튼이 안 눌려요.", SituationId.TECHNICAL_HELP),
        ("다음 조각은 어디예요?", SituationId.NAVIGATION_HELP),
        ("휠체어로 갈 수 있어요?", SituationId.SAFETY_ACCESSIBILITY),
    ],
)
def test_v03_capability_classification_never_uses_history_rag(message, situation) -> None:
    result = classify(message)
    assert result.primary_situation_id == situation
    assert not result.requires_rag
    assert "no_rag" in result.policy_flags


def test_safety_overrides_history_or_reflection_topic() -> None:
    result = classify("이 건물 역사가 궁금한데 계단 말고 다른 길 있어요?")
    assert result.primary_situation_id == SituationId.SAFETY_ACCESSIBILITY
    assert not result.requires_rag
