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


def test_bare_interrogative_gets_intent_specific_clarification() -> None:
    result = decide("언제?")
    assert not result.should_retrieve and not result.should_call_llm
    assert result.answer == "어떤 사건이나 장소의 시점을 알고 싶은가요?"


def test_gratitude_gets_conversational_response_without_rag() -> None:
    result = decide("고마워")
    assert not result.should_retrieve
    assert "천만에" in result.answer


def test_out_of_scope_question_redirects_to_history_without_rag() -> None:
    result = decide("오늘 날씨 어때?")
    assert not result.should_retrieve
    assert "역사 이야기를 중심" in result.answer


def test_korean_character_answers_use_final_banmal_style() -> None:
    result = decide("안녕하세요")
    assert "기록새야" in result.answer
    assert "예요" not in result.answer and "습니다" not in result.answer


def test_zh_cn_locale_hook_is_accepted_without_translation_generation() -> None:
    result = decide("你好", locale="zh-CN")
    assert result.classification.primary_situation_id == SituationId.PERSONAL_AND_LIGHT_CHAT


def test_technical_help_without_provider_uses_generic_fallback() -> None:
    result = decide("사진이 안 겹쳐져요.")
    assert not result.should_retrieve and result.citations == ()
    assert not result.capability_supported and result.fallback_used
    assert result.missing_context == ("app_state",)
    assert "구체적인 버튼이나 아이콘 위치는 안내할 수 없습니다" in result.response_text


def test_navigation_without_context_does_not_invent_route() -> None:
    result = decide("여기서 얼마나 걸려요?")
    assert not result.should_retrieve
    assert result.missing_context == ("current_location", "map_data")
    assert "안내할 수 없습니다" in result.answer
    assert result.next_action_code == "CALCULATE_ROUTE_ETA"


def test_journey_state_missing_does_not_name_next_piece() -> None:
    result = decide("다음 조각은 어디예요?")
    assert "다음 조각 위치를 안내할 수 없습니다" in result.answer
    assert result.missing_context == ("journey_state", "map_data")


def test_accessibility_without_verified_data_does_not_assert_access() -> None:
    result = decide("휠체어로 갈 수 있어요?")
    assert result.classification.primary_situation_id == SituationId.SAFETY_ACCESSIBILITY
    assert not result.should_retrieve and result.citations == ()
    assert "단정할 수 없습니다" in result.answer
    assert result.missing_context == ("verified_facility_data",)


def test_capability_requires_explicit_provider_support_even_with_context() -> None:
    result = decide("소리가 안 나요.", app_state={"audio": "unknown"})
    assert result.missing_context == ()
    assert not result.capability_supported
    assert result.fallback_used


def test_storage_claim_requires_capability_and_consent() -> None:
    engine = GiroksaeDialogueEngine()
    assert not engine.can_claim_persisted(ClassificationInput("기억해 줘"))
    assert not engine.can_claim_persisted(ClassificationInput("기억해 줘", storage_capability=True))
    assert engine.can_claim_persisted(ClassificationInput("기억해 줘", storage_capability=True, user_consent=True))
