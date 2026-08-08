from pathlib import Path
from types import SimpleNamespace

import pytest

from history_chatbot.chat.service import create_development_orchestrator
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.persona import (
    ConversationStage, GiroksaeStyleGuard, OutputDomain, PERSONA_ID,
    PERSONA_SOURCE, SourceSufficiency, SpeechLevel, TranslationStatus,
    build_persona_prompt, locale_policy, output_domain_for, speech_level_for,
)
from history_chatbot.dialogue.response_renderer import GiroksaeResponseRenderer
from history_chatbot.dialogue.situation_models import SituationId


def orchestrator(tmp_path: Path):
    return create_development_orchestrator(
        runtime_dir=tmp_path / "runtime", session_path=tmp_path / "sessions.json",
    )


def test_v11_is_the_only_canonical_runtime_persona_source() -> None:
    assert PERSONA_SOURCE == "docs/GIROKSAE_CHARACTER_PRINCIPLES_V11.md"
    assert Path(PERSONA_SOURCE).is_file()
    assert PERSONA_ID == "giroksae-v1.1"
    source = Path("src/history_chatbot/dialogue/persona.py").read_text(encoding="utf-8")
    assert "GIROKSAE_DIALOGUE_V03.md" not in source
    assert "GIROKSAE_PERSONA_V12_PROJECT_READY.md" not in source
    for legacy in ("docs/GIROKSAE_DIALOGUE.md", "docs/GIROKSAE_DIALOGUE_V03.md"):
        text = Path(legacy).read_text(encoding="utf-8")
        assert "superseded" in text
        assert "GIROKSAE_CHARACTER_PRINCIPLES_V11.md" in text


def test_output_domain_and_speech_level_are_strict() -> None:
    assert {item.value for item in OutputDomain} == {
        "character_dialogue", "system_ui", "historical_docent", "journey_film_caption",
    }
    assert speech_level_for(OutputDomain.CHARACTER_DIALOGUE) == SpeechLevel.BANMAL
    assert speech_level_for(OutputDomain.SYSTEM_UI) == SpeechLevel.POLITE_UI
    assert speech_level_for(OutputDomain.HISTORICAL_DOCENT) == SpeechLevel.FORMAL_DOCENT
    assert speech_level_for(OutputDomain.JOURNEY_FILM_CAPTION) == SpeechLevel.NEUTRAL_CAPTION
    with pytest.raises(ValueError):
        OutputDomain("mixed")


def test_system_help_and_character_domains_are_separate() -> None:
    assert output_domain_for(SituationId.TECHNICAL_HELP) == OutputDomain.SYSTEM_UI
    assert output_domain_for(SituationId.SAFETY_ACCESSIBILITY) == OutputDomain.SYSTEM_UI
    assert output_domain_for(SituationId.HISTORY_FACT_QUESTION) == OutputDomain.HISTORICAL_DOCENT
    assert output_domain_for(SituationId.INTEREST_PEOPLE) == OutputDomain.HISTORICAL_DOCENT
    assert output_domain_for(SituationId.FREE_CHAT_GREETING) == OutputDomain.CHARACTER_DIALOGUE


def test_prompt_is_composed_from_final_policy_by_context() -> None:
    character = build_persona_prompt(
        domain=OutputDomain.CHARACTER_DIALOGUE, locale="ko",
        mode=ConversationMode.FREE_CHAT, situation=SituationId.HISTORY_FACT_QUESTION,
    )
    system = build_persona_prompt(
        domain=OutputDomain.SYSTEM_UI, locale="ko",
        mode=ConversationMode.PIECE_CHAT, situation=SituationId.TECHNICAL_HELP,
    )
    assert PERSONA_SOURCE in character and "친근한 해체 중심 반말" in character
    assert "system_ui는 정중하고 기능 중심인 존댓말" in system
    assert "역사 사실" not in character  # facts are injected separately, not hard-coded here


def test_historical_docent_prompt_enforces_grounded_concise_answers() -> None:
    docent = build_persona_prompt(
        domain=OutputDomain.HISTORICAL_DOCENT, locale="ko",
        mode=ConversationMode.FREE_CHAT, situation=SituationId.HISTORY_FACT_QUESTION,
    )
    for unsupported_detail in ("인물명", "날짜", "연도", "숫자", "사건 관계"):
        assert unsupported_detail in docent
    assert "일반 지식으로 보완하지 않는다" in docent
    assert "첫 1~2문장에서 질문에 직접 답" in docent
    assert "같은 사실을 표현만 바꾸어 반복하지 않으며" in docent
    assert "동일하거나 거의 동일한 문장을 반복하지 않는다" in docent
    assert "어느 한쪽을 임의로 확정하지 말고" in docent
    assert "사건과 직접 관련된 역할을 먼저 설명" in docent
    assert "전체 생애나 전기를 길게 나열하지 않는다" in docent


def test_character_dialogue_prompt_does_not_inherit_docent_constraints() -> None:
    character = build_persona_prompt(
        domain=OutputDomain.CHARACTER_DIALOGUE, locale="ko",
        mode=ConversationMode.FREE_CHAT, situation=SituationId.FREE_CHAT_GREETING,
    )
    assert "친근한 해체 중심 반말" in character
    assert "검색 근거에 없는 인물명, 날짜, 연도, 숫자, 사건 관계" not in character
    assert "무관한 전체 생애나 전기를 길게 나열하지 않는다" not in character


def test_character_banmal_and_system_ui_polite_are_enforced(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    greeting = chat.ask("안녕하세요", conversation_mode="free_chat")
    assert greeting.output_domain == "character_dialogue"
    assert greeting.speech_level == "banmal"
    assert "기록새야" in greeting.answer
    assert not any(ending in greeting.answer for ending in ("습니다", "예요", "해요"))
    assert len(greeting.answer) <= 120
    assert 1 <= sum(greeting.answer.count(mark) for mark in ".?!") <= 3
    technical = chat.ask("다음 버튼이 안 눌려요.", conversation_mode="free_chat")
    assert technical.output_domain == "system_ui"
    assert technical.speech_level == "polite_ui"
    assert "습니다" in technical.answer or "주세요" in technical.answer
    assert technical.rag_used is False


def test_character_address_guard_is_context_aware() -> None:
    guard = GiroksaeStyleGuard()
    context = dict(
        domain=OutputDomain.CHARACTER_DIALOGUE,
        situation=SituationId.FREE_CHAT_GREETING,
        stage=ConversationStage.GREETING,
        locale="ko",
    )
    assert guard.validate("오래된 기록이 남아 있네.", **context) == ()
    codes = {item.code for item in guard.validate("너는 다시 촬영해야 한다.", **context)}
    assert "forbidden_user_address" in codes and "commanding_tone" in codes
    assert {item.code for item in guard.validate("이 감상은 100점이야.", **context)} >= {"user_rating"}


def test_historical_docent_accepts_polite_endings_but_character_guard_remains() -> None:
    guard = GiroksaeStyleGuard()
    historical = dict(
        situation=SituationId.HISTORY_FACT_QUESTION,
        stage=ConversationStage.HISTORICAL_QUESTION,
        locale="ko",
    )
    answer = "목포역을 중심으로 학생운동이 전개되었습니다. 관련 기록을 함께 확인합니다."
    assert guard.validate(answer, domain=OutputDomain.HISTORICAL_DOCENT, **historical) == ()
    assert {item.code for item in guard.validate(
        answer, domain=OutputDomain.CHARACTER_DIALOGUE, **historical,
    )} >= {"character_polite_ending"}


def test_great_giroksae_is_only_allowed_at_first_greeting() -> None:
    guard = GiroksaeStyleGuard()
    assert guard.validate(
        "난 오래된 기록을 고증하는 위대한 기록새야.",
        domain=OutputDomain.CHARACTER_DIALOGUE,
        situation=SituationId.INTRO_GIROKSAE,
        stage=ConversationStage.GREETING,
    ) == ()
    violations = guard.validate(
        "역사는 위대한 기록새인 내가 다 알아.",
        domain=OutputDomain.CHARACTER_DIALOGUE,
        situation=SituationId.HISTORY_FACT_QUESTION,
        stage=ConversationStage.HISTORICAL_QUESTION,
    )
    assert {item.code for item in violations} >= {"repeated_great_giroksae"}


def test_great_giroksae_runtime_is_limited_to_first_intro_turn(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    first = chat.ask("뭘 하면 돼?", conversation_mode="piece_chat", screen_type="intro")
    second = chat.ask(
        "다시 소개해줘", session_id=first.session_id,
        conversation_mode="piece_chat", screen_type="intro",
    )
    assert first.answer.count("위대한 기록새") == 1
    assert "위대한 기록새" not in second.answer


def test_no_fake_citation_or_cultural_generalization() -> None:
    guard = GiroksaeStyleGuard()
    context = dict(
        domain=OutputDomain.CHARACTER_DIALOGUE,
        situation=SituationId.CROSS_CULTURAL_COMPARISON,
        stage=ConversationStage.HISTORICAL_QUESTION,
    )
    violations = guard.validate("중국인들은 다 비슷하지. [출처: 가짜 자료]", citations=(), **context)
    assert {item.code for item in violations} >= {"cultural_generalization", "invented_citation"}


def test_low_interest_and_fatigue_have_no_follow_up(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    low = chat.ask("딱히 없어요.", conversation_mode="piece_chat")
    fatigue = chat.ask("조금 피곤해요.", conversation_mode="piece_chat")
    assert low.follow_up_question is None and not low.answer.endswith("?")
    assert fatigue.follow_up_question is None and not fatigue.answer.endswith("?")
    assert not low.rag_used and not fatigue.rag_used


def test_api_metadata_and_source_sufficiency(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    factual = chat.ask("붉은 등대 전시관은 언제 만들어졌어요?", conversation_mode="free_chat")
    missing = chat.ask("서울 궁궐의 왕은 누구야?", conversation_mode="free_chat")
    assert factual.persona_id == PERSONA_ID
    assert factual.language == "ko" and factual.culture == "korea"
    assert factual.conversation_stage == "historical_question"
    assert factual.source_sufficiency == SourceSufficiency.SUFFICIENT.value
    assert missing.source_sufficiency == SourceSufficiency.INSUFFICIENT.value
    assert missing.citations == ()
    assert missing.answer == "현재 검수된 자료만으로는 확인할 수 없습니다."
    assert missing.speech_level == "formal_docent"
    conflict = SimpleNamespace(chunk=SimpleNamespace(payload={"source_conflict": True}))
    assert ConversationalRagOrchestrator._source_sufficiency([conflict]) == SourceSufficiency.CONFLICTING


def test_chinese_contract_is_explicitly_pending_not_reviewed() -> None:
    language, culture, status = locale_policy("zh-CN", OutputDomain.CHARACTER_DIALOGUE)
    assert (language, culture) == ("zh-CN", "china")
    assert status == TranslationStatus.REVIEW_PENDING
    prompt = build_persona_prompt(
        domain=OutputDomain.CHARACTER_DIALOGUE, locale="zh-CN",
        mode=ConversationMode.FREE_CHAT, situation=SituationId.FREE_CHAT_GREETING,
    )
    assert "使用自然、亲切的口语和“你”" in prompt
    assert "不用“伟大的记录鸟”" in prompt


def test_unreviewed_chinese_runtime_is_marked_pending(tmp_path) -> None:
    response = orchestrator(tmp_path).ask("你好", conversation_mode="free_chat", locale="zh-CN")
    assert response.translation_status == "review_pending"
    assert response.language == "zh-CN" and response.culture == "china"
    assert "尚未完成审核" in response.answer
    assert "伟大的记录鸟" not in response.answer


def test_korean_and_chinese_locale_keep_the_same_retrieved_facts(tmp_path) -> None:
    chat = orchestrator(tmp_path)
    query = "붉은 등대 전시관은 언제 만들어졌어요?"
    korean = chat.ask(query, conversation_mode="free_chat", locale="ko")
    chinese_locale = chat.ask(query, conversation_mode="free_chat", locale="zh-CN")
    assert korean.retrieved_chunk_ids == chinese_locale.retrieved_chunk_ids
    assert korean.retrieved_source_ids == chinese_locale.retrieved_source_ids
    assert chinese_locale.translation_status == "review_pending"


def test_free_input_end_intent_returns_structured_action_without_long_reply(tmp_path) -> None:
    response = orchestrator(tmp_path).ask("채팅창 닫아줘", conversation_mode="free_chat")
    assert response.next_action_code == "RETURN_TO_GAME"
    assert response.game_state_mutation is False
    assert len(response.answer) < 30


def test_journey_caption_uses_only_user_words_and_does_not_claim_storage() -> None:
    rendered = GiroksaeResponseRenderer().journey_caption("오래된 간판이 기억에 남았다")
    assert rendered.text == "오래된 간판이 기억에 남았다"
    assert rendered.output_domain == OutputDomain.JOURNEY_FILM_CAPTION
    assert rendered.speech_level == SpeechLevel.NEUTRAL_CAPTION
    with pytest.raises(ValueError):
        GiroksaeResponseRenderer().journey_caption("   ")


def test_historical_docent_remains_formal_and_separate() -> None:
    rendered = GiroksaeResponseRenderer().render(
        "검수된 역사 설명입니다.",
        domain=OutputDomain.HISTORICAL_DOCENT,
        situation=SituationId.HISTORY_FACT_QUESTION,
        stage=ConversationStage.HISTORICAL_QUESTION,
    )
    assert rendered.speech_level == SpeechLevel.FORMAL_DOCENT
    assert rendered.output_domain == OutputDomain.HISTORICAL_DOCENT
