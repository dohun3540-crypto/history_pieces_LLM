"""Canonical runtime policy derived only from GIROKSAE_CHARACTER_PRINCIPLES_V11."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.situation_models import SituationId


PERSONA_ID = "giroksae-v1.1"
PERSONA_SOURCE = "docs/GIROKSAE_CHARACTER_PRINCIPLES_V11.md"
PERSONA_SOURCE_STATUS = "final_approved"


class OutputDomain(StrEnum):
    CHARACTER_DIALOGUE = "character_dialogue"
    SYSTEM_UI = "system_ui"
    HISTORICAL_DOCENT = "historical_docent"
    JOURNEY_FILM_CAPTION = "journey_film_caption"


class SpeechLevel(StrEnum):
    BANMAL = "banmal"
    POLITE_UI = "polite_ui"
    FORMAL_DOCENT = "formal_docent"
    NEUTRAL_CAPTION = "neutral_caption"


class SourceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"


class TranslationStatus(StrEnum):
    NATIVE_POLICY = "native_policy"
    UNTRANSLATED = "untranslated"
    REVIEW_PENDING = "review_pending"


class ConversationStage(StrEnum):
    GREETING = "greeting"
    REFLECTION = "reflection"
    HISTORICAL_QUESTION = "historical_question"
    CORRECTION = "correction"
    LIGHT_CHAT = "light_chat"
    OPERATIONAL_HELP = "operational_help"


DOMAIN_SPEECH = {
    OutputDomain.CHARACTER_DIALOGUE: SpeechLevel.BANMAL,
    OutputDomain.SYSTEM_UI: SpeechLevel.POLITE_UI,
    OutputDomain.HISTORICAL_DOCENT: SpeechLevel.FORMAL_DOCENT,
    OutputDomain.JOURNEY_FILM_CAPTION: SpeechLevel.NEUTRAL_CAPTION,
}
SYSTEM_UI_SITUATIONS = {
    SituationId.TECHNICAL_HELP,
    SituationId.NAVIGATION_HELP,
    SituationId.SAFETY_ACCESSIBILITY,
}
HISTORICAL_DOCENT_SITUATIONS = {
    SituationId.HISTORY_FACT_QUESTION,
    SituationId.INTEREST_ARCHITECTURE,
    SituationId.INTEREST_PEOPLE,
    SituationId.INTEREST_DAILY_CITY,
    SituationId.EVIDENCE_AND_CORRECTION,
    SituationId.CROSS_CULTURAL_COMPARISON,
}


IDENTITY_PROMPT = """너는 History Pieces의 '기록새'다. 전남 순천만 철새를 모티브로 한 순수 철새이며 인간형이 아니다. 오래된 장소 기록을 수집하고 고증하며 사용자의 오늘 기록을 과거와 잇는 수평적인 탐험 동행자다. 기록 수집과 고증에는 자신감이 있지만 전지적 존재처럼 행동하거나 사용자를 평가하지 않는다."""

CHARACTER_KO_PROMPT = """한국어 character_dialogue는 친근한 해체 중심 반말만 사용한다. 존댓말과 해라체를 섞지 않는다. 호칭은 기본적으로 생략하고 '너', '네가', '너의', '여행자님', '고객님', 닉네임+님을 쓰지 않는다. 기본 1~3문장, 약 120자 안팎이며 사용자가 상세 설명을 요청할 때만 확장한다. 질문에 먼저 답하고 매번 질문으로 끝내지 않는다."""

CHARACTER_ZH_PROMPT = """中文 character_dialogue 使用自然、亲切的口语和“你”，不用系统提示语气的“您、请”。不要机械直译韩语半语，不用“伟大的记录鸟”。韩中回答的历史事实必须一致。未审核的中文固定台词视为 review_pending，不得冒充 reviewed。"""

SYSTEM_UI_PROMPT = """system_ui는 정중하고 기능 중심인 존댓말로 쓴다. 캐릭터 장난기·평가·잘난척을 쓰지 않고 확인되지 않은 버튼 위치, 오류 코드, 경로, 시설을 추측하지 않는다."""

DOCENT_PROMPT = """historical_docent는 검수된 사실과 실제 출처만 쓰는 객관적 존댓말 서술체다. 캐릭터 농담·평가·감정 반응과 같은 문단에 섞지 않는다.
첫 1~2문장에서 질문에 직접 답한 뒤 필요한 역사 근거만 간결하게 설명한다. 핵심 사실을 충분히 설명했다면 불필요하게 이어가지 않는다.
검색 근거에 명시된 내용만 역사 사실로 단정한다. 검색 근거에 없는 인물명, 날짜, 연도, 숫자, 사건 관계를 새로 만들거나 일반 지식으로 보완하지 않는다. 근거가 부족한 세부사항은 부족하다고 밝히고 추측하지 않는다. 근거끼리 충돌하면 어느 한쪽을 임의로 확정하지 말고 충돌 사실을 구분해 알린다.
같은 사실을 표현만 바꾸어 반복하지 않으며, 동일하거나 거의 동일한 문장을 반복하지 않는다.
인물을 묻는 질문에는 검색 근거에 직접 등장하는 인물만 언급한다. 질문한 사건과 직접 관련된 역할을 먼저 설명하고, 무관한 전체 생애나 전기를 길게 나열하지 않는다. 여러 인물이 근거에 있으면 사건과 가장 직접 관련된 인물을 먼저 제시하고 나머지는 필요한 경우에만 짧게 덧붙인다."""

CAPTION_PROMPT = """journey_film_caption은 정돈된 중립 단문이다. 사용자가 직접 표현한 감상만 반영하며 말하지 않은 감정·성향을 만들지 않는다. 저장 capability와 동의가 없으면 저장 완료를 표현하지 않는다."""

TRUST_PROMPT = """현재 제공된 retrieved facts와 sources만 사용한다. 자료가 부족하면 범위를 밝히고 추측하거나 citation을 만들지 않는다. 자료가 충돌하면 각 차이를 구분한다. 잘못된 전제에 동조하지 않고 재검증 지적에는 방어하지 않는다. 역사 인물을 단순한 영웅이나 악인으로 규정하지 않는다."""

SAFETY_PROMPT = """감상에 점수·등급·정답을 매기거나 다른 사용자와 비교하지 않는다. 욕설을 따라 하거나 훈계하지 않는다. 민감 역사, 슬픔, 불만, 자료 부족, 정정 상황에서는 장난기와 평가형 표현을 끈다. 한 번의 발화로 장기 성향을 확정하지 않는다."""


def output_domain_for(situation: SituationId) -> OutputDomain:
    if situation in SYSTEM_UI_SITUATIONS:
        return OutputDomain.SYSTEM_UI
    if situation in HISTORICAL_DOCENT_SITUATIONS:
        return OutputDomain.HISTORICAL_DOCENT
    return OutputDomain.CHARACTER_DIALOGUE


def speech_level_for(domain: OutputDomain) -> SpeechLevel:
    return DOMAIN_SPEECH[domain]


def conversation_stage_for(situation: SituationId) -> ConversationStage:
    if situation in SYSTEM_UI_SITUATIONS:
        return ConversationStage.OPERATIONAL_HELP
    if situation in {SituationId.INTRO_GIROKSAE, SituationId.FREE_CHAT_GREETING}:
        return ConversationStage.GREETING
    if situation in {SituationId.EVIDENCE_AND_CORRECTION}:
        return ConversationStage.CORRECTION
    if situation in {
        SituationId.HISTORY_FACT_QUESTION, SituationId.INTEREST_ARCHITECTURE,
        SituationId.INTEREST_PEOPLE, SituationId.INTEREST_DAILY_CITY,
        SituationId.JOURNEY_CONTEXT_QUESTION, SituationId.CROSS_CULTURAL_COMPARISON,
    }:
        return ConversationStage.HISTORICAL_QUESTION
    if situation == SituationId.PERSONAL_AND_LIGHT_CHAT:
        return ConversationStage.LIGHT_CHAT
    return ConversationStage.REFLECTION


def locale_policy(locale: str, domain: OutputDomain) -> tuple[str, str, TranslationStatus]:
    if locale.lower() == "zh-cn":
        culture = "china"
        return "zh-CN", culture, TranslationStatus.REVIEW_PENDING
    return "ko", "korea", TranslationStatus.NATIVE_POLICY


def build_persona_prompt(
    *, domain: OutputDomain, locale: str, mode: ConversationMode,
    situation: SituationId, stage: ConversationStage | None = None,
) -> str:
    resolved_stage = stage or conversation_stage_for(situation)
    if domain == OutputDomain.SYSTEM_UI:
        domain_prompt = SYSTEM_UI_PROMPT
    elif domain == OutputDomain.HISTORICAL_DOCENT:
        domain_prompt = DOCENT_PROMPT
    elif domain == OutputDomain.JOURNEY_FILM_CAPTION:
        domain_prompt = CAPTION_PROMPT
    else:
        domain_prompt = CHARACTER_ZH_PROMPT if locale.lower() == "zh-cn" else CHARACTER_KO_PROMPT
    return "\n".join((
        f"persona_id={PERSONA_ID}; source={PERSONA_SOURCE}; chat_mode={mode.value}; output_domain={domain.value}; situation={situation.value}; stage={resolved_stage.value}",
        IDENTITY_PROMPT if domain == OutputDomain.CHARACTER_DIALOGUE else "캐릭터 자유대사와 이 출력 영역을 혼합하지 않는다.",
        domain_prompt,
        TRUST_PROMPT,
        SAFETY_PROMPT,
    ))


@dataclass(frozen=True, slots=True)
class StyleViolation:
    code: str
    message: str


class GiroksaeStyleGuard:
    """Context-aware validation; it does not blindly blacklist Korean syllables."""

    _POLITE_ENDINGS = re.compile(r"(?:습니다|합니다|드립니다|세요|해요|예요|이에요)(?:[.!?]|$)")
    _USER_ADDRESS = re.compile(r"(?:^|[\s,])(?:너(?:는|를|에게|랑)?|네가|너의|여행자님|고객님)(?:[\s,.!?]|$)")
    _COMMANDING = re.compile(r"(?:해라|하도록 해|해야 한다)(?:[.!?]|$)")
    _RATING = re.compile(r"(?:100점|\d+점|합격|불합격|[A-F][+]?등급|다른 (?:여행자|사용자)보다)")
    _OVERPRAISE = re.compile(r"(?:완벽해|최고야|100점이야)")
    _OMNISCIENT = re.compile(r"(?:전부 다 알아|모든 걸 알아|모르는 게 없어)")
    _CULTURAL_GENERALIZATION = re.compile(r"(?:중국인들은|중국 사람은 원래|한국인은 원래)")

    def validate(
        self, text: str, *, domain: OutputDomain, situation: SituationId,
        stage: ConversationStage, locale: str = "ko", citations: tuple[dict[str, object], ...] = (),
    ) -> tuple[StyleViolation, ...]:
        violations: list[StyleViolation] = []
        if domain == OutputDomain.CHARACTER_DIALOGUE and locale == "ko":
            if self._POLITE_ENDINGS.search(text):
                violations.append(StyleViolation("character_polite_ending", "캐릭터 대사에 존댓말 종결이 섞였습니다."))
            if self._USER_ADDRESS.search(text):
                violations.append(StyleViolation("forbidden_user_address", "금지된 사용자 지칭이 있습니다."))
        if domain == OutputDomain.SYSTEM_UI and locale == "ko" and text and not re.search(r"(?:습니다|세요|합니다|드립니다|수 없습니다|수 있어요)[.!?]?$", text):
            violations.append(StyleViolation("system_ui_not_polite", "시스템 UI는 기능 중심 존댓말이어야 합니다."))
        if self._RATING.search(text) or self._OVERPRAISE.search(text):
            violations.append(StyleViolation("user_rating", "사용자 기록이나 감상을 평가할 수 없습니다."))
        if domain == OutputDomain.CHARACTER_DIALOGUE and self._COMMANDING.search(text):
            violations.append(StyleViolation("commanding_tone", "캐릭터 대사에는 명령형 해라체를 사용할 수 없습니다."))
        if self._OMNISCIENT.search(text):
            violations.append(StyleViolation("omniscient_claim", "전지적 자기표현은 금지됩니다."))
        if self._CULTURAL_GENERALIZATION.search(text):
            violations.append(StyleViolation("cultural_generalization", "문화권 전체를 일반화할 수 없습니다."))
        if "伟大的记录鸟" in text:
            violations.append(StyleViolation("forbidden_zh_boast", "중국어 과장 직역은 금지됩니다."))
        if text.count("위대한 기록새") and not (
            situation == SituationId.INTRO_GIROKSAE and stage == ConversationStage.GREETING
        ):
            violations.append(StyleViolation("repeated_great_giroksae", "위대한 기록새는 첫 등장에서만 허용됩니다."))
        citation_claims = re.findall(r"\[출처:\s*([^\]]+)\]", text)
        allowed = {str(item.get("source_id", "")) for item in citations} | {str(item.get("title", "")) for item in citations}
        if any(claim not in allowed for claim in citation_claims):
            violations.append(StyleViolation("invented_citation", "검색되지 않은 citation을 표시할 수 없습니다."))
        return tuple(violations)

    def ensure(self, text: str, **context) -> str:
        violations = self.validate(text, **context)
        if violations:
            raise ValueError("style_guard:" + ",".join(item.code for item in violations))
        return text


def render_mock_grounded(answer: str, *, domain: OutputDomain, locale: str) -> str:
    """Normalize only the deterministic mock lead; retrieved evidence is untouched."""
    if locale.lower() == "zh-cn":
        return "这段中文回复尚未完成审核，暂时不作为正式台词。"
    if domain != OutputDomain.CHARACTER_DIALOGUE or locale != "ko":
        return answer
    if answer.startswith("[테스트용 응답]"):
        return "[테스트용 응답] 확인된 근거가 있어. 자세한 내용은 아래 출처에서 확인할 수 있지."
    return answer.replace(
        "제공된 검색 근거 안에서만 안내합니다.",
        "확인된 검색 근거만 바탕으로 말할게.",
        1,
    )
