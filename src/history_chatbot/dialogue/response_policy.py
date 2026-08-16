"""상황 분류 결과를 검색 여부와 안전한 기록새 응답으로 라우팅한다."""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass

from history_chatbot.dialogue.personalization_tags import observations
from history_chatbot.dialogue.situation_classifier import SituationClassifier
from history_chatbot.dialogue.situation_models import ClassificationInput, ClassificationResult, RequiredContext, SituationId as S


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    classification: ClassificationResult
    answer: str
    follow_up_question: str | None
    should_retrieve: bool
    should_call_llm: bool
    warnings: tuple[str, ...] = ()
    response_template_id: str | None = None
    next_action_code: str | None = None
    required_context: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    capability_supported: bool = True
    fallback_used: bool = False
    policy_flags: tuple[str, ...] = ()
    personalization_tags: tuple[str, ...] = ()
    context_state: tuple[str, ...] = ()
    citations: tuple[dict[str, object], ...] = ()

    @property
    def situation_id(self) -> str:
        return self.classification.primary_situation_id.value

    @property
    def response_text(self) -> str:
        return self.answer


class GiroksaeDialogueEngine:
    def __init__(self, classifier: SituationClassifier | None = None) -> None:
        self.classifier = classifier or SituationClassifier()

    def decide(self, value: ClassificationInput) -> PolicyDecision:
        result = self.classifier.classify(value)
        if result.primary_situation_id in {S.TECHNICAL_HELP, S.NAVIGATION_HELP, S.SAFETY_ACCESSIBILITY}:
            return self._capability_decision(result, value)
        if result.requires_clarification:
            question = "어느 부분을 말하는지 조금만 더 구체적으로 알려줄래?"
            if result.primary_situation_id == S.CROSS_CULTURAL_COMPARISON:
                question = "어느 나라나 지역의 역사와 비교하고 싶은지 알려줄래?"
            elif result.classification_reason_code == "MISSING_SUBJECT":
                question = self._missing_subject_question(value.user_message)
            elif result.classification_reason_code == "UNINTELLIGIBLE_INPUT":
                question = "어떤 내용인지 조금만 더 알려줄래? 장소나 사건, 인물 이름을 함께 말해주면 찾아볼게."
            return PolicyDecision(result, question, question, False, False)
        if result.requires_rag:
            return PolicyDecision(result, "", None, True, True)
        answer, follow_up = self._non_rag_answer(result, value)
        return PolicyDecision(result, answer, follow_up, False, False)

    @staticmethod
    def can_claim_persisted(value: ClassificationInput) -> bool:
        """A persistence claim needs both an implemented capability and consent."""
        return value.storage_capability and value.user_consent

    @classmethod
    def _capability_decision(cls, result: ClassificationResult, value: ClassificationInput) -> PolicyDecision:
        missing = tuple(context.value for context in result.required_context if not cls._has_context(value, context))
        supported = not missing and result.next_action_code in value.supported_action_codes
        if supported:
            answer = "필요한 상태가 확인됐습니다. 연결된 기능에서 요청을 처리할 수 있습니다."
        elif result.primary_situation_id == S.TECHNICAL_HELP:
            answer = "현재 연결된 앱 진단 기능이 없어 구체적인 버튼이나 아이콘 위치는 안내할 수 없습니다. 화면 상태와 기기 설정을 확인하고, 문제가 계속되면 현장 안내에 도움을 요청해 주세요."
        elif result.primary_situation_id == S.NAVIGATION_HELP:
            answer = "현재 위치와 지도 정보를 확인할 수 없어 거리·시간·방향이나 다음 조각 위치를 안내할 수 없습니다. 안전한 곳에서 공식 지도나 현장 안내를 확인해 주세요."
        else:
            if "current_fatigue" in result.context_state:
                answer = "우선 무리하지 말고 안전한 곳에서 쉬어 주세요. 검증된 쉼터 정보가 없어 현장 표지나 직원에게 확인해 주세요."
            else:
                answer = "검증된 시설 정보가 없어 접근 가능 여부나 경사로·엘리베이터 유무를 단정할 수 없습니다. 공식 시설 안내나 현장 직원에게 확인해 주세요."
        warnings = ("capability_provider_unavailable",) if not supported else ()
        return PolicyDecision(
            result, answer, None, False, False, warnings,
            response_template_id=None, next_action_code=result.next_action_code,
            required_context=tuple(x.value for x in result.required_context), missing_context=missing,
            capability_supported=supported, fallback_used=not supported,
            policy_flags=result.policy_flags,
            personalization_tags=result.personalization_tag_candidates,
            context_state=result.context_state, citations=(),
        )

    @staticmethod
    def _has_context(value: ClassificationInput, context: RequiredContext) -> bool:
        if context == RequiredContext.STORAGE_CAPABILITY:
            return value.storage_capability
        if context == RequiredContext.USER_CONSENT:
            return value.user_consent
        return getattr(value, context.value) is not None

    @staticmethod
    def _non_rag_answer(result: ClassificationResult, value: ClassificationInput) -> tuple[str, str | None]:
        situation = result.primary_situation_id
        if situation == S.FREE_CHAT_GREETING:
            return "안녕. 난 오래된 장소 기록을 모으고 고증하는 철새, 기록새야. 목포에서 발견한 이야기를 함께 이어 보자.", None
        if situation == S.INTRO_GIROKSAE:
            return "난 오래된 장소의 서사를 모으고 고증하는 위대한 기록새야. 오늘 남긴 기록으로 목포의 이야기를 함께 이어 보자.", None
        if situation == S.STRONG_DISSATISFACTION:
            return "알겠어. 불필요한 말은 빼고 필요한 내용만 줄게.", None
        if situation == S.LOW_ENGAGEMENT:
            return "특별히 남는 게 없을 수도 있지. 이번에는 다음 단계로 가볍게 넘어가자.", None
        if situation == S.EMOTION_NEGATIVE_HISTORY:
            return "가볍게 넘기기 어려운 이야기지. 좋게 포장하지 않고 확인된 기록만 차분히 살펴볼게.", None
        if situation in {S.REFLECTION_POSITIVE_GENERAL, S.EMOTION_POSITIVE}:
            q = "어떤 장면이 가장 먼저 떠올랐어?"
            return "그 장면이 기억에 남았구나. " + q, q
        if situation == S.PERSONAL_AND_LIGHT_CHAT:
            if result.classification_reason_code == "OUT_OF_HISTORY_SCOPE":
                return "나는 역사 이야기를 중심으로 안내하고 있어. 궁금한 장소나 사건, 인물이 있으면 물어봐 줘.", None
            compact = re.sub(r"\s+", "", value.user_message).rstrip(".!?")
            if re.fullmatch(r"(?:고마워|감사해|감사합니다)", compact):
                return "천만에. 다른 역사 이야기도 궁금하면 이어서 물어봐 줘.", None
            if re.fullmatch(r"(?:알겠어|알겠습니다|그렇구나|응|좋아|오케이)", compact):
                return "좋아. 더 궁금한 내용이 있으면 이어서 물어봐 줘.", None
            if "지쳤" in value.user_message or "피곤" in value.user_message:
                return "많이 걸었나 보네. 설명은 한두 문장으로 줄이고 천천히 갈게.", None
            return "그 이야기를 들려줘서 고마워. 여정과 억지로 엮지 않고 편하게 들을게.", None
        if situation == S.RESPONSE_STYLE_REQUEST:
            return "말한 방식에 맞춰 바로 조정할게.", None
        if situation == S.COMPARISON_CONTEXT:
            q = "어떤 점이 이어지거나 달라 보였어?"
            return q, q
        return "그렇게 바라봤구나. 이번 조각은 여기까지 두고 다음 흐름으로 이어갈 수 있어.", None

    @staticmethod
    def _missing_subject_question(message: str) -> str:
        if re.search(r"왜", message):
            return "어떤 사건이나 내용의 이유가 궁금한가요?"
        if re.search(r"언제", message):
            return "어떤 사건이나 장소의 시점을 알고 싶은가요?"
        if re.search(r"누구|누가", message):
            return "어떤 인물이나 사건에 대해 궁금한가요?"
        if re.search(r"어디", message):
            return "어떤 사건이나 장소의 위치가 궁금한가요?"
        return "어떤 역사 주제를 말하는지 조금만 더 알려줄래?"

    def tag_candidates(self, result: ClassificationResult, *, turn_id: str, user_message: str) -> list[dict[str, object]]:
        return [asdict(item) | {"scope": item.scope.value} for item in observations(result.personalization_tag_candidates, turn_id=turn_id, user_message=user_message)]
