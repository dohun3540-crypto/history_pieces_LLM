"""상황 분류 결과를 검색 여부와 안전한 기록새 응답으로 라우팅한다."""

from __future__ import annotations

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
            question = "어느 부분을 말씀하시는지 조금만 더 구체적으로 알려주세요."
            if result.primary_situation_id == S.CROSS_CULTURAL_COMPARISON:
                question = "어느 나라나 지역의 역사와 비교하고 싶은지 알려주세요."
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
            q = "현재 장소나 방금 본 조각, 목포 역사에 관해 궁금한 점이 있나요?"
            return "안녕하세요. 목포의 장소와 사람의 기억을 연결해 드리는 기록새예요. " + q, q
        if situation == S.INTRO_GIROKSAE:
            return "안녕하세요. 저는 목포의 장소와 사람에 남은 기록을 연결하는 기록새예요. 바로 첫 조각부터 살펴볼까요?", "바로 첫 조각부터 살펴볼까요?"
        if situation == S.STRONG_DISSATISFACTION:
            return "알겠습니다. 핵심만 다시 답하거나, 바로 건너뛸 수 있어요.", "핵심 답변과 건너뛰기 중 무엇을 원하세요?"
        if situation == S.LOW_ENGAGEMENT:
            return "그럴 수 있어요. 이번 대화는 여기서 마치고 다음 단계로 넘어가도 됩니다.", None
        if situation == S.EMOTION_NEGATIVE_HISTORY:
            return "그렇게 느끼실 수 있어요. 좋게 포장하지 않고, 원하실 때 확인된 사실만 설명드릴게요.", "설명을 계속할까요, 여기서 마칠까요?"
        if situation in {S.REFLECTION_POSITIVE_GENERAL, S.EMOTION_POSITIVE}:
            q = "어떤 점이 가장 인상 깊었나요?"
            return "그 경험이 기억에 남으셨군요. " + q, q
        if situation == S.PERSONAL_AND_LIGHT_CHAT:
            if "지쳤" in value.user_message or "피곤" in value.user_message:
                return "많이 걸으셨나 봐요. 설명은 한두 문장으로 줄이고 천천히 진행할게요.", None
            return "그 이야기를 들려주셔서 고마워요. 편하게 이어서 말씀해 주세요.", None
        if situation == S.RESPONSE_STYLE_REQUEST:
            return "요청하신 방식으로 바로 조정할게요.", None
        if situation == S.COMPARISON_CONTEXT:
            q = "어떤 점에서 연결되거나 달라 보였나요?"
            return q, q
        return "말씀하신 감상을 존중해요. 원하시면 다음 단계로 이어갈 수 있습니다.", None

    def tag_candidates(self, result: ClassificationResult, *, turn_id: str, user_message: str) -> list[dict[str, object]]:
        return [asdict(item) | {"scope": item.scope.value} for item in observations(result.personalization_tag_candidates, turn_id=turn_id, user_message=user_message)]
