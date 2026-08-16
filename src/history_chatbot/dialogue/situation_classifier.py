"""화면·모드·최근 문맥을 함께 쓰는 결정론적 기록새 분류기."""

from __future__ import annotations

import re

from history_chatbot.dialogue.situation_models import (
    ActionCode,
    ClassificationInput,
    ClassificationResult,
    RequiredContext,
    ResponseLengthMode,
    SituationId as S,
)


class SituationClassifier:
    """LLM 없이도 안전 정책을 선행 적용하는 보수적 분류기.

    설명 가능한 내부 reason code만 내보내며 추론 과정은 저장하지 않는다.
    """

    def classify(self, value: ClassificationInput) -> ClassificationResult:
        text = value.user_message.strip()
        compact = re.sub(r"\s+", "", text.lower())
        secondary: list[S] = []
        tags: list[str] = []
        length = self._length(text, value.existing_style_preferences)

        if self._is_bare_interrogative(text) and not value.recent_turns:
            return self._result(S.HISTORY_FACT_QUESTION, (), .72, "missing_subject", False, True, length, (), "명확화 질문", "MISSING_SUBJECT")
        if self._is_unintelligible(text):
            return self._result(S.PERSONAL_AND_LIGHT_CHAT, (), .30, "unintelligible", False, True, length, (), "명확화 질문", "UNINTELLIGIBLE_INPUT")
        if self._is_out_of_scope(text):
            primary, reason, confidence = S.PERSONAL_AND_LIGHT_CHAT, "OUT_OF_HISTORY_SCOPE", 0.98
        elif self._contains(text, ("계단", "휠체어", "경사로", "엘리베이터", "쉼터", "안전", "너무 더워", "쉬고 싶")):
            primary, reason, confidence = S.SAFETY_ACCESSIBILITY, "SAFETY_CAPABILITY_OVERRIDE", 0.99
        elif self._contains(text, ("사진이 안 겹", "버튼이 안", "버튼 안", "소리가 안", "오디오가 안", "화면 오류")):
            primary, reason, confidence = S.TECHNICAL_HELP, "TECHNICAL_CAPABILITY_REQUEST", 0.98
        elif self._contains(text, ("다음 조각", "길을 잃", "얼마나 걸", "거리", "가는 길", "방향 알려")):
            primary, reason, confidence = S.NAVIGATION_HELP, "NAVIGATION_CAPABILITY_REQUEST", 0.98
        elif self._contains(text, ("지랄", "개노잼", "뭐래", "ㅈㄴ", "씨발", "꺼져")):
            primary, reason, confidence = S.STRONG_DISSATISFACTION, "DISSATISFACTION_OVERRIDE", 0.99
            tags.append("frustration")
            if "많" in text:
                tags.append("prefers_very_short")
                length = ResponseLengthMode.VERY_SHORT
        elif self._contains(text, ("출처", "자료 어디", "근거 보여")):
            primary, reason, confidence = S.RESPONSE_STYLE_REQUEST, "SOURCE_REQUEST", 0.98
            secondary.append(S.EVIDENCE_AND_CORRECTION)
            tags.append("asks_source")
            length = ResponseLengthMode.SOURCE_VIEW
        elif self._contains(text, ("틀린", "아닌 것", "다시 확인", "정확히 무슨", "직접 지은")):
            primary, reason, confidence = S.EVIDENCE_AND_CORRECTION, "EVIDENCE_RECHECK", 0.96
        elif self._contains(text, ("그게", "그거", "그런 거")) and not value.recent_turns:
            return self._result(S.EVIDENCE_AND_CORRECTION, (), .55, "ambiguous_reference", False, True, length, (), "명확화 질문", "AMBIGUOUS_NO_CONTEXT")
        elif self._contains(text, ("이전 조각", "아까 본 조각", "지금까지 본", "세 조각")):
            primary, reason, confidence = S.JOURNEY_CONTEXT_QUESTION, "JOURNEY_CONTEXT", 0.96
        elif value.recent_turns and re.fullmatch(r"\s*(?:더|조금\s*더|자세히)[?.!]*\s*", text):
            primary, reason, confidence = S.HISTORY_FACT_QUESTION, "CONTEXTUAL_DETAIL_REQUEST", 0.94
        elif self._contains(text, ("쉽게", "간단히", "간단하게", "짧게", "자세히", "요약", "정리해", "풀어", "한 줄", "한 문장", "핵심만", "무슨 말", "초등학생")):
            primary, reason, confidence = S.RESPONSE_STYLE_REQUEST, "EXPLICIT_STYLE_REQUEST", 0.97
            tags.extend(self._style_tags(length))
        elif self._contains(text, ("피곤", "지쳤", "아무 얘기", "할머니", "할아버지", "처음 목포")):
            primary, reason, confidence = S.PERSONAL_AND_LIGHT_CHAT, "PERSONAL_CONTEXT", 0.94
            if self._contains(text, ("피곤", "지쳤")):
                tags.extend(("current_fatigue", "prefers_short"))
                length = ResponseLengthMode.SHORT
        elif self._contains(text, ("중국", "일본", "다른 나라", "우리나라", "해외")) and self._contains(text, ("비슷", "다르", "비교")):
            primary, reason, confidence = S.CROSS_CULTURAL_COMPARISON, "CROSS_CULTURAL_COMPARE", 0.94
            tags.append("interest_cross_cultural")
        elif self._contains(text, ("재미없", "딱히", "잘 모르겠", "그냥 그랬")):
            primary, reason, confidence = S.LOW_ENGAGEMENT, "LOW_ENGAGEMENT_SIGNAL", 0.95
            tags.append("engagement_low")
        elif self._contains(text, ("슬퍼", "불편", "좋게 느껴지지", "안타까")):
            primary, reason, confidence = S.EMOTION_NEGATIVE_HISTORY, "NEGATIVE_HISTORY_EMOTION", 0.93
            tags.append("emotion_sadness")
        elif self._contains(text, ("인상 깊", "기억에 남", "괜찮았")):
            primary, reason, confidence = S.REFLECTION_POSITIVE_GENERAL, "POSITIVE_REFLECTION", 0.91
        elif self._contains(text, ("재미있", "신기", "뿌듯", "몰랐")) and not text.endswith("?"):
            primary, reason, confidence = S.EMOTION_POSITIVE, "POSITIVE_EMOTION", 0.90
        elif self._contains(text, ("건물", "건축", "구조", "외관", "보존")):
            primary, reason, confidence = S.INTEREST_ARCHITECTURE, "ARCHITECTURE_TOPIC", 0.93
            tags.append("interest_architecture")
        elif self._contains(text, ("사람", "인물", "누구")):
            primary, reason, confidence = S.INTEREST_PEOPLE, "PEOPLE_TOPIC", 0.92
            tags.append("interest_people")
        elif self._contains(text, ("생활", "교통", "상업", "주변", "거리", "도시", "기차", "역")) and self._is_fact_request(text):
            primary, reason, confidence = S.INTEREST_DAILY_CITY, "DAILY_CITY_TOPIC", 0.90
            tags.append("interest_daily_life")
        elif self._contains(text, ("달라", "남아 있는", "비교", "연결되는")):
            primary, reason, confidence = S.COMPARISON_CONTEXT, "COMPARISON_TOPIC", 0.86
        elif value.screen_type.value == "intro" or self._contains(compact, ("너는누구", "바로시작")):
            primary, reason, confidence = S.INTRO_GIROKSAE, "INTRO_SCREEN_CONTEXT", 0.92
        elif self._is_greeting(compact):
            primary, reason, confidence = S.FREE_CHAT_GREETING, "GREETING", 0.98
        elif self._is_acknowledgement(text):
            primary, reason, confidence = S.PERSONAL_AND_LIGHT_CHAT, "ACKNOWLEDGEMENT", 0.98
        elif self._looks_like_entity(text):
            primary, reason, confidence = S.HISTORY_FACT_QUESTION, "ENTITY_ONLY_HISTORY_QUERY", 0.78
        elif self._is_fact_request(text):
            primary, reason, confidence = S.HISTORY_FACT_QUESTION, "FACT_QUESTION", 0.82
        else:
            primary, reason, confidence = S.PERSONAL_AND_LIGHT_CHAT, "LOW_CONFIDENCE_LIGHT_CHAT", 0.58

        requires_clarification = primary == S.CROSS_CULTURAL_COMPARISON and "우리나라" in text
        requires_rag = self._requires_rag(primary, text, secondary)
        if reason == "OUT_OF_HISTORY_SCOPE":
            requires_rag = False
        if primary == S.RESPONSE_STYLE_REQUEST and reason == "EXPLICIT_STYLE_REQUEST":
            requires_clarification = not bool(value.recent_turns)
            requires_rag = bool(value.recent_turns)
        if requires_clarification:
            requires_rag = False
        contract = self._capability_contract(primary, text)
        return self._result(primary, tuple(secondary), confidence, self._intent(primary), requires_rag, requires_clarification, length, tuple(dict.fromkeys(tags)), self._next_action(primary, requires_clarification), reason, *contract)

    @staticmethod
    def _contains(text: str, values: tuple[str, ...]) -> bool:
        return any(value in text for value in values)

    @staticmethod
    def _is_greeting(compact: str) -> bool:
        return compact.rstrip(".!?") in {"안녕", "안녕하세요", "반가워", "안녕기록새"}

    @staticmethod
    def _is_acknowledgement(text: str) -> bool:
        compact = re.sub(r"\s+", "", text).rstrip(".!?")
        return bool(re.fullmatch(r"(?:고마워|감사해|감사합니다|알겠어|알겠습니다|그렇구나|응|좋아|오케이)", compact))

    @staticmethod
    def _is_bare_interrogative(text: str) -> bool:
        return bool(re.fullmatch(r"\s*(?:왜|언제|어디|누구|누가|무엇|뭐|어떻게)[?.!]*\s*", text))

    @staticmethod
    def _is_out_of_scope(text: str) -> bool:
        return bool(re.search(r"(?:오늘|내일|이번\s*주)\s*(?:날씨|기온)|주가|환율|레시피|요리|컴퓨터|코딩|프로그래밍", text))

    @staticmethod
    def _is_unintelligible(text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return bool(compact) and not re.search(r"[가-힣一-龥]", compact)

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*[가-힣·]{1,30}(?:\s+[가-힣·]{2,30}){0,3}[?.!]*\s*",
            text,
        ))

    @staticmethod
    def _is_fact_request(text: str) -> bool:
        explicit_history_request = bool(
            re.search(
                r"(?:역사|사건|인물|목포(?:역|항|진)|영사관|주식회사|학생운동|"
                r"독립운동|건물).*(?:(?:이야기|얘기)(?:해\s*줘|\s*좀\s*해줘)?|넘어가자|돌아가자|살펴보자)",
                text,
            )
        )
        return explicit_history_request or text.endswith(("?", "요?", "？")) or any(
            x in text
            for x in (
                "왜", "언제", "무엇", "어떤", "알려", "설명", "관계", "확인", "구분",
                "이야기해줘", "이야기해 줘", "얘기해줘", "얘기 좀 해줘",
                "什么", "何时", "哪年", "哪里",
                "多少", "是谁", "为什么",
            )
        )

    def _requires_rag(self, situation: S, text: str, secondary: list[S]) -> bool:
        if situation in {S.TECHNICAL_HELP, S.NAVIGATION_HELP, S.SAFETY_ACCESSIBILITY}:
            return False
        if situation in {S.INTEREST_PEOPLE, S.INTEREST_DAILY_CITY, S.HISTORY_FACT_QUESTION, S.EVIDENCE_AND_CORRECTION, S.CROSS_CULTURAL_COMPARISON}:
            return True
        if S.EVIDENCE_AND_CORRECTION in secondary:
            return True
        if situation == S.RESPONSE_STYLE_REQUEST and self._is_fact_request(text):
            return True
        if situation in {S.INTEREST_ARCHITECTURE, S.JOURNEY_CONTEXT_QUESTION, S.COMPARISON_CONTEXT, S.EMOTION_NEGATIVE_HISTORY, S.PERSONAL_AND_LIGHT_CHAT}:
            return self._is_fact_request(text) or "출처" in text
        return False

    @staticmethod
    def _length(text: str, existing: tuple[str, ...]) -> ResponseLengthMode:
        if "prefers_very_short" in existing or "한두 문장" in text:
            return ResponseLengthMode.VERY_SHORT
        if "짧게" in text or "간단히" in text or "prefers_short" in existing:
            return ResponseLengthMode.SHORT
        if "쉽게" in text:
            return ResponseLengthMode.SIMPLE
        if "자세히" in text:
            return ResponseLengthMode.DETAILED
        if "요약" in text or "정리" in text:
            return ResponseLengthMode.SUMMARY
        return ResponseLengthMode.DEFAULT

    @staticmethod
    def _style_tags(length: ResponseLengthMode) -> tuple[str, ...]:
        return {ResponseLengthMode.VERY_SHORT: ("prefers_very_short",), ResponseLengthMode.SHORT: ("prefers_short",), ResponseLengthMode.SIMPLE: ("prefers_simple",), ResponseLengthMode.DETAILED: ("prefers_detailed",), ResponseLengthMode.SUMMARY: ("prefers_summary",)}.get(length, ())

    @staticmethod
    def _intent(situation: S) -> str:
        return situation.value.lower()

    @staticmethod
    def _next_action(situation: S, clarification: bool) -> str:
        if clarification:
            return "clarify"
        if situation in {S.LOW_ENGAGEMENT, S.STRONG_DISSATISFACTION}:
            return "offer_skip_or_continue"
        return "retrieve_and_answer" if situation in {S.HISTORY_FACT_QUESTION, S.INTEREST_PEOPLE, S.INTEREST_DAILY_CITY, S.EVIDENCE_AND_CORRECTION, S.CROSS_CULTURAL_COMPARISON} else "respond"

    @staticmethod
    def _capability_contract(situation: S, text: str):
        if situation == S.TECHNICAL_HELP:
            action = ActionCode.OPEN_TECH_DIAGNOSTIC_OVERLAY
            if "버튼" in text:
                action = ActionCode.CHECK_MISSION_COMPLETION_STATE
            elif "소리" in text or "오디오" in text:
                action = ActionCode.OPEN_AUDIO_TROUBLESHOOTING
            return action.value, (RequiredContext.APP_STATE,), ("requires_app_state", "no_rag"), ("technical_issue",)
        if situation == S.NAVIGATION_HELP:
            if "다음 조각" in text:
                return ActionCode.OPEN_ROUTE_TO_NEXT_PIECE.value, (RequiredContext.JOURNEY_STATE, RequiredContext.MAP_DATA), ("requires_journey_state", "requires_map_data", "no_rag"), ("navigation_issue",)
            if "얼마나" in text or "시간" in text or "거리" in text:
                return ActionCode.CALCULATE_ROUTE_ETA.value, (RequiredContext.CURRENT_LOCATION, RequiredContext.MAP_DATA), ("requires_location", "requires_map_data", "no_rag"), ("navigation_issue",)
            return ActionCode.RECALCULATE_ROUTE_OR_SHOW_HELP.value, (RequiredContext.CURRENT_LOCATION,), ("requires_location", "safety_first", "no_rag"), ("navigation_issue",)
        if situation == S.SAFETY_ACCESSIBILITY:
            action = ActionCode.CHECK_ACCESSIBLE_ROUTE
            state = ("accessibility_request",)
            if "휠체어" in text:
                action = ActionCode.CHECK_WHEELCHAIR_ACCESS
            elif "더워" in text or "쉬" in text or "쉼터" in text:
                action = ActionCode.SHOW_VERIFIED_REST_AREAS_OR_HELP
                state = ("current_fatigue", "heat_discomfort")
            return action.value, (RequiredContext.VERIFIED_FACILITY_DATA,), ("requires_verified_facility_data", "safety_first", "no_rag"), state
        return None, (), (), ()

    @staticmethod
    def _result(primary, secondary, confidence, intent, rag, clarify, length, tags, action, reason, action_code=None, required_context=(), policy_flags=(), context_state=()):
        return ClassificationResult(primary, secondary, confidence, intent, rag, clarify, length, tags, action, reason, action_code, required_context, policy_flags, context_state)
