"""Mode-specific policy layered over the shared Giroksae classifier."""

from __future__ import annotations

from dataclasses import dataclass

from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.response_policy import PolicyDecision
from history_chatbot.dialogue.situation_models import ActionCode, SituationId as S
from history_chatbot.dialogue.track_models import (
    FreeChatUiContract, FreeChatUiState, ModeTransition, PieceChatUiContract,
    PieceChatUiState, RequestState, SharedSessionContext,
)


@dataclass(frozen=True, slots=True)
class TrackDecision:
    action_code: str | None
    response_override: str | None
    follow_up_question: str | None
    should_retrieve: bool
    should_call_llm: bool
    transition: ModeTransition | None
    request_state: RequestState
    ui_state: str
    piece_ui: PieceChatUiContract | None = None
    free_ui: FreeChatUiContract | None = None
    capability_supported: bool = False
    fallback_used: bool = False


class ChatTrackPolicy:
    _FREE_CHAT_DETAIL_SIGNALS = ("출처", "자세히", "배경", "비교", "관계", "근거")
    _TRANSFER_SITUATIONS = {
        S.RESPONSE_STYLE_REQUEST, S.EVIDENCE_AND_CORRECTION,
        S.CROSS_CULTURAL_COMPARISON, S.JOURNEY_CONTEXT_QUESTION,
        S.INTEREST_PEOPLE,
    }

    def route(
        self, *, mode: ConversationMode, query: str, decision: PolicyDecision,
        context: SharedSessionContext, piece_follow_up_count: int = 0,
        return_target: str = "game",
    ) -> TrackDecision:
        if piece_follow_up_count < 0:
            raise ValueError("piece_follow_up_count는 0 이상이어야 합니다.")
        compact = query.replace(" ", "")
        if any(signal in compact for signal in ("그만할래", "대화끝낼게", "채팅창닫아줘")):
            action = (
                ActionCode.SKIP_REFLECTION.value
                if mode == ConversationMode.PIECE_CHAT
                else ActionCode.RETURN_TO_GAME.value
            )
            ui_state = (
                PieceChatUiState.SKIPPED.value
                if mode == ConversationMode.PIECE_CHAT
                else FreeChatUiState.RETURNING_TO_GAME.value
            )
            return TrackDecision(
                action, "알겠어. 여기서 마칠게.", None, False, False, None,
                RequestState.SUCCESS, ui_state,
                piece_ui=(PieceChatUiContract(PieceChatUiState.SKIPPED) if mode == ConversationMode.PIECE_CHAT else None),
                free_ui=(FreeChatUiContract(FreeChatUiState.RETURNING_TO_GAME) if mode == ConversationMode.FREE_CHAT else None),
                capability_supported=action in context.available_capabilities,
                fallback_used=action not in context.available_capabilities,
            )
        if mode == ConversationMode.PIECE_CHAT and "다음으로넘어가자" in compact:
            action = ActionCode.GO_NEXT_PIECE.value
            return TrackDecision(
                action, "좋아. 다음 조각으로 이어가자.", None, False, False, None,
                RequestState.SUCCESS, PieceChatUiState.READY_FOR_NEXT_PIECE.value,
                PieceChatUiContract(PieceChatUiState.READY_FOR_NEXT_PIECE, next_piece_available=True),
                capability_supported=action in context.available_capabilities,
                fallback_used=action not in context.available_capabilities,
            )
        if mode == ConversationMode.PIECE_CHAT:
            return self._piece(query, decision, context, piece_follow_up_count, return_target)
        return self._free(decision)

    def _piece(self, query, decision, context, follow_up_count, return_target):
        situation = decision.classification.primary_situation_id
        if situation in {S.TECHNICAL_HELP, S.NAVIGATION_HELP, S.SAFETY_ACCESSIBILITY}:
            return TrackDecision(
                decision.next_action_code, decision.answer, None, False, False, None,
                RequestState.CAPABILITY_UNAVAILABLE if not decision.capability_supported else RequestState.SUCCESS,
                PieceChatUiState.RESPONDING.value,
                PieceChatUiContract(PieceChatUiState.RESPONDING),
                capability_supported=decision.capability_supported,
                fallback_used=decision.fallback_used,
            )
        if "저장" in query or "기록해" in query:
            permitted = (
                context.storage_capability and context.user_consent
                and ActionCode.SAVE_SHORT_REFLECTION.value in context.available_capabilities
            )
            answer = (
                "짧은 감상으로 저장했습니다."
                if permitted
                else "저장 기능과 동의가 확인되지 않아 이번 대화에서만 참고하겠습니다."
            )
            return TrackDecision(
                ActionCode.SAVE_SHORT_REFLECTION.value, answer, None, False, False,
                None, RequestState.SUCCESS if permitted else RequestState.CAPABILITY_UNAVAILABLE,
                PieceChatUiState.RESPONDING.value,
                PieceChatUiContract(PieceChatUiState.RESPONDING, storage_capability=context.storage_capability),
                capability_supported=permitted, fallback_used=not permitted,
            )
        if "짧게" in query or "간단히" in query:
            supported = ActionCode.CONTINUE_WITH_SHORT_MODE.value in context.available_capabilities
            return TrackDecision(
                ActionCode.CONTINUE_WITH_SHORT_MODE.value,
                "앞으로는 한두 문장으로 짧게 말할게.", None,
                False, False, None, RequestState.SUCCESS,
                PieceChatUiState.RESPONDING.value,
                PieceChatUiContract(PieceChatUiState.RESPONDING),
                capability_supported=supported, fallback_used=not supported,
            )
        if situation == S.LOW_ENGAGEMENT:
            return TrackDecision(
                ActionCode.SKIP_REFLECTION.value,
                "특별히 남는 게 없을 수도 있지. 감상은 건너뛰고 넘어가자.",
                None, False, False, None, RequestState.SUCCESS,
                PieceChatUiState.SKIPPED.value,
                PieceChatUiContract(PieceChatUiState.SKIPPED, reflection_input_enabled=False),
            )
        if "current_fatigue" in decision.classification.personalization_tag_candidates or "current_fatigue" in decision.context_state:
            return TrackDecision(
                ActionCode.PAUSE_JOURNEY.value,
                "잠시 쉬어도 괜찮아. 멈추거나 짧은 설명으로 이어갈 수 있어.",
                None, False, False, None, RequestState.SUCCESS,
                PieceChatUiState.PAUSED.value,
                PieceChatUiContract(PieceChatUiState.PAUSED, reflection_input_enabled=False),
            )
        detailed = decision.should_retrieve and (
            situation in self._TRANSFER_SITUATIONS
            or any(signal in query for signal in self._FREE_CHAT_DETAIL_SIGNALS)
        )
        if detailed:
            transition = ModeTransition.open_free_chat(
                question=query, context=context, return_target=return_target,
            )
            return TrackDecision(
                ActionCode.OPEN_FREE_CHAT.value,
                "이 질문은 근거와 배경을 함께 보는 편이 좋아. 자유대화에서 이어서 확인해보자.",
                None, False, False, transition, RequestState.CAPABILITY_UNAVAILABLE,
                PieceChatUiState.RESPONDING.value,
                PieceChatUiContract(PieceChatUiState.RESPONDING, free_chat_available=True),
                capability_supported=ActionCode.OPEN_FREE_CHAT.value in context.available_capabilities,
                fallback_used=ActionCode.OPEN_FREE_CHAT.value not in context.available_capabilities,
            )
        follow_up = decision.follow_up_question if follow_up_count == 0 else None
        answer = decision.answer
        if follow_up_count > 0 and answer:
            answer = answer.split(". ", 1)[0].strip()
            if answer and not answer.endswith("."):
                answer += "."
        return TrackDecision(
            decision.next_action_code, answer or None, follow_up,
            decision.should_retrieve, decision.should_call_llm, None,
            RequestState.LOADING if decision.should_retrieve else RequestState.SUCCESS,
            PieceChatUiState.RESPONDING.value,
            PieceChatUiContract(PieceChatUiState.RESPONDING),
            capability_supported=decision.capability_supported,
            fallback_used=decision.fallback_used,
        )

    @staticmethod
    def _free(decision: PolicyDecision) -> TrackDecision:
        situation = decision.classification.primary_situation_id
        if situation in {S.TECHNICAL_HELP, S.NAVIGATION_HELP, S.SAFETY_ACCESSIBILITY}:
            state = RequestState.CAPABILITY_UNAVAILABLE if not decision.capability_supported else RequestState.SUCCESS
            return TrackDecision(
                decision.next_action_code, decision.answer, decision.follow_up_question,
                False, False, None, state, FreeChatUiState.ACTIVE.value,
                free_ui=FreeChatUiContract(FreeChatUiState.ACTIVE),
                capability_supported=decision.capability_supported,
                fallback_used=decision.fallback_used,
            )
        action = ActionCode.ANSWER_WITH_CITATIONS.value if decision.should_retrieve else None
        if situation == S.JOURNEY_CONTEXT_QUESTION:
            action = ActionCode.SUMMARIZE_COMPLETED_PIECES.value
        elif situation == S.RESPONSE_STYLE_REQUEST and decision.should_retrieve:
            action = ActionCode.OPEN_CITATION_PANEL.value
        return TrackDecision(
            action, decision.answer or None, decision.follow_up_question,
            decision.should_retrieve, decision.should_call_llm, None,
            RequestState.LOADING if decision.should_retrieve else RequestState.SUCCESS,
            FreeChatUiState.LOADING.value if decision.should_retrieve else FreeChatUiState.ACTIVE.value,
            free_ui=FreeChatUiContract(
                FreeChatUiState.LOADING if decision.should_retrieve else FreeChatUiState.ACTIVE,
                suggested_questions=("이 장소는 언제 만들어졌나요?", "관련 인물은 누구인가요?"),
            ),
            capability_supported=True,
        )
