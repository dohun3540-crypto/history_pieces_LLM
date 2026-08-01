"""세션부터 검색·프롬프트·답변·출처까지 연결하는 grounded RAG."""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterator

from history_chatbot.chat.citation_builder import build_citations
from history_chatbot.chat.interfaces import Citation
from history_chatbot.chat.prompt_builder import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTIONS,
    build_prompt,
)
from history_chatbot.chat.session import ChatSession, SessionStore
from history_chatbot.dialogue.response_policy import GiroksaeDialogueEngine
from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.situation_models import ClassificationInput, ScreenType
from history_chatbot.dialogue.track_models import SharedSessionContext
from history_chatbot.dialogue.track_policy import ChatTrackPolicy
from history_chatbot.models.context_budget import ContextBudgetManager
from history_chatbot.models.contract import ChatCompletionBackend, LLMMessage, LLMRequest
from history_chatbot.models.remote import RemoteLLMError
from history_chatbot.retrieval.base import RankedChunk
from history_chatbot.retrieval.service import HybridRetrievalService
from history_chatbot.runtime import RuntimeMode


@dataclass(frozen=True, slots=True)
class ChatResponse:
    answer: str
    status: str
    sources: tuple[Citation, ...]
    used_chunks: int
    session_id: str
    locale: str
    prompt_version: str
    error: dict[str, object] | None = None
    context_metadata: dict[str, object] | None = None
    provisional_sources_used: int = 0
    rights_notice: str = ""
    usage_scope: str = ""
    source_ids: tuple[str, ...] = ()
    conversation_mode: str = "free_chat"
    screen_type: str = "free_chat"
    primary_situation_id: str = "HISTORY_FACT_QUESTION"
    secondary_situation_ids: tuple[str, ...] = ()
    next_action: str = "respond"
    follow_up_question: str | None = None
    personalization_tag_candidates: tuple[dict[str, object], ...] = ()
    personalization_tags: tuple[str, ...] = ()
    citations: tuple[dict[str, object], ...] = ()
    evidence: tuple[str, ...] = ()
    grounded: bool = False
    confidence: float = 0.0
    refusal_reason: str | None = None
    response_length_mode: str = "default"
    retrieved_chunk_ids: tuple[str, ...] = ()
    retrieved_source_ids: tuple[str, ...] = ()
    model_backend: str = ""
    embedding_backend: str = ""
    latency_ms: int = 0
    warnings: tuple[str, ...] = ()
    chat_mode: str = "free_chat"
    response_text: str = ""
    response_template_id: str | None = None
    example_id: str | None = None
    next_action_code: str | None = None
    required_context: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    capability_supported: bool = False
    fallback_used: bool = False
    policy_flags: tuple[str, ...] = ()
    context_state: tuple[str, ...] = ()
    current_place_id: str | None = None
    current_piece_id: str | None = None
    completed_piece_ids: tuple[str, ...] = ()
    game_state_mutation: bool = False
    mode_transition: dict[str, object] | None = None
    rag_used: bool = False
    storage_requested: bool = False
    storage_permitted: bool = False
    request_state: str = "success"
    ui_state: str = "active"
    suggested_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.response_text:
            object.__setattr__(self, "response_text", self.answer)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["sources"] = [asdict(source) for source in self.sources]
        value["response_text"] = self.answer
        return value


@dataclass(frozen=True, slots=True)
class StreamEvent:
    event: str
    data: dict[str, object]


class ConversationalRagOrchestrator:
    def __init__(
        self,
        retrieval: HybridRetrievalService,
        llm: ChatCompletionBackend,
        sessions: SessionStore,
        *,
        mode: RuntimeMode,
        max_chunks_per_document: int = 2,
        context_window: int = 8192,
        max_new_tokens: int = 512,
        dialogue: GiroksaeDialogueEngine | None = None,
    ) -> None:
        if mode == RuntimeMode.PRODUCTION and llm.backend_name == "mock":
            raise ValueError("production 모드에서는 MockLLM 기반 오케스트레이터를 사용할 수 없습니다.")
        self.retrieval = retrieval
        self.llm = llm
        self.sessions = sessions
        self.mode = mode
        self.max_chunks_per_document = max_chunks_per_document
        backend_config = getattr(llm, "config", None)
        self.max_new_tokens = getattr(backend_config, "max_new_tokens", max_new_tokens)
        self.budget = ContextBudgetManager(
            getattr(backend_config, "context_window", context_window)
        )
        self.dialogue = dialogue or GiroksaeDialogueEngine()
        self.track_policy = ChatTrackPolicy()

    def ask(
        self,
        user_query: str,
        *,
        session_id: str | None = None,
        locale: str = "ko",
        top_k: int = 3,
        conversation_mode: str = "free_chat",
        screen_type: str | None = None,
        current_piece_id: str | None = None,
        current_place_id: str | None = None,
        visited_piece_ids: tuple[str, ...] = (),
        existing_style_preferences: tuple[str, ...] = (),
        current_journey_step: str | None = None,
        piece_follow_up_count: int | None = None,
        return_target: str = "game",
        available_capabilities: tuple[str, ...] = (),
        storage_capability: bool = False,
        user_consent: bool = False,
    ) -> ChatResponse:
        started = time.perf_counter()
        query = self._validate(user_query, locale, top_k)
        session = self.sessions.get_or_create(session_id, locale)
        chat_mode = ConversationMode(conversation_mode)
        resolved_screen = screen_type or chat_mode.value
        if ScreenType(resolved_screen).value != chat_mode.value:
            raise ValueError("chat_mode와 screen_type이 일치해야 합니다.")
        shared_context = SharedSessionContext(
            session_id=session.session_id, locale=locale,
            current_place_id=current_place_id, current_piece_id=current_piece_id,
            completed_piece_ids=visited_piece_ids, current_journey_step=current_journey_step,
            temporary_response_length_preference=(existing_style_preferences[0] if existing_style_preferences else None),
            available_capabilities=available_capabilities,
            storage_capability=storage_capability, user_consent=user_consent,
        )
        classification_input = ClassificationInput(
            query,
            conversation_mode=conversation_mode,
            screen_type=ScreenType(resolved_screen),
            locale=locale,
            current_piece_id=current_piece_id,
            current_place_id=current_place_id,
            recent_turns=tuple(turn.user for turn in session.turns[-3:]),
            visited_piece_ids=visited_piece_ids,
            existing_style_preferences=existing_style_preferences,
            storage_capability=storage_capability,
            user_consent=user_consent,
            supported_action_codes=available_capabilities,
        )
        decision = self.dialogue.decide(classification_input)
        track = self.track_policy.route(
            mode=chat_mode, query=query, decision=decision, context=shared_context,
            piece_follow_up_count=(
                min(len(session.turns), 1)
                if piece_follow_up_count is None else piece_follow_up_count
            ),
            return_target=return_target,
        )
        classification = decision.classification
        common = {
            "conversation_mode": conversation_mode,
            "screen_type": resolved_screen,
            "primary_situation_id": classification.primary_situation_id.value,
            "secondary_situation_ids": tuple(x.value for x in classification.secondary_situation_ids),
            "next_action": track.action_code or classification.next_action,
            "follow_up_question": track.follow_up_question,
            "personalization_tag_candidates": tuple(
                self.dialogue.tag_candidates(classification, turn_id=uuid.uuid4().hex, user_message=query)
            ),
            "personalization_tags": classification.personalization_tag_candidates,
            "confidence": classification.confidence,
            "response_length_mode": classification.response_length_mode.value,
            "model_backend": self.llm.backend_name,
            "embedding_backend": self.retrieval.encoder.model_id,
            "warnings": decision.warnings,
            "chat_mode": chat_mode.value,
            "response_template_id": decision.response_template_id,
            "next_action_code": track.action_code,
            "required_context": decision.required_context,
            "missing_context": decision.missing_context,
            "capability_supported": track.capability_supported,
            "fallback_used": track.fallback_used,
            "policy_flags": decision.policy_flags,
            "context_state": tuple(dict.fromkeys(
                decision.context_state + tuple(
                    tag for tag in classification.personalization_tag_candidates
                    if tag.startswith(("current_", "emotion_", "engagement_"))
                    or tag in {"frustration", "service_dissatisfaction"}
                )
            )),
            "current_place_id": current_place_id,
            "current_piece_id": current_piece_id,
            "completed_piece_ids": visited_piece_ids,
            "game_state_mutation": False,
            "mode_transition": asdict(track.transition) if track.transition else None,
            "rag_used": track.should_retrieve,
            "storage_requested": track.action_code == "SAVE_SHORT_REFLECTION",
            "storage_permitted": (
                track.action_code == "SAVE_SHORT_REFLECTION"
                and storage_capability and user_consent
                and track.action_code in available_capabilities
            ),
            "request_state": track.request_state.value,
            "ui_state": track.ui_state,
            "suggested_questions": track.free_ui.suggested_questions if track.free_ui else (),
        }
        if not track.should_retrieve:
            response = ChatResponse(
                track.response_override or decision.answer, "ok", (), 0, session.session_id, locale, PROMPT_VERSION,
                grounded=False,
                latency_ms=round((time.perf_counter() - started) * 1000),
                **common,
            )
            self.sessions.add_turn(session.session_id, query, response.answer)
            return response
        previous = session.turns[-1].user if session.turns else ""
        search_query = self._rewrite_followup(query, previous)
        chunks = self._select(self.retrieval.search(search_query), top_k)
        self._assert_mode_boundary(chunks)
        conversation = self._conversation_lines(session)
        budget = self.budget.fit(
            system_prompt=SYSTEM_INSTRUCTIONS,
            user_prompt=query,
            evidence=[item.chunk.text for item in chunks],
            conversation=conversation,
            max_new_tokens=self.max_new_tokens,
        )
        chunks = chunks[: len(budget.evidence)]
        prompt = build_prompt(
            user_query=self._journey_scoped_query(query, classification.primary_situation_id.value, visited_piece_ids),
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
        )
        if not chunks:
            insufficient_common = dict(common)
            insufficient_common.update(request_state="insufficient_evidence", ui_state="insufficient_evidence", rag_used=True)
            response = ChatResponse(
                "확인 가능한 자료가 부족합니다.",
                "insufficient_evidence",
                (),
                0,
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                },
                refusal_reason="insufficient_evidence",
                latency_ms=round((time.perf_counter() - started) * 1000),
                **insufficient_common,
            )
        else:
            is_fixture = all(
                item.chunk.payload.get("data_classification") == "fictional_fixture"
                for item in chunks
            )
            request = self._llm_request(
                query, prompt, session, chunks, is_fixture, budget
            )
            try:
                completion = self.llm.complete(request)
                citations = build_citations(chunks)
                answer = self._apply_hackathon_policy(
                    completion.generated_text, chunks
                )
                if chat_mode == ConversationMode.PIECE_CHAT:
                    answer = self._limit_piece_answer(answer)
                response = ChatResponse(
                    answer,
                    "ok",
                    citations,
                    len(chunks),
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    context_metadata=request.metadata.get("context_budget"),  # type: ignore[arg-type]
                    citations=tuple(asdict(item) for item in citations),
                    evidence=tuple(item.chunk.text for item in chunks),
                    grounded=True,
                    retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                    retrieved_source_ids=tuple(dict.fromkeys(str(item.chunk.payload.get("source_id", item.chunk.document_id)) for item in chunks)),
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    **(dict(common) | {"request_state": "success", "ui_state": "showing_citations", "rag_used": True}),
                    **self._provisional_metadata(chunks),
                )
            except RemoteLLMError as error:
                response = ChatResponse(
                    "",
                    "llm_error",
                    (),
                    0,
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    error=error.to_dict(),
                    context_metadata=request.metadata.get("context_budget"),  # type: ignore[arg-type]
                    refusal_reason="llm_error",
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    **common,
                )
        self.sessions.add_turn(session.session_id, query, response.answer)
        return response

    def stream(self, *args, **kwargs) -> Iterator[StreamEvent]:
        user_query = args[0] if args else kwargs.pop("user_query")
        session_id = kwargs.pop("session_id", None)
        locale = kwargs.pop("locale", "ko")
        top_k = kwargs.pop("top_k", 3)
        if kwargs:
            raise TypeError(f"지원하지 않는 인자: {', '.join(kwargs)}")
        query = self._validate(user_query, locale, top_k)
        session = self.sessions.get_or_create(session_id, locale)
        previous = session.turns[-1].user if session.turns else ""
        chunks = self._select(
            self.retrieval.search(self._rewrite_followup(query, previous)), top_k
        )
        self._assert_mode_boundary(chunks)
        budget = self.budget.fit(
            system_prompt=SYSTEM_INSTRUCTIONS,
            user_prompt=query,
            evidence=[item.chunk.text for item in chunks],
            conversation=self._conversation_lines(session),
            max_new_tokens=self.max_new_tokens,
        )
        chunks = chunks[: len(budget.evidence)]
        if not chunks:
            response = ChatResponse(
                "확인 가능한 자료가 부족합니다.",
                "insufficient_evidence",
                (),
                0,
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                },
            )
            self.sessions.add_turn(session.session_id, query, response.answer)
            yield StreamEvent("completed", response.to_dict())
            return
        prompt = build_prompt(
            user_query=query,
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
        )
        is_fixture = all(
            item.chunk.payload.get("data_classification") == "fictional_fixture"
            for item in chunks
        )
        request = self._llm_request(query, prompt, session, chunks, is_fixture, budget)
        for event in self.llm.stream_complete(request):
            if event.event in {"start", "token", "delta"}:
                yield StreamEvent(event.event, event.data)
            elif event.event == "error":
                response = ChatResponse(
                    "",
                    "llm_error",
                    (),
                    0,
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    error=event.data,
                    context_metadata=request.metadata.get("context_budget"),  # type: ignore[arg-type]
                )
                self.sessions.add_turn(session.session_id, query, "")
                yield StreamEvent("error", response.to_dict())
                return
            elif event.event == "completed":
                answer = self._apply_hackathon_policy(
                    str(event.data.get("generated_text", "")), chunks
                )
                response = ChatResponse(
                    answer,
                    "ok",
                    build_citations(chunks),
                    len(chunks),
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    context_metadata=request.metadata.get("context_budget"),  # type: ignore[arg-type]
                    **self._provisional_metadata(chunks),
                )
                self.sessions.add_turn(session.session_id, query, answer)
                yield StreamEvent(
                    "completed",
                    {
                        **response.to_dict(),
                        "llm": event.data,
                    },
                )
                return
        error = RemoteLLMError(
            "generation_failed",
            "LLM 스트림이 완료 이벤트 없이 종료되었습니다.",
            retryable=True,
        )
        yield StreamEvent("error", {"status": "llm_error", "error": error.to_dict()})

    def reset(self, session_id: str) -> bool:
        return self.sessions.reset(session_id)

    @staticmethod
    def _validate(query: str, locale: str, top_k: int) -> str:
        value = query.strip()
        if not value:
            raise ValueError("질문을 입력하세요.")
        if len(value) > 2000:
            raise ValueError("질문은 2,000자 이하여야 합니다.")
        if not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", locale):
            raise ValueError("locale 형식이 올바르지 않습니다.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k는 1~10이어야 합니다.")
        return value

    @staticmethod
    def _rewrite_followup(query: str, previous_query: str) -> str:
        if previous_query and re.search(r"(그곳|그 건물|그때|그와|그 과정|그 자료)", query):
            return f"{previous_query} {query}"
        return query

    @staticmethod
    def _journey_scoped_query(query: str, situation_id: str, visited_piece_ids: tuple[str, ...]) -> str:
        if situation_id != "JOURNEY_CONTEXT_QUESTION":
            return query
        completed = ", ".join(visited_piece_ids) if visited_piece_ids else "없음"
        return (
            f"{query}\n[게임 메타데이터] 실제 완료 조각 ID: {completed}. "
            "이 목록 밖의 조각을 완료했다고 말하지 마세요. 역사 관계는 검색 근거와 구분하세요."
        )

    def _select(self, results: list[RankedChunk], top_k: int) -> list[RankedChunk]:
        selected: list[RankedChunk] = []
        seen_chunks: set[str] = set()
        per_document: Counter[str] = Counter()
        for item in results:
            if item.chunk.chunk_id in seen_chunks:
                continue
            if per_document[item.chunk.document_id] >= self.max_chunks_per_document:
                continue
            seen_chunks.add(item.chunk.chunk_id)
            per_document[item.chunk.document_id] += 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _conversation_lines(session: ChatSession) -> list[str]:
        lines = [session.summary] if session.summary else []
        lines.extend(
            f"사용자: {turn.user}\n응답: {turn.assistant}" for turn in session.turns[-3:]
        )
        return lines

    def _llm_request(self, query, prompt, session, chunks, is_fixture, budget):
        remote_config = getattr(self.llm, "config", None)
        return LLMRequest(
            system_prompt=SYSTEM_INSTRUCTIONS,
            user_prompt=prompt,
            messages=tuple(
                message
                for turn in session.turns[-3:]
                for message in (
                    LLMMessage("user", turn.user),
                    LLMMessage("assistant", turn.assistant),
                )
            ),
            temperature=getattr(remote_config, "temperature", 0.2),
            top_p=getattr(remote_config, "top_p", 0.9),
            max_new_tokens=getattr(remote_config, "max_new_tokens", self.max_new_tokens),
            request_id=uuid.uuid4().hex,
            timeout=getattr(remote_config, "timeout_seconds", 60),
            metadata={
                "evidence": tuple(item.chunk.text for item in chunks),
                "is_fixture": is_fixture,
                "context_budget": {
                    "estimated_input_tokens": budget.estimated_input_tokens,
                    "reserved_output_tokens": budget.reserved_output_tokens,
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                },
                "user_query": query,
            },
        )

    def _assert_mode_boundary(self, chunks: list[RankedChunk]) -> None:
        provisional = any(
            item.chunk.payload.get("usage_status") == "provisional_hackathon"
            for item in chunks
        )
        if provisional and self.mode != RuntimeMode.HACKATHON:
            raise ValueError(
                "provisional_hackathon 자료는 hackathon 모드 외 검색·프롬프트에 사용할 수 없습니다."
            )

    @staticmethod
    def _provisional_metadata(chunks: list[RankedChunk]) -> dict[str, object]:
        source_ids = tuple(
            dict.fromkeys(
                str(item.chunk.payload.get("source_id", item.chunk.document_id))
                for item in chunks
                if item.chunk.payload.get("usage_status") == "provisional_hackathon"
            )
        )
        if not source_ids:
            return {
                "provisional_sources_used": 0,
                "rights_notice": "",
                "usage_scope": "",
                "source_ids": (),
            }
        return {
            "provisional_sources_used": len(source_ids),
            "rights_notice": (
                "이 답변은 비상업적 해커톤 시연을 위해 수집한 공식 기관 참고자료를 "
                "기반으로 생성되었습니다. 일부 자료의 재사용 범위는 검토 중이며, "
                "원문은 표시된 공식 출처에서 확인할 수 있습니다."
            ),
            "usage_scope": "noncommercial_hackathon_demo",
            "source_ids": source_ids,
        }

    def _apply_hackathon_policy(
        self, answer: str, chunks: list[RankedChunk]
    ) -> str:
        if not any(
            item.chunk.payload.get("usage_status") == "provisional_hackathon"
            for item in chunks
        ):
            return answer
        # 원격 모델이 근거를 장문 복원하는 경우에도 시연 응답 크기를 제한한다.
        normalized = answer.strip()
        if len(normalized) > 1200:
            normalized = normalized[:1200].rstrip() + "…"
        return normalized

    @staticmethod
    def _limit_piece_answer(answer: str) -> str:
        sentences = re.split(r"(?<=[.!?。！？])\s+", answer.strip())
        return " ".join(sentences[:2]).strip()
