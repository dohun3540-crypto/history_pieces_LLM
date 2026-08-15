"""세션부터 검색·프롬프트·답변·출처까지 연결하는 grounded RAG."""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Iterator

from history_chatbot.chat.citation_builder import build_citations
from history_chatbot.chat.context_resolver import (
    ConversationRequestKind,
    ConversationContextResolver,
    ResolvedContext,
    is_placeholder_context,
)
from history_chatbot.chat.interfaces import Citation
from history_chatbot.chat.prompt_builder import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTIONS,
    build_prompt,
)
from history_chatbot.chat.remote_safe import serialize_remote_prompt
from history_chatbot.chat.session import ChatSession, SessionStore
from history_chatbot.dialogue.response_policy import GiroksaeDialogueEngine
from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.persona import (
    OutputDomain, PERSONA_ID, SourceSufficiency,
    build_persona_prompt,
    conversation_stage_for, locale_policy, output_domain_for,
    render_mock_grounded, speech_level_for,
)
from history_chatbot.dialogue.response_renderer import GiroksaeResponseRenderer
from history_chatbot.dialogue.situation_models import ClassificationInput, ScreenType, SituationId
from history_chatbot.dialogue.track_models import SharedSessionContext
from history_chatbot.dialogue.track_policy import ChatTrackPolicy
from history_chatbot.models.context_budget import ContextBudgetManager
from history_chatbot.models.contract import ChatCompletionBackend, LLMMessage, LLMRequest
from history_chatbot.models.remote import RemoteLLMError
from history_chatbot.retrieval.base import RankedChunk
from history_chatbot.retrieval.query_normalizer import explicit_subject_words
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
    completed_place_ids: tuple[str, ...] = ()
    completed_piece_ids: tuple[str, ...] = ()
    game_state_mutation: bool = False
    mode_transition: dict[str, object] | None = None
    rag_used: bool = False
    storage_requested: bool = False
    storage_permitted: bool = False
    request_state: str = "success"
    ui_state: str = "active"
    suggested_questions: tuple[str, ...] = ()
    output_domain: str = "character_dialogue"
    speech_level: str = "banmal"
    persona_id: str = PERSONA_ID
    language: str = "ko"
    culture: str = "korea"
    conversation_stage: str = "historical_question"
    source_sufficiency: str = "sufficient"
    translation_status: str = "native_policy"

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


@dataclass(frozen=True, slots=True)
class PreparedTurnEvidence:
    resolved_context: ResolvedContext
    chunks: tuple[RankedChunk, ...]
    needs_new_evidence: bool
    retrieval_performed: bool
    memory_evidence_used: bool
    partial_evidence_used: bool
    detail_evidence_sufficient: bool | None
    requested_detail_supported: bool


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
        self.response_renderer = GiroksaeResponseRenderer()
        self.context_resolver = ConversationContextResolver()

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
        completed_place_ids: tuple[str, ...] = (),
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
        resolved_screen_type = ScreenType(resolved_screen)
        if resolved_screen_type.value != chat_mode.value and not (
            chat_mode == ConversationMode.PIECE_CHAT
            and resolved_screen_type == ScreenType.INTRO
        ):
            raise ValueError("chat_mode와 screen_type이 일치해야 합니다.")
        shared_context = SharedSessionContext(
            session_id=session.session_id, locale=locale,
            current_place_id=current_place_id, current_piece_id=current_piece_id,
            completed_place_ids=completed_place_ids,
            completed_piece_ids=visited_piece_ids, current_journey_step=current_journey_step,
            temporary_response_length_preference=(existing_style_preferences[0] if existing_style_preferences else None),
            available_capabilities=available_capabilities,
            storage_capability=storage_capability, user_consent=user_consent,
        )
        classification_input = ClassificationInput(
            query,
            conversation_mode=conversation_mode,
            screen_type=resolved_screen_type,
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
        output_domain = output_domain_for(classification.primary_situation_id)
        if track.action_code == "SAVE_SHORT_REFLECTION":
            output_domain = OutputDomain.SYSTEM_UI
        speech_level = speech_level_for(output_domain)
        stage = conversation_stage_for(classification.primary_situation_id)
        language, culture, translation_status = locale_policy(locale, output_domain)
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
            "completed_place_ids": completed_place_ids,
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
            "output_domain": output_domain.value,
            "speech_level": speech_level.value,
            "persona_id": PERSONA_ID,
            "language": language,
            "culture": culture,
            "conversation_stage": stage.value,
            "source_sufficiency": SourceSufficiency.SUFFICIENT.value,
            "translation_status": translation_status.value,
        }
        if not track.should_retrieve:
            raw_answer = track.response_override or decision.answer
            if locale.lower() == "zh-cn":
                raw_answer = self._pending_zh_text(output_domain)
            if (
                classification.primary_situation_id == SituationId.INTRO_GIROKSAE
                and session.turns
            ):
                raw_answer = raw_answer.replace("위대한 기록새", "기록새")
            guarded_answer, style_warnings = self._guard_answer(
                raw_answer,
                output_domain=output_domain,
                situation=classification.primary_situation_id,
                stage=stage,
                locale=locale,
            )
            non_rag_common = dict(common)
            non_rag_common["warnings"] = tuple(dict.fromkeys(common["warnings"] + style_warnings))
            response = ChatResponse(
                guarded_answer, "ok", (), 0, session.session_id, locale, PROMPT_VERSION,
                grounded=False,
                latency_ms=round((time.perf_counter() - started) * 1000),
                **non_rag_common,
            )
            self.sessions.add_turn(session.session_id, query, response.answer)
            return response
        prepared = self._prepare_turn_evidence(
            query,
            session,
            top_k=top_k,
            current_place_id=current_place_id,
            current_piece_id=current_piece_id,
        )
        resolved_context = prepared.resolved_context
        search_query = resolved_context.search_query
        chunks = list(prepared.chunks)
        detail_supported = prepared.requested_detail_supported
        interpreted_query = self._conversation_scoped_query(
            query,
            search_query=search_query,
            followup_resolved=resolved_context.followup_resolved,
        )
        conversation = self._conversation_lines(session)
        runtime_system_prompt = SYSTEM_INSTRUCTIONS + "\n" + build_persona_prompt(
            domain=output_domain, locale=locale, mode=chat_mode,
            situation=classification.primary_situation_id, stage=stage,
        )
        budget = self.budget.fit(
            system_prompt=runtime_system_prompt,
            user_prompt=interpreted_query,
            evidence=[item.chunk.text for item in chunks],
            conversation=conversation,
            max_new_tokens=self.max_new_tokens,
        )
        chunks = chunks[: len(budget.evidence)]
        sufficiency = self._source_sufficiency(chunks)
        if chunks and (
            not detail_supported
            or (
                prepared.partial_evidence_used
                and resolved_context.request_kind
                != ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
            )
        ):
            sufficiency = SourceSufficiency.PARTIAL
        if (
            sufficiency == SourceSufficiency.SUFFICIENT
            and len(chunks) == 1
            and chunks[0].chunk.payload.get("usage_status") == "verified_hackathon"
        ):
            sufficiency = SourceSufficiency.PARTIAL
        contextual_query = self._contextualize_query(
            self._journey_scoped_query(
                interpreted_query,
                classification.primary_situation_id.value,
                visited_piece_ids,
            ),
            current_place_id=current_place_id,
            current_piece_id=current_piece_id,
            completed_place_ids=completed_place_ids,
            completed_piece_ids=visited_piece_ids,
        )
        contextual_query = self._evidence_scoped_query(
            contextual_query, prepared, selected_chunks=chunks
        )
        prompt = build_prompt(
            user_query=query,
            resolved_question=contextual_query,
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
            conversation_mode=chat_mode,
            output_domain=output_domain,
            situation=classification.primary_situation_id,
            conversation_stage=stage,
            include_system=False,
            conversation_in_messages=True,
        )
        if not chunks:
            fallback_place = (
                resolved_context.active_place
                if resolved_context.followup_resolved
                or resolved_context.active_place in search_query
                else ""
            )
            fallback_answer, fallback_suggestions = self._insufficient_guidance(
                query,
                output_domain,
                locale,
                active_place=fallback_place,
            )
            insufficient_common = dict(common)
            insufficient_common.update(
                request_state="insufficient_evidence", ui_state="insufficient_evidence",
                rag_used=True, source_sufficiency=SourceSufficiency.INSUFFICIENT.value,
                suggested_questions=fallback_suggestions,
            )
            response = ChatResponse(
                fallback_answer,
                "insufficient_evidence",
                (),
                0,
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
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
                contextual_query, prompt, session, chunks, is_fixture, budget,
                locale=locale, conversation_mode=chat_mode,
                output_domain=output_domain,
                situation=classification.primary_situation_id,
                stage=stage,
            )
            try:
                completion = self.llm.complete(request)
                citations = build_citations(chunks)
                completion_text, generation_warnings = self._completion_text(completion)
                answer = self._apply_hackathon_policy(
                    completion_text, chunks
                )
                answer = self._apply_repetition_guard(
                    answer, output_domain=output_domain
                )
                answer, stabilization_warnings, output_limited = (
                    self._stabilize_grounded_answer(answer, query=query, chunks=chunks)
                )
                if output_limited and sufficiency == SourceSufficiency.SUFFICIENT:
                    sufficiency = SourceSufficiency.PARTIAL
                if self.llm.backend_name == "mock":
                    answer = render_mock_grounded(answer, domain=output_domain, locale=locale)
                    if sufficiency == SourceSufficiency.CONFLICTING:
                        answer = "자료마다 설명이 달라. 확인된 차이를 나눠서 볼게. " + answer
                    elif sufficiency == SourceSufficiency.PARTIAL:
                        answer = "확인되는 범위부터 말씀드리면, " + answer
                if chat_mode == ConversationMode.PIECE_CHAT:
                    answer = self._limit_piece_answer(answer)
                answer, style_warnings = self._guard_answer(
                    answer, output_domain=output_domain,
                    situation=classification.primary_situation_id,
                    stage=stage, locale=locale,
                    citations=tuple(asdict(item) for item in citations),
                )
                grounded_common = dict(common) | {
                    "request_state": "success", "ui_state": "showing_citations",
                    "rag_used": True,
                    "source_sufficiency": sufficiency.value,
                    "warnings": tuple(dict.fromkeys(
                        common["warnings"] + generation_warnings
                        + stabilization_warnings + style_warnings
                    )),
                }
                response = ChatResponse(
                    answer,
                    "ok",
                    citations,
                    len(chunks),
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    context_metadata={
                        **dict(request.metadata.get("context_budget", {})),
                        "active_piece": resolved_context.active_piece,
                        "finish_reason": getattr(completion, "finish_reason", "stop"),
                        **self._evidence_decision_metadata(prepared, chunks),
                    },
                    citations=tuple(asdict(item) for item in citations),
                    evidence=tuple(item.chunk.text for item in chunks),
                    grounded=True,
                    retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                    retrieved_source_ids=tuple(dict.fromkeys(str(item.chunk.payload.get("source_id", item.chunk.document_id)) for item in chunks)),
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    **grounded_common,
                    **self._provisional_metadata(chunks),
                )
            except RemoteLLMError as error:
                error_common = dict(common)
                error_common.update(request_state="error", ui_state="error")
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
                    **error_common,
                )
        self.sessions.add_turn(session.session_id, query, response.answer)
        if chunks:
            if prepared.retrieval_performed:
                self.sessions.add_evidence_turn(
                    session.session_id,
                    user=query,
                    active_place=resolved_context.active_place,
                    active_topic=resolved_context.active_topic,
                    chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                )
            retrieved_entities = tuple(dict.fromkeys(
                title
                for title in (item.chunk.title for item in chunks)
                if title and not is_placeholder_context(title)
            ))[:8]
            self.sessions.update_context(
                session.session_id,
                recent_entities=tuple(dict.fromkeys(
                    (*resolved_context.recent_entities, *retrieved_entities)
                ))[:8],
                recent_people=self._recent_people(
                    chunks, resolved_context.recent_people
                ),
            )
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
        prepared = self._prepare_turn_evidence(
            query, session, top_k=top_k,
            current_place_id=None, current_piece_id=None,
        )
        resolved_context = prepared.resolved_context
        chunks = list(prepared.chunks)
        interpreted_query = self._conversation_scoped_query(
            query,
            search_query=resolved_context.search_query,
            followup_resolved=resolved_context.followup_resolved,
        )
        stream_system_prompt = SYSTEM_INSTRUCTIONS + "\n" + build_persona_prompt(
            domain=OutputDomain.CHARACTER_DIALOGUE,
            locale=locale,
            mode=ConversationMode.FREE_CHAT,
            situation=SituationId.HISTORY_FACT_QUESTION,
            stage=None,
        )
        budget = self.budget.fit(
            system_prompt=stream_system_prompt,
            user_prompt=interpreted_query,
            evidence=[item.chunk.text for item in chunks],
            conversation=self._conversation_lines(session),
            max_new_tokens=self.max_new_tokens,
        )
        chunks = chunks[: len(budget.evidence)]
        interpreted_query = self._evidence_scoped_query(
            interpreted_query, prepared, selected_chunks=chunks
        )
        if not chunks:
            fallback, suggestions = self._insufficient_guidance(
                query, OutputDomain.CHARACTER_DIALOGUE, locale
            )
            response = ChatResponse(
                fallback,
                "insufficient_evidence",
                (),
                0,
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
                },
                request_state="insufficient_evidence",
                ui_state="insufficient_evidence",
                source_sufficiency=SourceSufficiency.INSUFFICIENT.value,
                suggested_questions=suggestions,
            )
            self.sessions.add_turn(session.session_id, query, response.answer)
            yield StreamEvent("completed", response.to_dict())
            return
        prompt = build_prompt(
            user_query=query,
            resolved_question=interpreted_query,
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
            include_system=False,
            conversation_in_messages=True,
        )
        is_fixture = all(
            item.chunk.payload.get("data_classification") == "fictional_fixture"
            for item in chunks
        )
        request = self._llm_request(
            interpreted_query, prompt, session, chunks, is_fixture, budget,
            locale=locale, conversation_mode=ConversationMode.FREE_CHAT,
            output_domain=OutputDomain.CHARACTER_DIALOGUE,
            situation=SituationId.HISTORY_FACT_QUESTION,
            stage=None,
        )
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
                    request_state="error",
                    ui_state="error",
                )
                self.sessions.add_turn(session.session_id, query, "")
                yield StreamEvent("error", response.to_dict())
                return
            elif event.event == "completed":
                try:
                    completion_text, generation_warnings = self._completion_text_values(
                        str(event.data.get("generated_text", "")),
                        str(event.data.get("finish_reason", "stop")),
                    )
                except RemoteLLMError as error:
                    yield StreamEvent("error", {
                        "status": "llm_error", "request_state": "error",
                        "ui_state": "error", "error": error.to_dict(),
                    })
                    return
                answer = self._apply_hackathon_policy(completion_text, chunks)
                answer = self._apply_repetition_guard(
                    answer, output_domain=OutputDomain.CHARACTER_DIALOGUE
                )
                answer, stabilization_warnings, _output_limited = (
                    self._stabilize_grounded_answer(
                        answer, query=user_query, chunks=chunks
                    )
                )
                response = ChatResponse(
                    answer,
                    "ok",
                    build_citations(chunks),
                    len(chunks),
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    context_metadata={
                        **dict(request.metadata.get("context_budget", {})),
                        "finish_reason": event.data.get("finish_reason", "stop"),
                        **self._evidence_decision_metadata(prepared, chunks),
                    },
                    warnings=tuple(dict.fromkeys(
                        generation_warnings + stabilization_warnings
                    )),
                    **self._provisional_metadata(chunks),
                )
                self.sessions.add_turn(session.session_id, query, answer)
                if chunks and prepared.retrieval_performed:
                    self.sessions.add_evidence_turn(
                        session.session_id,
                        user=query,
                        active_place=resolved_context.active_place,
                        active_topic=resolved_context.active_topic,
                        chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                    )
                self.sessions.update_context(
                    session.session_id,
                    recent_people=self._recent_people(
                        chunks, resolved_context.recent_people
                    ),
                )
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
        explicit_people_followup = re.fullmatch(
            r"(?:관련(?:된)?\s*(?:인물|사람)(?:은|이)?"
            r"(?:\s*누구(?:인가요|예요|야)?)?"
            r"|누가\s*(?:참여|관여|주도)(?:했나요|했어요|했어|했습니까)?)"
            r"[?.!]?",
            query.strip(),
        )
        if previous_query and (
            re.search(r"(그곳|그 건물|그때|그와|그 과정|그 자료)", query)
            or explicit_people_followup
        ):
            return f"{previous_query} {query}"
        return query

    @staticmethod
    def _conversation_scoped_query(
        query: str, *, search_query: str, followup_resolved: bool
    ) -> str:
        """해석된 지시 대상을 대화 문맥으로 표시하되 역사 근거로 승격하지 않는다."""

        if not followup_resolved or search_query.strip() == query.strip():
            return query
        return (
            f"{query}\n[대화 문맥 해석 | 역사적 사실의 근거가 아님] "
            f"{search_query}"
        )

    @staticmethod
    def _journey_scoped_query(query: str, situation_id: str, visited_piece_ids: tuple[str, ...]) -> str:
        if situation_id != "JOURNEY_CONTEXT_QUESTION":
            return query
        historical_piece_ids = tuple(
            value for value in visited_piece_ids if not is_placeholder_context(value)
        )
        completed = ", ".join(historical_piece_ids) if historical_piece_ids else "없음"
        return (
            f"{query}\n[게임 메타데이터] 실제 완료 조각 ID: {completed}. "
            "이 목록 밖의 조각을 완료했다고 말하지 마세요. 역사 관계는 검색 근거와 구분하세요."
        )

    @staticmethod
    def _contextualize_query(
        query: str,
        *,
        current_place_id: str | None,
        current_piece_id: str | None,
        completed_place_ids: tuple[str, ...],
        completed_piece_ids: tuple[str, ...],
    ) -> str:
        context: list[str] = []
        if current_place_id and not is_placeholder_context(current_place_id):
            context.append(f"현재 장소 ID: {current_place_id}")
        if current_piece_id and not is_placeholder_context(current_piece_id):
            context.append(f"현재 조각 ID: {current_piece_id}")
        safe_places = tuple(
            value for value in completed_place_ids if not is_placeholder_context(value)
        )
        safe_pieces = tuple(
            value for value in completed_piece_ids if not is_placeholder_context(value)
        )
        if safe_places:
            context.append("완료 장소 ID: " + ", ".join(safe_places))
        if safe_pieces:
            context.append("완료 조각 ID: " + ", ".join(safe_pieces))
        if not context:
            return query
        return (
            query
            + "\n[관광 여정 문맥 | 역사적 사실의 근거가 아님] "
            + "; ".join(context)
        )

    def _prepare_turn_evidence(
        self,
        query: str,
        session: ChatSession,
        *,
        top_k: int,
        current_place_id: str | None,
        current_piece_id: str | None,
    ) -> PreparedTurnEvidence:
        """Share conversation resolution and evidence policy across generation modes."""

        resolved = self.context_resolver.resolve(
            query,
            session,
            current_place_id=current_place_id,
            current_piece_id=current_piece_id,
        )
        self.sessions.update_context(
            session.session_id,
            active_place=resolved.active_place,
            active_piece=resolved.active_piece,
            active_topic=resolved.active_topic,
            recent_entities=resolved.recent_entities,
            recent_people=resolved.recent_people,
            recent_event=resolved.recent_event,
            recent_period=resolved.recent_period,
        )
        place_filter = (
            resolved.active_place
            if resolved.active_place in resolved.search_query else ""
        )
        reuses_memory = resolved.request_kind in {
            ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER,
            ConversationRequestKind.EXPAND_PREVIOUS_ANSWER,
        }
        remembered = (
            self._remembered_evidence(
                session,
                active_place=resolved.active_place,
                active_topic=resolved.active_topic,
                top_k=top_k,
                prefer_latest=True,
            )
            if reuses_memory else []
        )
        remembered = self._select(
            self._place_aware_results(remembered, place_filter), top_k
        )

        detail_sufficient: bool | None = None
        if resolved.request_kind == ConversationRequestKind.EXPAND_PREVIOUS_ANSWER:
            detail_sufficient = self._detail_evidence_sufficient(
                resolved.resolved_question, remembered
            )
            needs_new_evidence = not detail_sufficient
        elif resolved.request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER:
            needs_new_evidence = not remembered
        else:
            needs_new_evidence = True

        retrieval_performed = needs_new_evidence
        retrieved: list[RankedChunk] = []
        if retrieval_performed:
            retrieved = self._select(
                self._place_aware_results(
                    self._subject_aware_search(resolved.search_query, top_k), place_filter
                ),
                top_k,
            )

        partial_evidence_used = bool(remembered and retrieval_performed)
        if partial_evidence_used:
            chunks = self._compose_partial_evidence(remembered, retrieved, top_k)
        elif remembered and not retrieval_performed:
            chunks = remembered
        else:
            chunks = retrieved

        explicit_topic_return = bool(
            re.search(r"돌아가|돌아오|다시\s*(?:첫|처음)\s*(?:사건|주제)", query)
        )
        if not chunks and not explicit_topic_return and (
            resolved.followup_resolved
            or not self._supports_requested_detail(query, chunks)
        ):
            chunks = self._remembered_evidence(
                session,
                active_place=resolved.active_place,
                active_topic=resolved.active_topic,
                top_k=top_k,
            )
            chunks = self._select(
                self._place_aware_results(chunks, place_filter), top_k
            )
            remembered = chunks
            partial_evidence_used = bool(chunks and retrieval_performed)

        self._assert_mode_boundary(chunks)
        remembered_ids = {item.chunk.chunk_id for item in remembered}
        memory_evidence_used = any(
            item.chunk.chunk_id in remembered_ids for item in chunks
        )
        return PreparedTurnEvidence(
            resolved_context=resolved,
            chunks=tuple(chunks),
            needs_new_evidence=needs_new_evidence,
            retrieval_performed=retrieval_performed,
            memory_evidence_used=memory_evidence_used,
            partial_evidence_used=partial_evidence_used,
            detail_evidence_sufficient=detail_sufficient,
            requested_detail_supported=self._supports_requested_detail(query, chunks),
        )

    def _subject_aware_search(
        self, search_query: str, top_k: int
    ) -> list[RankedChunk]:
        """Preserve one result per explicit subject in comparison questions."""

        subjects = explicit_subject_words(search_query)
        is_comparison = bool(re.search(r"구분|비교|차이|둘(?:을|은|이)|같은\s*단체", search_query))
        if not is_comparison or not 2 <= len(subjects) <= 3:
            return self.retrieval.search(search_query)
        batches = [self.retrieval.search(subject) for subject in subjects]
        primary = [batch[0] for batch in batches if batch]
        remainder = [item for batch in batches for item in batch[1:]]
        return self._select([*primary, *remainder], top_k)

    def _compose_partial_evidence(
        self,
        remembered: list[RankedChunk],
        retrieved: list[RankedChunk],
        top_k: int,
    ) -> list[RankedChunk]:
        """Keep relevant prior provenance while adding new, deduplicated evidence."""

        candidates = [*remembered[:1], *retrieved, *remembered[1:]]
        return self._select(candidates, top_k)

    @staticmethod
    def _detail_evidence_sufficient(
        resolved_question: str, chunks: list[RankedChunk]
    ) -> bool:
        if not chunks:
            return False
        evidence = " ".join(
            f"{item.chunk.title} {item.chunk.text} "
            f"{item.chunk.payload.get('keywords', '')} "
            f"{item.chunk.payload.get('period', '')}"
            for item in chunks
        )
        aspects = {
            "background": (r"배경|이유|원인|계기", r"배경|이유|원인|계기|때문|위해|따라"),
            "result": (r"결과|영향|이후|그\s*뒤", r"결과|영향|이후|이어졌|되었|됐다|남겼"),
            "people": (r"인물|사람|누가|누구|참석자", r"인물|사람|참석|주도|대표|장관|교수"),
            "time": (r"시점|언제|연도|날짜|당시", r"(?:18|19|20)\d{2}년|\d{1,2}월|\d{1,2}일|당시"),
            "process": (r"과정|전개|구체적|어떻게", r"과정|전개|진행|도착|참석|개최|발생"),
        }
        requested = {
            name for name, (query_pattern, _evidence_pattern) in aspects.items()
            if re.search(query_pattern, resolved_question)
        }
        supported = {
            name for name, (_query_pattern, evidence_pattern) in aspects.items()
            if re.search(evidence_pattern, evidence)
        }
        if requested:
            return requested <= supported and len(evidence) >= 160
        return len(evidence) >= 220 and (
            len(supported) >= 3 or (len(chunks) >= 2 and len(evidence) >= 350)
        )

    @staticmethod
    def _evidence_decision_metadata(
        prepared: PreparedTurnEvidence,
        selected_chunks: list[RankedChunk] | None = None,
    ) -> dict[str, object]:
        resolved = prepared.resolved_context
        return {
            "followup_resolved": resolved.followup_resolved,
            "search_query": resolved.search_query,
            "resolved_question": resolved.resolved_question,
            "request_kind": resolved.request_kind.value,
            "needs_new_evidence": prepared.needs_new_evidence,
            "retrieval_performed": prepared.retrieval_performed,
            "memory_evidence_used": prepared.memory_evidence_used,
            "partial_evidence_used": prepared.partial_evidence_used,
            "detail_evidence_sufficient": prepared.detail_evidence_sufficient,
            "requested_detail_supported": prepared.requested_detail_supported,
            "selected_evidence_ids": tuple(
                item.chunk.chunk_id
                for item in (
                    selected_chunks
                    if selected_chunks is not None else prepared.chunks
                )
            ),
            "active_place": resolved.active_place,
        }

    @staticmethod
    def _evidence_scoped_query(
        query: str,
        prepared: PreparedTurnEvidence,
        *,
        selected_chunks: list[RankedChunk] | None = None,
    ) -> str:
        """Apply the same evidence-coverage guidance to sync and streaming prompts."""

        evidence = prepared.chunks if selected_chunks is None else selected_chunks
        if evidence and not prepared.requested_detail_supported:
            query += (
                "\n[근거 충족도] 질문한 세부사항은 검색 근거에 직접 없습니다. "
                "그 세부사항은 확인하기 어렵다고 밝히고, 근거로 확인되는 부분은 답하세요."
            )
        if any(
            item.chunk.payload.get("source_conflict") is True
            or item.chunk.payload.get("fact_status") == "conflicting"
            for item in evidence
        ):
            query += (
                "\n[근거 상태] 선택된 자료 사이에 충돌 표시가 있습니다. "
                "차이를 숨기거나 하나의 사실로 합치지 마세요."
            )
        return query

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

    def _remembered_evidence(
        self,
        session: ChatSession,
        *,
        active_place: str,
        active_topic: str,
        top_k: int,
        prefer_latest: bool = False,
    ) -> list[RankedChunk]:
        """Recover only chunks that retrieval actually returned in an earlier turn."""

        chosen_ids: tuple[str, ...] = ()
        for turn in reversed(session.evidence_turns):
            if prefer_latest:
                chosen_ids = turn.chunk_ids
                break
            same_place = bool(active_place and turn.active_place == active_place)
            same_topic = bool(active_topic and turn.active_topic == active_topic)
            if same_place or same_topic or (not active_place and not active_topic):
                chosen_ids = turn.chunk_ids
                break
        if not chosen_ids:
            return []
        by_id = {chunk.chunk_id: chunk for chunk in self.retrieval.store.chunks()}
        return [
            RankedChunk(by_id[chunk_id], 1.0, ("verified_conversation_memory",))
            for chunk_id in chosen_ids[:top_k]
            if chunk_id in by_id
        ]

    @staticmethod
    def _place_aware_results(
        results: list[RankedChunk], active_place: str
    ) -> list[RankedChunk]:
        if not active_place:
            return results
        compact_place = re.sub(r"\s+", "", active_place)
        if compact_place not in {"목포"}:
            exact = [
                item for item in results
                if compact_place in re.sub(
                    r"\s+", "", f"{item.chunk.title} {item.chunk.text}"
                )
            ]
            if exact:
                return exact
        return results

    @staticmethod
    def _supports_requested_detail(
        query: str, chunks: list[RankedChunk]
    ) -> bool:
        """Reject narrow visual/interior claims when retrieval lacks that detail."""

        if not re.search(r"내부|실내|안쪽|건축\s*양식|천장|색깔|색상", query):
            return True
        evidence = " ".join(item.chunk.text for item in chunks)
        requested_terms = {
            term
            for term in ("천장", "색깔", "색상", "내부", "실내", "안쪽", "건축 양식")
            if term in query
        }
        if requested_terms & {"천장", "색깔", "색상"}:
            return bool(requested_terms & {"천장", "색깔", "색상"} & set(re.findall(r"천장|색깔|색상", evidence)))
        return bool(re.search(r"내부|실내|대합실|매표소|승강장|평면|구조|건축\s*양식", evidence))

    @staticmethod
    def _conversation_lines(session: ChatSession) -> list[str]:
        lines = [f"[OLDER SUMMARY]\n{session.summary}"] if session.summary else []
        lines.extend(
            ConversationalRagOrchestrator._turn_line(turn.user, turn.assistant)
            for turn in session.turns[-4:]
        )
        return lines

    @staticmethod
    def _turn_line(user: str, assistant: str) -> str:
        return (
            f"[USER]\n{user}\n"
            f"[ASSISTANT | 대화 문맥, 사실 근거 아님]\n{assistant}"
        )

    @staticmethod
    def _recent_people(
        chunks: list[RankedChunk], existing: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Keep only user-resolved people; generated prose never creates referents."""

        del chunks
        return existing[:4]

    def _llm_request(
        self, query, prompt, session, chunks, is_fixture, budget, *,
        locale, conversation_mode, output_domain, situation, stage,
    ):
        remote_config = getattr(self.llm, "config", None)
        kept_conversation = set(budget.conversation)
        messages = tuple(
            message
            for turn in session.turns[-4:]
            if self._turn_line(turn.user, turn.assistant) in kept_conversation
            for message in (
                LLMMessage("user", turn.user),
                LLMMessage("assistant", turn.assistant),
            )
        )
        system_prompt = (
            SYSTEM_INSTRUCTIONS
            + "\n"
            + build_persona_prompt(
                domain=output_domain, locale=locale,
                mode=conversation_mode, situation=situation, stage=stage,
            )
        )
        user_prompt = prompt
        if self.llm.backend_name == "remote" and remote_config is not None:
            safe = serialize_remote_prompt(
                system_prompt=(
                    system_prompt
                ),
                user_query=query,
                chunks=chunks,
                history=tuple((turn.user, turn.assistant) for turn in session.turns),
                policy=remote_config.remote_prompt_policy(),
            )
            system_prompt = safe.system_prompt
            user_prompt = safe.user_prompt
            messages = safe.messages
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
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
        hackathon_only = any(
            item.chunk.payload.get("usage_status") in {
                "provisional_hackathon", "verified_hackathon"
            }
            for item in chunks
        )
        if hackathon_only and self.mode != RuntimeMode.HACKATHON:
            raise ValueError(
                "hackathon 전용 자료는 hackathon 모드 외 검색·프롬프트에 사용할 수 없습니다."
            )

    @staticmethod
    def _provisional_metadata(chunks: list[RankedChunk]) -> dict[str, object]:
        source_ids = tuple(
            dict.fromkeys(
                str(item.chunk.payload.get("source_id", item.chunk.document_id))
                for item in chunks
                if item.chunk.payload.get("usage_status") in {
                    "provisional_hackathon", "verified_hackathon"
                }
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
            item.chunk.payload.get("usage_status") in {
                "provisional_hackathon", "verified_hackathon"
            }
            for item in chunks
        ):
            return answer
        # 원격 모델이 근거를 장문 복원하는 경우에도 시연 응답 크기를 제한한다.
        normalized = answer.strip()
        if len(normalized) > 1200:
            normalized = self._complete_sentence_prefix(normalized, max_chars=1200)
            if not normalized:
                return "답변이 지나치게 길어 안전하게 표시하지 못했습니다. 질문 범위를 좁혀 다시 질문해 주세요."
        return normalized

    @staticmethod
    def _completion_text(completion) -> tuple[str, tuple[str, ...]]:
        """모델 출력의 종료 사유를 확인하고 미완성 꼬리를 노출하지 않는다."""

        return ConversationalRagOrchestrator._completion_text_values(
            str(completion.generated_text),
            str(getattr(completion, "finish_reason", "stop")),
        )

    @staticmethod
    def _completion_text_values(
        generated_text: str, finish_reason: str
    ) -> tuple[str, tuple[str, ...]]:
        text = re.sub(r"^\s*\[답변\]\s*", "", generated_text, count=1)
        finish_reason = finish_reason.casefold()
        if finish_reason not in {"length", "max_tokens"}:
            normalized = text.strip()
            incomplete_at = ConversationalRagOrchestrator._unclosed_delimiter_position(
                normalized
            )
            if incomplete_at is None:
                return normalized, ()
            complete = ConversationalRagOrchestrator._complete_sentence_prefix(
                normalized[:incomplete_at]
            )
            if not complete:
                raise RemoteLLMError(
                    "generation_failed",
                    "원격 LLM 응답에 열린 괄호나 미완성 구문이 남았습니다.",
                    retryable=True,
                )
            return complete, ("generation_incomplete_tail_removed",)
        complete = ConversationalRagOrchestrator._complete_sentence_prefix(
            text, discard_terminal_boundary=True
        )
        if not complete:
            return text.strip(), ("generation_no_complete_sentence",)
        return complete, ("generation_truncated_at_sentence_boundary",)

    @staticmethod
    def _unclosed_delimiter_position(text: str) -> int | None:
        pairs = {")": "(", "]": "[", "}": "{"}
        stack: list[tuple[str, int]] = []
        for index, character in enumerate(text):
            if character in "([{":
                stack.append((character, index))
            elif character in pairs:
                if stack and stack[-1][0] == pairs[character]:
                    stack.pop()
        return stack[0][1] if stack else None

    @staticmethod
    def _complete_sentence_prefix(
        text: str, *, max_chars: int | None = None,
        discard_terminal_boundary: bool = False,
    ) -> str:
        normalized = text.strip()
        limit = len(normalized) if max_chars is None else min(max_chars, len(normalized))
        boundaries = [
            match.end()
            for match in re.finditer(r"[.!?。！？](?=\s|$)", normalized[:limit])
        ]
        if discard_terminal_boundary and boundaries and boundaries[-1] == len(normalized):
            boundaries.pop()
        return normalized[:boundaries[-1]].strip() if boundaries else ""

    @staticmethod
    def _apply_repetition_guard(
        answer: str, *, output_domain: OutputDomain
    ) -> str:
        """Remove only conservative sentence repetition from docent answers."""
        if output_domain != OutputDomain.HISTORICAL_DOCENT:
            return answer

        sentences = re.split(r"(?<=[.!?。！？])\s+", answer.strip())
        kept: list[str] = []
        exact_keys: set[str] = set()
        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue
            exact_key = re.sub(r"\s+", " ", candidate)
            if exact_key in exact_keys:
                continue
            if any(
                ConversationalRagOrchestrator._is_near_duplicate_sentence(
                    candidate, previous
                )
                for previous in kept[-2:]
            ):
                continue
            kept.append(candidate)
            exact_keys.add(exact_key)
        return " ".join(kept)

    @staticmethod
    def _is_near_duplicate_sentence(left: str, right: str) -> bool:
        if min(len(left), len(right)) < 20:
            return False
        if re.findall(r"\d[\d,.]*", left) != re.findall(r"\d[\d,.]*", right):
            return False

        subject_pattern = re.compile(
            r"^(?:\[답변\]\s*)?([가-힣]{2,10})(?:은|는|이|가)(?:\s|,)"
        )
        left_subject = subject_pattern.match(left)
        right_subject = subject_pattern.match(right)
        if (
            left_subject is not None
            and right_subject is not None
            and left_subject.group(1) != right_subject.group(1)
        ):
            return False

        normalize = lambda value: re.sub(r"[\W_]+", "", value).casefold()
        left_normalized = normalize(left)
        right_normalized = normalize(right)
        if not left_normalized or not right_normalized:
            return False
        return SequenceMatcher(
            None, left_normalized, right_normalized, autojunk=False
        ).ratio() >= 0.94

    @staticmethod
    def _limit_piece_answer(answer: str) -> str:
        sentences = re.split(r"(?<=[.!?。！？])\s+", answer.strip())
        return " ".join(sentences[:2]).strip()

    @staticmethod
    def _stabilize_grounded_answer(
        answer: str, *, query: str, chunks: list[RankedChunk]
    ) -> tuple[str, tuple[str, ...], bool]:
        """Bound weak-model output and replace prompt leakage with a scoped limitation."""

        value = answer.strip()
        value = re.sub(
            r"^\s*(?:\[(?:답변|대화\s*문맥\s*해석|최종\s*답변)\]\s*)+",
            "",
            value,
        )
        raw_prompt_leak = bool(
            re.search(
                r"\[(?:검색\s*근거|자료\d+|사용자(?:\s*질문)?|USER)\]"
                r"|(?:^|\n)\s*(?:사용자|USER)\s*[:：]",
                value,
                re.IGNORECASE,
            )
        )
        suspicious_years = len(set(re.findall(r"(?:18|19|20|21|22|23|24)\d{2}년", value))) > 8
        sentences = re.split(r"(?<=[.!?。！？])\s+", value)
        echo_sentences = {
            index for index, sentence in enumerate(sentences)
            if ConversationalRagOrchestrator._is_question_echo(sentence, query)
        }
        if echo_sentences:
            value = " ".join(
                sentence.strip() for index, sentence in enumerate(sentences)
                if index not in echo_sentences and sentence.strip()
            )
        no_complete_sentence = not bool(
            re.search(r"[.!?。！？](?=\s|$)", value)
        )
        if raw_prompt_leak or suspicious_years or no_complete_sentence:
            return (
                ConversationalRagOrchestrator._grounded_limitation(query, chunks),
                ("generation_output_replaced_with_grounded_limit",),
                True,
            )

        sentences = re.split(r"(?<=[.!?。！？])\s+", value)
        forbidden_fragments = (
            "검색 근거에 직접 확인되는 부분은 답하세요",
            "사실 확인 과정·판정표·초안",
            "현재 사용자 메시지에 직접 자연스럽게 답하세요",
        )
        kept = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
            and "�" not in sentence
            and not any(fragment in sentence for fragment in forbidden_fragments)
        ][:3]
        stabilized = " ".join(kept).strip()
        if not stabilized:
            return (
                ConversationalRagOrchestrator._grounded_limitation(query, chunks),
                ("generation_output_replaced_with_grounded_limit",),
                True,
            )
        warnings: tuple[str, ...] = ()
        if stabilized != answer.strip() or echo_sentences:
            warnings = ("generation_output_stabilized",)
        return stabilized, warnings, False

    @staticmethod
    def _is_question_echo(answer_sentence: str, query: str) -> bool:
        normalize = lambda item: re.sub(r"[\W_]+", "", item).casefold()
        answer_value = normalize(answer_sentence)
        query_value = normalize(query)
        if len(query_value) < 6 or not answer_value:
            return False
        if query_value in answer_value:
            return True
        return SequenceMatcher(
            None, answer_value, query_value, autojunk=False
        ).ratio() >= 0.92

    @staticmethod
    def _grounded_limitation(query: str, chunks: list[RankedChunk]) -> str:
        subject = (
            chunks[0].chunk.title.split(" - ", 1)[0].strip()
            if chunks else "질문하신 주제"
        )
        if re.search(r"인물|사람|누구", query):
            detail = "관련된 특정 인물"
        elif re.search(r"언제|시기|연도|날짜|건립|설립|개통|준공", query):
            detail = "요청한 시점"
        elif re.search(r"장소|어디", query):
            detail = "요청한 장소"
        elif re.search(r"결과|영향|원인|이유", query):
            detail = "요청한 인과·결과"
        else:
            detail = "요청한 세부 내용"
        return f"선택된 검색 근거에서는 {subject}의 {detail}을 직접 확인하기 어렵습니다."

    @staticmethod
    def _source_sufficiency(chunks: list[RankedChunk]) -> SourceSufficiency:
        if not chunks:
            return SourceSufficiency.INSUFFICIENT
        if any(
            item.chunk.payload.get("source_conflict") is True
            or item.chunk.payload.get("fact_status") == "conflicting"
            for item in chunks
        ):
            return SourceSufficiency.CONFLICTING
        return SourceSufficiency.SUFFICIENT

    def _guard_answer(
        self, answer: str, *, output_domain: OutputDomain, situation,
        stage, locale: str, citations: tuple[dict[str, object], ...] = (),
    ) -> tuple[str, tuple[str, ...]]:
        violations = self.response_renderer.guard.validate(
            answer, domain=output_domain, situation=situation,
            stage=stage, locale=locale, citations=citations,
        )
        if not violations:
            rendered = self.response_renderer.render(
                answer, domain=output_domain, situation=situation,
                stage=stage, locale=locale, citations=citations,
            )
            return rendered.text, ()
        if output_domain == OutputDomain.SYSTEM_UI:
            fallback = "현재 확인된 기능 정보만으로는 안내하기 어렵습니다. 공식 안내 또는 현장 직원에게 확인해 주세요."
        elif locale.lower() == "zh-cn":
            fallback = "这段角色回复尚未完成审核，暂时无法显示。"
        else:
            fallback = "확인된 근거와 말투를 다시 점검한 뒤 답할게. 지금은 추측해서 말하지 않을게."
        return fallback, tuple(f"style_guard:{item.code}" for item in violations)

    @staticmethod
    def _pending_zh_text(domain: OutputDomain) -> str:
        if domain == OutputDomain.SYSTEM_UI:
            return "该功能信息尚未完成中文审核，请以现场官方说明为准。"
        return "这段记录鸟角色回复尚未完成审核，暂时不作为正式台词。"

    @staticmethod
    def _insufficient_guidance(
        query: str, domain: OutputDomain, locale: str, *, active_place: str = ""
    ) -> tuple[str, tuple[str, ...]]:
        if locale.lower() == "zh-cn":
            return ConversationalRagOrchestrator._pending_zh_text(domain), ()

        subject = active_place or "질문하신 역사 주제"
        asks_people = bool(re.search(r"인물|사람|누구", query))
        asks_date = bool(re.search(r"언제|건립|준공|만들|세워|생긴", query))
        if asks_people:
            limitation = f"현재 확보된 자료에서는 {subject} 관련 인물을 확정할 근거를 확인하지 못했습니다."
            suggestions = (f"{subject} 관련 기록에서 확인되는 사건을 알려줘.",)
        elif asks_date:
            limitation = f"현재 확보된 자료만으로는 {subject}의 건립·형성 시점을 정확히 확인하지 못했습니다."
            suggestions = (f"{subject} 관련 기록에서 확인되는 사건을 알려줘.",)
        elif active_place:
            limitation = f"현재 확보된 자료만으로는 {subject}에 관한 질문의 세부 내용을 확인하지 못했습니다."
            suggestions = (f"{subject}의 역사적 역할에 관해 확인 가능한 내용을 알려줘.",)
        else:
            limitation = "현재 확보된 역사 자료에서는 질문하신 내용을 확인하지 못했습니다."
            suggestions = ("목포의 철도와 항만에 관해 확인 가능한 기록을 알려줘.",)

        examples = "\n".join(f"- {item}" for item in suggestions)
        if domain == OutputDomain.CHARACTER_DIALOGUE:
            answer = (
                f"{limitation.replace('습니다.', '어.')} 확인되지 않은 내용은 덧붙이지 않을게. "
                f"원한다면 이렇게 이어갈 수 있어.\n\n{examples}"
            )
        else:
            answer = (
                f"{limitation} 확인되지 않은 내용은 추측하지 않습니다. "
                f"원하시면 다음 기록으로 이어가겠습니다.\n\n{examples}"
            )
        return answer, suggestions
