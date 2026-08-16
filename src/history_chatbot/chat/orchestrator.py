"""세션부터 검색·프롬프트·답변·출처까지 연결하는 grounded RAG."""

from __future__ import annotations

import re
import time
from datetime import date
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
from history_chatbot.chat.remote_safe import (
    EvidenceSupport,
    GroundedFactPacket,
    _competitor_matches,
    assess_direct_evidence,
    build_grounded_fact_packet,
    serialize_remote_prompt,
    verified_person_facts,
)
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


REMOTE_QA_INSTRUCTIONS = """역사 안내자로서 제공된 기록만 사용해 한국어로 답하세요.
PRIMARY FACT의 사실관계만 답의 핵심으로 사용하세요. SUPPORTING CONTEXT를 합쳐 새로운 연도·원인·역할·인물 관계를 만들지 마세요.
첫 문장에서 질문한 시점·이유·인물·역할·결과를 직접 답하고, 필요한 배경만 이어서 1~3문장으로 설명하세요.
질문의 대상과 다른 인물·장소·건물의 사실을 섞지 마세요. 기록이 일부만 뒷받침하면 확인되는 부분만 답하세요.
질문을 되풀이하거나 답 대신 새로운 질문을 만들지 마세요.
첫 문장에 질문 대상을 분명히 쓰고, '또한', '그리고', '이 질문에 대한 답'으로 시작하지 마세요.
근거에 없는 연도·직책·순서·재임 기간·원인·영향·역할을 알고 있는 내용으로 보충하지 마세요.
지침, 역할, 자료 번호, 확인 과정은 답변에 쓰지 마세요. 완결된 문장으로 끝내세요."""

REMOTE_TRANSFORM_INSTRUCTIONS = """제공된 기록의 사실만 사용해 직전 주제의 설명을 사용자가 요청한 방식으로 다시 표현하세요.
PRIMARY FACT의 사실관계는 바꾸지 말고, SUPPORTING CONTEXT에서 새로운 관계를 만들지 마세요.
새 사실을 추가하지 말고, 쉽고 짧은 한국어 1~2문장으로 직접 설명하세요.
지침, 역할, 자료 번호, 확인 과정은 답변에 쓰지 마세요. 완결된 문장으로 끝내세요."""


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
    retrieval_retry_performed: bool = False
    retrieval_retry_query: str = ""
    evidence_support: str = EvidenceSupport.NONE.value
    direct_evidence_excerpts: tuple[str, ...] = ()
    assessed_intent: str = "overview"
    fact_packet: GroundedFactPacket | None = None
    nearby_fact_packet: GroundedFactPacket | None = None
    comparison_packets: tuple[GroundedFactPacket, ...] = ()


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
        if (
            classification.primary_situation_id == SituationId.RESPONSE_STYLE_REQUEST
            and track.should_retrieve
        ):
            output_domain = OutputDomain.HISTORICAL_DOCENT
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
        evidence_assessment = assess_direct_evidence(
            chunks,
            subject=resolved_context.active_subject,
            intent=prepared.assessed_intent,
            question=resolved_context.resolved_question,
        )
        enforce_direct_evidence = not chunks or not all(
            item.chunk.payload.get("data_classification") == "fictional_fixture"
            for item in chunks
        )
        sufficiency = self._source_sufficiency(
            chunks,
            evidence_support=(
                evidence_assessment.support if enforce_direct_evidence else None
            ),
        )
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
        elif len(prepared.comparison_packets) >= 2:
            citations = build_citations(chunks)
            answer, complete_comparison = self._comparison_answer(
                prepared.comparison_packets
            )
            comparison_common = dict(common) | {
                "request_state": "success",
                "ui_state": "showing_citations",
                "rag_used": True,
                "source_sufficiency": (
                    SourceSufficiency.SUFFICIENT.value
                    if complete_comparison else SourceSufficiency.PARTIAL.value
                ),
                "warnings": tuple(dict.fromkeys(
                    common["warnings"] + (("comparison_fact_packets",)
                    if complete_comparison else ("comparison_partial_packet",))
                )),
            }
            response = ChatResponse(
                answer, "ok" if complete_comparison else "partial_evidence",
                citations, len(chunks), session.session_id, locale, PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
                },
                citations=tuple(asdict(item) for item in citations),
                evidence=tuple(item.chunk.text for item in chunks),
                grounded=True,
                retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                latency_ms=round((time.perf_counter() - started) * 1000),
                **comparison_common,
                **self._provisional_metadata(chunks),
            )
        elif enforce_direct_evidence and (
            not (prepared.fact_packet and prepared.fact_packet.facts)
            or prepared.fact_packet.conflicting
        ):
            citations = build_citations(chunks)
            nearby = (
                None if prepared.assessed_intent in {"people", "role"}
                else prepared.nearby_fact_packet
            )
            answer = self._conflicting_evidence_guidance(
                resolved_context.active_subject, prepared.assessed_intent
            ) if prepared.fact_packet and prepared.fact_packet.conflicting else self._nearby_supported_answer(
                resolved_context.active_subject,
                prepared.assessed_intent,
                nearby,
            ) if nearby and nearby.facts else (
                self._explain_evidence_boundary(
                    resolved_context.active_subject, prepared.assessed_intent
                )
                if resolved_context.request_kind
                == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
                else self._related_evidence_guidance(
                    resolved_context.active_subject, prepared.assessed_intent
                )
            )
            related_common = dict(common) | {
                "request_state": "success",
                "ui_state": "showing_citations",
                "rag_used": True,
                "source_sufficiency": SourceSufficiency.PARTIAL.value,
                "warnings": tuple(dict.fromkeys(
                    common["warnings"] + (("nearby_supported_answer",)
                    if nearby and nearby.facts else ("direct_evidence_missing",))
                )),
            }
            response = ChatResponse(
                answer,
                "partial_evidence" if prepared.fact_packet and prepared.fact_packet.conflicting
                else "ok" if nearby and nearby.facts else "partial_evidence",
                citations,
                len(chunks),
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
                },
                citations=tuple(asdict(item) for item in citations),
                evidence=tuple(item.chunk.text for item in chunks),
                grounded=True,
                retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                retrieved_source_ids=tuple(dict.fromkeys(
                    str(item.chunk.payload.get("source_id", item.chunk.document_id))
                    for item in chunks
                )),
                latency_ms=round((time.perf_counter() - started) * 1000),
                **related_common,
                **self._provisional_metadata(chunks),
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
                request_kind=resolved_context.request_kind,
                active_subject=resolved_context.active_subject,
                current_intent=resolved_context.current_intent,
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
                    self._stabilize_grounded_answer(
                        answer,
                        query=resolved_context.resolved_question,
                        chunks=chunks,
                        fact_packet=None if is_fixture else prepared.fact_packet,
                    )
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
                answer, style_warnings = self._guard_grounded_answer(
                    answer, output_domain=output_domain,
                    situation=classification.primary_situation_id,
                    stage=stage, locale=locale,
                    citations=tuple(asdict(item) for item in citations),
                    fact_packet=None if is_fixture else prepared.fact_packet,
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
                if prepared.fact_packet.facts:
                    citations = build_citations(chunks)
                    answer = self._extractive_fact_answer(prepared.fact_packet)
                    fallback_common = dict(common)
                    fallback_common.update(
                        request_state="success",
                        ui_state="showing_citations",
                        rag_used=True,
                        source_sufficiency=sufficiency.value,
                        warnings=tuple(dict.fromkeys(
                            common["warnings"] + ("remote_generation_extractive_fallback",)
                        )),
                    )
                    response = ChatResponse(
                        answer,
                        "ok",
                        citations,
                        len(chunks),
                        session.session_id,
                        locale,
                        PROMPT_VERSION,
                        context_metadata=self._evidence_decision_metadata(prepared, chunks),
                        citations=tuple(asdict(item) for item in citations),
                        evidence=tuple(item.chunk.text for item in chunks),
                        grounded=True,
                        retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                        retrieved_source_ids=tuple(dict.fromkeys(
                            str(item.chunk.payload.get("source_id", item.chunk.document_id))
                            for item in chunks
                        )),
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        **fallback_common,
                        **self._provisional_metadata(chunks),
                    )
                else:
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
                    active_subject=resolved_context.active_subject,
                    active_person=resolved_context.active_person,
                    answered_intent=resolved_context.current_intent,
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
                active_subject=resolved_context.active_subject,
                active_person=resolved_context.active_person,
                stable_evidence_anchor=resolved_context.active_subject,
                last_answered_intent=resolved_context.current_intent,
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
        evidence_assessment = assess_direct_evidence(
            chunks,
            subject=resolved_context.active_subject,
            intent=prepared.assessed_intent,
            question=resolved_context.resolved_question,
        )
        enforce_direct_evidence = not all(
            item.chunk.payload.get("data_classification") == "fictional_fixture"
            for item in chunks
        )
        if len(prepared.comparison_packets) >= 2:
            answer, complete_comparison = self._comparison_answer(
                prepared.comparison_packets
            )
            citations = build_citations(chunks)
            response = ChatResponse(
                answer, "ok" if complete_comparison else "partial_evidence",
                citations, len(chunks), session.session_id, locale, PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
                },
                warnings=("comparison_fact_packets",)
                if complete_comparison else ("comparison_partial_packet",),
                request_state="success", ui_state="showing_citations", rag_used=True,
                source_sufficiency=(
                    SourceSufficiency.SUFFICIENT.value
                    if complete_comparison else SourceSufficiency.PARTIAL.value
                ),
                evidence=tuple(item.chunk.text for item in chunks), grounded=True,
                retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                **self._provisional_metadata(chunks),
            )
            self.sessions.add_turn(session.session_id, query, answer)
            yield StreamEvent("completed", response.to_dict())
            return
        if enforce_direct_evidence and (
            not (prepared.fact_packet and prepared.fact_packet.facts)
            or prepared.fact_packet.conflicting
        ):
            nearby = (
                None if prepared.assessed_intent in {"people", "role"}
                else prepared.nearby_fact_packet
            )
            answer = self._conflicting_evidence_guidance(
                resolved_context.active_subject, prepared.assessed_intent
            ) if prepared.fact_packet and prepared.fact_packet.conflicting else self._nearby_supported_answer(
                resolved_context.active_subject,
                prepared.assessed_intent,
                nearby,
            ) if nearby and nearby.facts else (
                self._explain_evidence_boundary(
                    resolved_context.active_subject, prepared.assessed_intent
                )
                if resolved_context.request_kind
                == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
                else self._related_evidence_guidance(
                    resolved_context.active_subject, prepared.assessed_intent
                )
            )
            citations = build_citations(chunks)
            response = ChatResponse(
                answer,
                "partial_evidence" if prepared.fact_packet and prepared.fact_packet.conflicting
                else "ok" if nearby and nearby.facts else "partial_evidence",
                citations,
                len(chunks),
                session.session_id,
                locale,
                PROMPT_VERSION,
                context_metadata={
                    "trimmed_evidence": budget.trimmed_evidence,
                    "trimmed_conversation": budget.trimmed_conversation,
                    **self._evidence_decision_metadata(prepared, chunks),
                },
                warnings=("nearby_supported_answer",)
                if nearby and nearby.facts else ("direct_evidence_missing",),
                request_state="success",
                ui_state="showing_citations",
                rag_used=True,
                source_sufficiency=SourceSufficiency.PARTIAL.value,
                citations=tuple(asdict(item) for item in citations),
                evidence=tuple(item.chunk.text for item in chunks),
                grounded=True,
                retrieved_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                **self._provisional_metadata(chunks),
            )
            self.sessions.add_turn(session.session_id, query, answer)
            if prepared.retrieval_performed:
                self.sessions.add_evidence_turn(
                    session.session_id,
                    user=query,
                    active_place=resolved_context.active_place,
                    active_topic=resolved_context.active_topic,
                    chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
                    active_subject=resolved_context.active_subject,
                    active_person=resolved_context.active_person,
                    answered_intent=resolved_context.current_intent,
                )
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
            request_kind=resolved_context.request_kind,
            active_subject=resolved_context.active_subject,
            current_intent=resolved_context.current_intent,
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
                        answer,
                        query=resolved_context.resolved_question,
                        chunks=chunks,
                        fact_packet=None if is_fixture else prepared.fact_packet,
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
                        active_subject=resolved_context.active_subject,
                        active_person=resolved_context.active_person,
                        answered_intent=resolved_context.current_intent,
                    )
                self.sessions.update_context(
                    session.session_id,
                    recent_people=self._recent_people(
                        chunks, resolved_context.recent_people
                    ),
                    active_subject=resolved_context.active_subject,
                    active_person=resolved_context.active_person,
                    stable_evidence_anchor=resolved_context.active_subject,
                    last_answered_intent=resolved_context.current_intent,
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
            active_subject=resolved.active_subject,
            active_person=resolved.active_person,
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
        retry_performed = False
        retry_query = ""
        if retrieval_performed:
            primary_results = self._subject_aware_search(resolved.search_query, top_k)
            if not primary_results:
                if resolved.request_kind == ConversationRequestKind.FACTUAL_FOLLOWUP:
                    intent_term = {
                        "time": "창립 시기",
                        "people": "관련 인물",
                        "place": "관련 장소",
                        "cause": "배경 원인",
                        "result": "결과 영향",
                        "role": "역할",
                        "current": "현재",
                    }.get(resolved.current_intent, "")
                    retry_query = " ".join(
                        value for value in (resolved.active_subject, intent_term) if value
                    )
                if not retry_query:
                    retry_query = " ".join(explicit_subject_words(resolved.search_query))
                if retry_query and retry_query != resolved.search_query.strip():
                    retry_performed = True
                    primary_results = self._subject_aware_search(retry_query, top_k)
            retrieved = self._select(
                self._place_aware_results(primary_results, place_filter), top_k
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

        if (
            resolved.request_kind == ConversationRequestKind.INDEPENDENT
            and self._unsupported_query_constraints(
            resolved.resolved_question,
            chunks,
            subject=resolved.active_subject,
            )
        ):
            verified_subject = self._verified_retrieval_subject(
                resolved.active_subject, chunks
            )
            # A topically related hit must not erase an unsupported qualifier
            # from the user's claim (for example, a fictitious location or
            # date attached to a real station).  Treat the whole claim as
            # unverified instead of answering the nearby real-world subject.
            chunks = []
            remembered = []
            partial_evidence_used = False
            if verified_subject:
                # Preserve only the real subject for later follow-ups.  The
                # rejected premise's date, event, and person never become
                # conversation facts or generation evidence.
                self.sessions.update_context(
                    session.session_id,
                    active_subject=verified_subject,
                    active_topic=verified_subject,
                    stable_evidence_anchor=verified_subject,
                    active_person="",
                    recent_period="",
                    recent_event="",
                )

        self._assert_mode_boundary(chunks)
        remembered_ids = {item.chunk.chunk_id for item in remembered}
        memory_evidence_used = any(
            item.chunk.chunk_id in remembered_ids for item in chunks
        )
        assessed_intent = (
            session.last_answered_intent or "overview"
            if resolved.request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
            else resolved.current_intent
        )
        actor_results: list[RankedChunk] = []
        if (
            assessed_intent == "people"
            and resolved.active_subject
            and retrieval_performed
            and re.search(r"\d+(?:\.\d+)+(?:[가-힣A-Za-z]+)", resolved.active_subject)
        ):
            actor_results = self._subject_aware_search(
                f"{resolved.active_subject} 주도하였던 인물", top_k
            )
            chunks = self._select(
                self._place_aware_results([*actor_results, *chunks], place_filter),
                top_k,
            )
        assessment = assess_direct_evidence(
            chunks,
            subject=resolved.active_subject,
            intent=assessed_intent,
            question=resolved.resolved_question,
        )
        fact_packet = build_grounded_fact_packet(
            chunks,
            subject=resolved.active_subject,
            intent=assessed_intent,
            question=resolved.resolved_question,
        )
        if actor_results:
            actor_packet = build_grounded_fact_packet(
                actor_results,
                subject=resolved.active_subject,
                intent="people",
                question=resolved.resolved_question,
            )
            if verified_person_facts(actor_packet):
                fact_packet = actor_packet
                chunks = self._select(actor_results, top_k)
                assessment = assess_direct_evidence(
                    chunks,
                    subject=resolved.active_subject,
                    intent="people",
                    question=resolved.resolved_question,
                )
        if (
            not fact_packet.facts
            and assessed_intent == "people"
            and re.search(r"인물(?:이나|과|또는)\s*장소|사람(?:이나|과|또는)\s*장소", query)
        ):
            place_packet = build_grounded_fact_packet(
                chunks,
                subject=resolved.active_subject,
                intent="place",
                question=resolved.resolved_question,
            )
            if place_packet.facts:
                assessed_intent = "place"
                assessment = assess_direct_evidence(
                    chunks,
                    subject=resolved.active_subject,
                    intent="place",
                    question=resolved.resolved_question,
                )
                fact_packet = place_packet
        if (
            not fact_packet.facts
            and assessed_intent == "role"
            and resolved.active_subject
        ):
            facet_term = "활동" if re.search(
                r"어떤\s*활동|무슨\s*일|무엇을\s*했|뭘\s*했",
                resolved.resolved_question,
            ) else "역할"
            facet_query = f"{resolved.active_subject} {facet_term}"
            facet_results = self._select(
                self._place_aware_results(
                    self._subject_aware_search(facet_query, top_k), place_filter
                ),
                top_k,
            )
            facet_packet = build_grounded_fact_packet(
                facet_results,
                subject=resolved.active_subject,
                intent=assessed_intent,
                question=resolved.resolved_question,
            )
            if facet_packet.facts:
                chunks = facet_results
                remembered = []
                memory_evidence_used = False
                partial_evidence_used = False
                retry_performed = True
                retry_query = facet_query
                assessment = assess_direct_evidence(
                    chunks,
                    subject=resolved.active_subject,
                    intent=assessed_intent,
                    question=resolved.resolved_question,
                )
                fact_packet = facet_packet
        nearby_fact_packet = None
        if not fact_packet.facts:
            for nearby_intent in (
                "overview", "time", "people", "place", "role", "current", "result"
            ):
                if nearby_intent == assessed_intent:
                    continue
                candidate = build_grounded_fact_packet(
                    chunks,
                    subject=resolved.active_subject,
                    intent=nearby_intent,
                    question=resolved.resolved_question,
                )
                if candidate.facts:
                    nearby_fact_packet = candidate
                    break
        comparison_packets: tuple[GroundedFactPacket, ...] = ()
        comparison_uncertainty = bool(re.search(
            r"확실하지\s*않은\s*부분|불확실한\s*부분", query
        ))
        if re.search(r"차이|달라|비교|구분|섞지|각각", query) or (
            comparison_uncertainty and len(resolved.recent_entities) >= 2
        ):
            subjects = tuple(dict.fromkeys(explicit_subject_words(resolved.search_query)))
            if comparison_uncertainty:
                subjects = tuple(dict.fromkeys(resolved.recent_entities))[:2]
            if re.search(r"둘\s*중|두\s*사건", query):
                for prior_turn in reversed(session.evidence_turns):
                    prior_subjects = explicit_subject_words(prior_turn.user)
                    if len(prior_subjects) >= 2:
                        subjects = prior_subjects[:2]
                        break
            comparing_subjects = bool(re.search(
                r"둘\s*중.*(?:사람|인물).*구분", query
            ))
            requested_facets = (
                ("overview",) if comparing_subjects else tuple(
                    facet for facet, pattern in (
                        ("time", r"날짜|시기|언제|연도|개통"),
                        ("people", r"인물|사람|누구|누가"),
                    )
                    if re.search(pattern, query)
                ) or ("overview",)
            )
            packets_list: list[GroundedFactPacket] = []
            comparison_chunks: list[RankedChunk] = []
            for value in subjects[:2]:
                for facet in requested_facets:
                    facet_question = (
                        "관련 인물은?" if facet == "people" else
                        "시기는 언제야?" if facet == "time" else query
                    )
                    facet_term = (
                        "관련 인물" if facet == "people" else
                        "시기 날짜" if facet == "time" else ""
                    )
                    facet_results = self._select(
                        self._place_aware_results(
                            self._subject_aware_search(
                                f"{value} {facet_term}".strip(), top_k
                            ),
                            place_filter,
                        ),
                        top_k,
                    )
                    comparison_chunks.extend(facet_results)
                    packets_list.append(build_grounded_fact_packet(
                        facet_results,
                        subject=value,
                        intent=facet,
                        question=facet_question,
                    ))
            packets = tuple(packets_list)
            if len(subjects) == 2 and len(packets) >= 2:
                comparison_packets = packets
                unique_chunks: list[RankedChunk] = []
                seen_chunk_ids: set[str] = set()
                for item in (*chunks, *comparison_chunks):
                    if item.chunk.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(item.chunk.chunk_id)
                    unique_chunks.append(item)
                chunks = unique_chunks
        return PreparedTurnEvidence(
            resolved_context=resolved,
            chunks=tuple(chunks),
            needs_new_evidence=needs_new_evidence,
            retrieval_performed=retrieval_performed,
            memory_evidence_used=memory_evidence_used,
            partial_evidence_used=partial_evidence_used,
            detail_evidence_sufficient=detail_sufficient,
            requested_detail_supported=self._supports_requested_detail(query, chunks),
            retrieval_retry_performed=retry_performed,
            retrieval_retry_query=retry_query if retry_performed else "",
            evidence_support=assessment.support.value,
            direct_evidence_excerpts=assessment.excerpts,
            assessed_intent=assessed_intent,
            fact_packet=fact_packet,
            nearby_fact_packet=nearby_fact_packet,
            comparison_packets=comparison_packets,
        )

    def _subject_aware_search(
        self, search_query: str, top_k: int
    ) -> list[RankedChunk]:
        """Preserve one result per explicit subject in comparison/relationship questions."""

        subjects = explicit_subject_words(search_query)
        is_multi_subject = bool(re.search(
            r"구분|비교|차이|둘(?:을|은|이)|같은\s*단체|관련|관계|함께|에서|와|과",
            search_query,
        ))
        if not is_multi_subject or not 2 <= len(subjects) <= 3:
            return self.retrieval.search(search_query)
        batches = [self.retrieval.search(subject) for subject in subjects]
        primary = [batch[0] for batch in batches if batch]
        remainder = [item for batch in batches for item in batch[1:]]
        return self._select([*primary, *remainder], top_k)

    @staticmethod
    def _unsupported_query_constraints(
        query: str, chunks: list[RankedChunk], *, subject: str
    ) -> bool:
        """Reject retrieved evidence that only matches part of an asserted claim."""

        if not chunks:
            return False
        evidence = " ".join(
            f"{item.chunk.title} {item.chunk.text}" for item in chunks
        )
        compact_evidence = re.sub(r"\s+", "", evidence).casefold()
        if re.search(r"가상(?:의)?\s*(?:인물|사건|장소)|자료에\s*없|존재하지\s*않", query):
            return True

        verification = bool(re.search(
            r"전제|(?:말이\s*)?맞(?:아|지|는지)|(?:났|했|였|됐|있)지\?",
            query,
        ))
        claimed_numbers = set(re.findall(r"(?<!\d)\d{2,4}(?!\d)", query))
        if verification and any(number not in evidence for number in claimed_numbers):
            return True
        claimed_floors = set(re.findall(r"\d+\s*층", query))
        evidence_floors = set(re.findall(r"\d+\s*층", evidence))
        if claimed_floors - evidence_floors:
            return True
        future_years = {
            int(value) for value in re.findall(r"(?<!\d)(?:20|21)\d{2}(?!\d)", query)
            if int(value) > date.today().year
        }
        if future_years and not any(str(value) in evidence for value in future_years):
            return True
        stop_terms = {
            "자료", "근거", "기록", "확인", "질문", "전제", "사실", "내용",
            "말", "답", "알려", "맞아", "맞지", "맞는지", "했지", "있지",
            "년에",
        }

        def terms(value: str) -> list[str]:
            found: list[str] = []
            for raw in re.findall(r"[가-힣A-Za-z]{2,}", value):
                token = re.sub(
                    r"(?:이라는|였다는|했다는|한다는|에서만|에게서|으로|에서|까지|부터|"
                    r"에는|에게|라는|다고|이고|이며|했지|됐지|있지|은|는|이|가|을|를|의)$",
                    "", raw,
                )
                if len(token) >= 2 and token not in stop_terms:
                    found.append(token.casefold())
            return found

        if verification:
            claim_terms = terms(query)
            return any(term not in compact_evidence for term in claim_terms)

        compact_subject = re.sub(r"\s+", "", subject).casefold()
        compact_query = re.sub(r"\s+", "", query).casefold()
        subject_index = compact_query.find(compact_subject) if compact_subject else -1
        if subject_index > 0 and re.search(r"지점|분관|지하|지부|별관", query):
            prefix = compact_query[:subject_index]
            missing_modifiers = [
                term for term in terms(prefix) if term not in compact_evidence
            ]
            if missing_modifiers:
                return True
        return False

    @staticmethod
    def _verified_retrieval_subject(
        subject: str, chunks: list[RankedChunk]
    ) -> str:
        """Keep a real subject anchor without accepting the surrounding claim."""

        compact_subject = re.sub(r"\s+", "", subject).casefold()
        if len(compact_subject) < 2:
            return ""
        for item in chunks:
            title_subject = item.chunk.title.split(" - ", 1)[0]
            compact_title = re.sub(r"\s+", "", title_subject).casefold()
            if compact_subject == compact_title or compact_subject in compact_title:
                return subject
        return ""

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
            "evidence_support": prepared.evidence_support,
            "assessed_intent": prepared.assessed_intent,
            "direct_evidence_count": len(prepared.direct_evidence_excerpts),
            "grounded_fact_count": (
                len(prepared.fact_packet.facts) if prepared.fact_packet else 0
            ) + (
                len(prepared.nearby_fact_packet.facts)
                if prepared.nearby_fact_packet else 0
            ),
            "grounded_fact_source_ids": tuple(dict.fromkeys(
                fact.source_id for fact in prepared.fact_packet.facts
            )) if prepared.fact_packet else (),
            "nearby_grounded_fact_count": len(prepared.nearby_fact_packet.facts)
            if prepared.nearby_fact_packet else 0,
            "selected_evidence_ids": tuple(
                item.chunk.chunk_id
                for item in (
                    selected_chunks
                    if selected_chunks is not None else prepared.chunks
                )
            ),
            "active_place": resolved.active_place,
            "active_subject": resolved.active_subject,
            "active_person": resolved.active_person,
            "stable_evidence_anchor": resolved.stable_evidence_anchor,
            "current_intent": resolved.current_intent,
            "retrieval_retry_performed": prepared.retrieval_retry_performed,
            "retrieval_retry_query": prepared.retrieval_retry_query,
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
            alias = (
                "목포근대역사관1관"
                if "일본영사관" in compact_place
                else "목포근대역사관2관"
                if "동양척식주식회사" in compact_place
                else ""
            )
            exact = [
                item for item in results
                if compact_place in re.sub(r"\s+", "", f"{item.chunk.title} {item.chunk.text}")
                or (
                    alias
                    and alias in re.sub(r"\s+", "", f"{item.chunk.title} {item.chunk.text}")
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
        request_kind=ConversationRequestKind.INDEPENDENT,
        active_subject="", current_intent="overview",
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
            is_transform = request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
            safe = serialize_remote_prompt(
                system_prompt=(
                    REMOTE_TRANSFORM_INSTRUCTIONS if is_transform
                    else REMOTE_QA_INSTRUCTIONS
                ),
                user_query=query.split("\n[", 1)[0].strip(),
                chunks=chunks,
                history=(
                    () if is_transform else
                    tuple((turn.user, turn.assistant) for turn in session.turns)
                ),
                policy=remote_config.remote_prompt_policy(),
                question_subject=active_subject,
                question_intent=current_intent,
                transform=is_transform,
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
            normalized = ConversationalRagOrchestrator._trim_incomplete_tail(normalized)
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
        incomplete_at = ConversationalRagOrchestrator._unclosed_delimiter_position(complete)
        if incomplete_at is not None:
            balanced = ConversationalRagOrchestrator._complete_sentence_prefix(
                complete[:incomplete_at]
            )
            if balanced:
                complete = balanced
        complete = ConversationalRagOrchestrator._trim_incomplete_tail(complete)
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
        positions = [stack[0][1]] if stack else []
        for opening, closing in (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")):
            if text.count(opening) > text.count(closing):
                positions.append(text.rfind(opening))
        if text.count('"') % 2:
            positions.append(text.rfind('"'))
        return min(positions) if positions else None

    @staticmethod
    def _trim_incomplete_tail(text: str) -> str:
        normalized = text.strip()
        tail = re.search(
            r"(?:^|\s)(?:\d+[.)]|[-*•])\s*$|[:;,]\s*$|"
            r"(?:그리고|그러나|하지만|따라서|또는|및|때문에)\s*$",
            normalized,
        )
        if tail is None:
            return normalized
        prefix = normalized[:tail.start()].rstrip()
        complete = ConversationalRagOrchestrator._complete_sentence_prefix(prefix)
        return complete or prefix

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
        answer: str, *, query: str, chunks: list[RankedChunk],
        fact_packet: GroundedFactPacket | None = None,
    ) -> tuple[str, tuple[str, ...], bool]:
        """Bound weak-model output and replace prompt leakage with a scoped limitation."""

        if (
            fact_packet
            and fact_packet.intent == "people"
            and re.search(r"관련(?:된)?\s*인물|인물은|주요\s*인물|누구", query)
            and verified_person_facts(fact_packet)
        ):
            return (
                ConversationalRagOrchestrator._extractive_fact_answer(fact_packet),
                ("verified_people_extractive",),
                True,
            )

        value = answer.strip()
        value = re.sub(
            r"^\s*(?:\[(?:답변|대화\s*문맥\s*해석|최종\s*답변)\]\s*)+",
            "",
            value,
        )
        evidence_text = " ".join(item.chunk.text for item in chunks)
        supported_numeric = set(
            re.findall(
                r"(?:18|19|20)\d{2}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?|"
                r"제\s*\d+\s*대|\d+\s*년간",
                evidence_text,
            )
        )
        numeric_claim = re.compile(
            r"(?:18|19|20)\d{2}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?|"
            r"제\s*\d+\s*대|\d+\s*년간"
        )
        numeric_sentences = re.split(r"(?<=[.!?。！？])\s+", value)
        kept_numeric = [
            sentence for sentence in numeric_sentences
            if not chunks or all(
                marker in supported_numeric or marker in evidence_text
                for marker in numeric_claim.findall(sentence)
            )
        ]
        unsupported_numeric_removed = len(kept_numeric) != len(numeric_sentences)
        if kept_numeric:
            value = " ".join(kept_numeric).strip()
        elif unsupported_numeric_removed:
            return (
                ConversationalRagOrchestrator._extractive_fact_answer(fact_packet)
                if fact_packet and fact_packet.facts
                else ConversationalRagOrchestrator._grounded_limitation(query, chunks),
                (
                    "unsupported_numeric_claim_removed",
                    "unsafe_generation_replaced_extractive",
                ) if fact_packet and fact_packet.facts else (
                    "unsupported_numeric_claim_removed",
                ),
                True,
            )
        incomplete_at = ConversationalRagOrchestrator._unclosed_delimiter_position(value)
        if incomplete_at is not None:
            completed = ConversationalRagOrchestrator._complete_sentence_prefix(
                value[:incomplete_at]
            )
            value = completed or value[:incomplete_at].rstrip(" ,;:-")
        value = re.sub(r"^\s*\[(?:ANSWER|답변)\]\s*", "", value, flags=re.IGNORECASE)
        leak_pattern = re.compile(
                r"\[(?:검색\s*근거|자료\d+|PRIMARY\s*FACT|SUPPORTING\s*CONTEXT|사용자(?:\s*질문)?|USER)\]"
                r"|(?:^|\n)\s*(?:사용자|USER)\s*[:：]"
                r"|(?:이러한|위)?\s*(?:역사적\s*)?(?:사실|근거)를\s*(?:확인(?:한\s*후에)?|바탕으로)"
                r"|사실\s*확인을\s*위해선?\s*역사적\s*사실의\s*근거를\s*확인"
                r"|지원되지\s*않은\s*(?:내용|사실)"
                r"|이번\s*대화는\s*후속\s*질문\s*해석에만\s*사용"
                r"|사실\s*확인\s*과정|필요한\s*정보를\s*파악"
                r"|지원되지\s*않은\s*역사적\s*주장|불필요한\s*제한\s*문구"
                r"|사용자가\s*자세한\s*설명을\s*요청하지\s*않았다면"
                r"|(?:이러한|위)\s*근거를\s*토대로"
                r"|(?:^|\b)(?:system|assistant|user)\s*(?:message|prompt|instruction)?\s*[:：]"
                r"|(?:^|\n)\s*(?:대상|의도|질문)\s*[:：]"
                r"|provided\s+(?:context|evidence)|according\s+to\s+(?:the\s+)?context"
                r"|(?:follow|ignore|do\s+not)\s+(?:the\s+)?(?:above\s+)?instructions?"
                r"|(?:시스템|모델)\s*(?:지침|프롬프트|역할)|제공된\s*(?:문맥|컨텍스트)"
                r"|답변에\s*쓰지\s*마세요|질문에\s*먼저\s*자연스럽게\s*답",
                re.IGNORECASE,
        )
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", value)
        leaked_sentences = {
            index for index, sentence in enumerate(sentences)
            if leak_pattern.search(sentence)
        }
        raw_prompt_leak = bool(leaked_sentences)
        if leaked_sentences:
            recovered = " ".join(
                sentence.strip() for index, sentence in enumerate(sentences)
                if index not in leaked_sentences and sentence.strip()
            )
            if recovered:
                value = recovered
                raw_prompt_leak = False
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
        final_incomplete_at = ConversationalRagOrchestrator._unclosed_delimiter_position(
            stabilized
        )
        if final_incomplete_at is not None:
            prefix = ConversationalRagOrchestrator._complete_sentence_prefix(
                stabilized[:final_incomplete_at]
            )
            stabilized = prefix or stabilized[:final_incomplete_at].rstrip(" ,;:-")
        stabilized = ConversationalRagOrchestrator._trim_incomplete_tail(stabilized)
        warnings: tuple[str, ...] = ()
        if (
            stabilized != answer.strip()
            or echo_sentences
            or leaked_sentences
            or unsupported_numeric_removed
        ):
            warnings = ("generation_output_stabilized",)
        if fact_packet and fact_packet.facts and not (
            ConversationalRagOrchestrator._generation_matches_fact_packet(
                stabilized, query=query, fact_packet=fact_packet
            )
        ):
            return (
                ConversationalRagOrchestrator._extractive_fact_answer(fact_packet),
                tuple(dict.fromkeys((*warnings, "unsafe_generation_replaced_extractive"))),
                True,
            )
        return stabilized, warnings, False

    @staticmethod
    def _generation_matches_fact_packet(
        answer: str, *, query: str, fact_packet: GroundedFactPacket
    ) -> bool:
        """Fail closed when generated claims cannot be traced to source fact units."""

        packet_text = " ".join(
            (*fact_packet.primary_sentences, *fact_packet.supporting_sentences)
        )
        def canonical(value: str) -> str:
            replacements = (
                (r"지어졌|지었다|세워졌|세웠|건립되었|건립했다", "건립"),
                (r"문을\s*열었|개관하였|개관했", "개관"),
                (r"도착하였|도착했", "도착"),
                (r"시작하였|시작했|개시하였|개시했", "시작"),
                (r"사용되었|사용됐", "사용"),
            )
            result = value
            for pattern, replacement in replacements:
                result = re.sub(pattern, replacement, result)
            return result

        canonical_packet = canonical(packet_text)
        canonical_answer = canonical(answer)
        if fact_packet.intent == "people" and re.search(
            r"관련(?:된)?\s*인물|인물은|누구|누가", query
        ):
            named_people = ConversationalRagOrchestrator._fact_packet_named_people(
                fact_packet
            )
            if named_people and not any(name in answer for name in named_people):
                return False
            if ConversationalRagOrchestrator._has_impossible_person_title_subject(
                answer, subject=fact_packet.subject,
                subject_is_person=fact_packet.subject in named_people,
            ):
                return False
        if fact_packet.intent == "place":
            places = ConversationalRagOrchestrator._fact_packet_places(fact_packet)
            if places and not any(place in answer for place in places):
                return False
        numeric = re.compile(
            r"(?:\d{3,4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?|"
            r"\d[\d,.]*\s*(?:명|회|개|척|평|㎡|㎞|%|톤)|제\s*\d+\s*대)"
        )
        if any(value not in packet_text for value in numeric.findall(answer)):
            return False
        sentences = [
            value.strip() for value in re.split(r"(?<=[.!?。！？])\s+|\n+", answer)
            if value.strip()
        ]
        if not sentences:
            return False
        if any(sentence.endswith(("?", "？")) for sentence in sentences):
            return False
        if any(
            ConversationalRagOrchestrator._is_question_echo(sentence, query)
            for sentence in sentences
        ):
            return False
        answer_subjects = explicit_subject_words(answer)
        packet_compact = re.sub(r"\s+", "", packet_text)
        if any(
            re.sub(r"\s+", "", subject) not in packet_compact
            for subject in answer_subjects
            if subject and subject != fact_packet.subject
        ):
            return False
        protected_claims = re.findall(
            r"[가-힣]{2,12}\s*(?:총리|장관|대사|교수|회장|사장|기관|회사)",
            answer,
        )
        if any(claim not in packet_text for claim in protected_claims):
            return False
        relation_families = (
            r"때문|위해|목적|원인|계기",
            r"영향|증가|감소|발전|확대|성장|형성",
            r"수탈|관리|운영|담당|참석|도착|건립|개통|개항|개관",
        )
        for family in relation_families:
            if re.search(family, canonical_answer) and not re.search(
                family, canonical_packet
            ):
                return False
        # These verbs change the historical relation rather than merely its
        # wording.  A fluent paraphrase may inflect them, but it may not add a
        # relation that is absent from the selected source fact units.
        factual_predicates = (
            "개조", "설치", "건립", "개통", "개항", "설립", "도착", "참석",
            "환영", "사용", "지정", "성장", "쇠퇴", "발전", "증가", "감소",
            "수탈", "운행", "증축", "신축", "준공", "원활",
        )
        for predicate in factual_predicates:
            if predicate in canonical_answer and predicate not in canonical_packet:
                return False
        packet_tokens = set(re.findall(r"[가-힣A-Za-z]{2,}|\d+", canonical_packet))
        ignored = {
            "그리고", "또한", "그래서", "하지만", "입니다", "있습니다",
            "했습니다", "했어요", "이에", "대한", "관한", "기록",
        }
        for sentence in sentences:
            tokens = [
                token for token in re.findall(r"[가-힣A-Za-z]{2,}|\d+", canonical(sentence))
                if token not in ignored
            ]
            if len(tokens) < 3:
                continue
            covered = sum(token in packet_tokens for token in tokens)
            if covered / len(tokens) < 0.35:
                return False
        return True

    @staticmethod
    def _extractive_fact_answer(fact_packet: GroundedFactPacket) -> str:
        if fact_packet.intent == "people":
            names = sorted(
                ConversationalRagOrchestrator._fact_packet_named_people(fact_packet)
            )
            if names:
                listed = ", ".join(names[:8])
                return f"{fact_packet.subject} 관련 인물로는 {listed} 등이 확인됩니다."
        if fact_packet.intent == "place":
            places = sorted(ConversationalRagOrchestrator._fact_packet_places(fact_packet))
            if places:
                listed = ", ".join(places[:5])
                return f"{fact_packet.subject}의 관련 장소로는 {listed}이 확인됩니다."
        primary = [
            sentence
            for sentence in dict.fromkeys(fact_packet.primary_sentences)
            if not _competitor_matches(sentence, fact_packet.subject)
        ]
        if not primary:
            return "확인되는 기록만으로는 그 내용을 정확히 설명하기 어려워요."
        answer = " ".join(primary[:2]).strip()
        if fact_packet.subject and fact_packet.subject not in answer[:80]:
            answer = f"{fact_packet.subject}에 관한 기록에는 {answer}"
        return ConversationalRagOrchestrator._trim_incomplete_tail(answer)

    @staticmethod
    def _fact_packet_named_people(
        fact_packet: GroundedFactPacket,
    ) -> set[str]:
        return {fact.person for fact in verified_person_facts(fact_packet)}

    @staticmethod
    def _fact_packet_places(fact_packet: GroundedFactPacket) -> set[str]:
        text = " ".join(
            (*fact_packet.primary_sentences, *fact_packet.supporting_sentences)
        )
        candidates = re.findall(
            r"[가-힣]{2,12}\s*지역|"
            r"[가-힣]{2,12}(?:동|리|읍|면|군|시|도|항|역|캠퍼스)"
            r"(?=\s*(?:과|와|,|에|에서|일대|$))",
            text,
        )
        return {
            re.sub(r"\s+", " ", value).strip()
            for value in candidates
            if not re.search(r"활동|운동|행동|노동", value)
        }

    @staticmethod
    def _has_impossible_person_title_subject(
        answer: str, *, subject: str, subject_is_person: bool,
    ) -> bool:
        if not subject or subject_is_person:
            return False
        title = r"(?:회장|지회장|대표|위원장|간사|총리|장관|대사|학장|총장)"
        predicate = r"(?:되|되어|됐|맡|역임|취임)"
        return bool(re.search(
            rf"(?:^|(?<=[.!?。！？])\s*){re.escape(subject)}(?:은|는|이|가)"
            rf"[^.!?。！？]{{0,40}}{title}(?:이|가|을|를)?\s*{predicate}",
            answer,
        ))

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
        subjects = explicit_subject_words(query)
        title = chunks[0].chunk.title.split(" - ", 1)[0].strip() if chunks else ""
        subject = subjects[0] if subjects else (
            title if title and title.casefold() not in {"evidence seed", "unknown"}
            else "관련 기록"
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
        return f"{subject}에 관해서는 {detail}을 정확히 말하기 어려워요. 확인되는 내용부터 이어서 설명해드릴게요."

    @staticmethod
    def _source_sufficiency(
        chunks: list[RankedChunk], *, evidence_support: EvidenceSupport | None = None
    ) -> SourceSufficiency:
        if not chunks:
            return SourceSufficiency.INSUFFICIENT
        if any(
            item.chunk.payload.get("source_conflict") is True
            or item.chunk.payload.get("fact_status") == "conflicting"
            for item in chunks
        ):
            return SourceSufficiency.CONFLICTING
        if evidence_support in {
            EvidenceSupport.PARTIAL,
            EvidenceSupport.RELATED_ONLY,
            EvidenceSupport.NONE,
        }:
            return SourceSufficiency.PARTIAL
        return SourceSufficiency.SUFFICIENT

    @staticmethod
    def _related_evidence_guidance(subject: str, intent: str) -> str:
        target = subject or "그 주제"
        detail = {
            "time": "정확한 시점",
            "cause": "그 이유나 목적",
            "people": "직접 관련된 인물과 행동",
            "result": "직접적인 영향이나 이후 변화",
            "role": "당시 맡았던 역할이나 업무",
            "current": "현재의 용도나 상태",
            "overview": "대상을 직접 설명하는 내용",
        }.get(intent, "질문한 내용")
        return (
            f"{target}에 관한 기록은 확인되지만, {detail}까지는 지금 확인되는 내용만으로 "
            "단정하기 어려워요. 확인되는 사건이나 시점을 중심으로 다시 물어보면 이어서 설명해드릴게요."
        )

    @staticmethod
    def _nearby_supported_answer(
        subject: str, intent: str, packet: GroundedFactPacket
    ) -> str:
        boundary = {
            "time": "질문한 정확한 시점은 직접 확인되지 않아요.",
            "cause": "그 이유 자체는 기록에서 직접 확인되지 않아요.",
            "people": "질문한 인물 관계는 직접 확인되지 않아요.",
            "result": "그것이 직접 일으킨 변화라고 단정할 수는 없어요.",
            "role": "질문한 구체적인 역할은 직접 확인되지 않아요.",
            "current": "현재 상태는 직접 확인되지 않아요.",
        }.get(intent, "질문한 세부 내용은 직접 확인되지 않아요.")
        fact = ConversationalRagOrchestrator._extractive_fact_answer(packet)
        if fact.startswith(f"{subject}에 관한 기록에는 "):
            fact = fact.removeprefix(f"{subject}에 관한 기록에는 ")
        return f"{boundary} 다만 기록에서는 {fact}"

    @staticmethod
    def _conflicting_evidence_guidance(subject: str, intent: str) -> str:
        target = subject or "그 대상"
        detail = "시점" if intent == "time" else "내용"
        return (
            f"{target}의 {detail}이 기록마다 다르게 나타나 하나로 단정하기 어려워요. "
            "서로 다른 기록을 임의로 합치지 않고 확인되는 차이만 알려드릴게요."
        )

    @staticmethod
    def _comparison_answer(
        packets: tuple[GroundedFactPacket, ...]
    ) -> tuple[str, bool]:
        parts: list[str] = []
        complete = True
        for packet in packets:
            if not packet.facts:
                complete = False
                parts.append(
                    f"{packet.subject}은 같은 기준으로 설명할 직접 기록을 찾기 어려워요."
                )
                continue
            fact = ConversationalRagOrchestrator._extractive_fact_answer(packet)
            parts.append(fact)
        if complete:
            parts.append("각 대상에 대해 기록으로 확인되는 내용은 이와 같아요.")
        return " ".join(parts), complete

    @staticmethod
    def _explain_evidence_boundary(subject: str, intent: str) -> str:
        target = subject or "그 주제"
        missing = {
            "time": "정확한 시점",
            "cause": "그 일이 일어난 직접적인 이유",
            "people": "직접 관련된 인물과 행동",
            "result": "그 일 때문에 생긴 직접적인 변화",
            "role": "당시 맡았던 구체적인 역할",
            "current": "현재의 용도나 상태",
        }.get(intent, "질문한 세부 내용")
        return (
            f"{target}과 관련된 기록은 있지만, {missing}에 관한 직접 설명은 "
            "찾기 어렵다는 뜻이에요. 기록에 나온 범위를 넘겨 짐작하지 않고 "
            "확인되는 부분까지만 설명한 거예요."
        )

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

    def _guard_grounded_answer(
        self, answer: str, *, output_domain: OutputDomain, situation,
        stage, locale: str, citations: tuple[dict[str, object], ...] = (),
        fact_packet: GroundedFactPacket | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        """Keep an answerable grounded turn from ending in a generic guard fallback."""

        guarded, warnings = self._guard_answer(
            answer,
            output_domain=output_domain,
            situation=situation,
            stage=stage,
            locale=locale,
            citations=citations,
        )
        if not warnings or fact_packet is None or not fact_packet.facts:
            return guarded, warnings

        extractive = self._extractive_fact_answer(fact_packet)
        repaired, repair_warnings = self._guard_answer(
            extractive,
            output_domain=output_domain,
            situation=situation,
            stage=stage,
            locale=locale,
            citations=citations,
        )
        if repair_warnings:
            return guarded, warnings
        return repaired, tuple(dict.fromkeys((*warnings, "style_guard_replaced_extractive")))

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

        explicit_subjects = explicit_subject_words(query)
        subject = active_place or (explicit_subjects[0] if explicit_subjects else "그 주제")
        verifies_premise = bool(re.search(
            r"전제|(?:말이\s*)?맞(?:아|지|는지)|(?:났|했|였|됐|있)지\?",
            query,
        ))
        asks_people = bool(re.search(r"인물|사람|누구", query))
        asks_date = bool(re.search(r"언제|건립|준공|만들|세워|생긴", query))
        entity_only = bool(re.fullmatch(r"\s*[가-힣一-龥·]{2,30}[?.!]*\s*", query))
        if verifies_premise:
            limitation = "그 전제는 확인된 기록과 맞지 않아요. 질문에 포함된 관계나 조건을 뒷받침하는 근거가 없습니다."
            suggestions = (f"{subject}에 관해 확인되는 사실을 알려줘.",)
        elif entity_only and explicit_subjects:
            limitation = f"{subject}에 대해 어떤 점이 궁금한가요? 활동이나 관련 사건, 시기를 함께 말해주면 더 정확히 찾아볼게요."
            suggestions = (f"{subject} 관련 사건을 알려줘.",)
        elif asks_people:
            limitation = f"{subject}과 직접 연결되는 인물은 지금 확인되는 기록만으로 특정하기 어려워요. 사건이나 시기를 조금 더 좁혀 주면 다시 찾아볼게요."
            suggestions = (f"{subject} 관련 사건을 알려줘.",)
        elif asks_date:
            limitation = f"{subject}의 정확한 시점은 지금 확인되는 기록만으로 단정하기 어려워요. 관련 사건이나 장소 이름을 덧붙여 주면 다시 찾아볼게요."
            suggestions = (f"{subject} 관련 사건을 알려줘.",)
        elif active_place:
            limitation = f"{subject}에 관한 그 부분은 지금 확인되는 기록만으로 정확히 설명하기 어려워요. 다른 사건이나 인물과의 관계를 물어보면 이어서 살펴볼 수 있어요."
            suggestions = (f"{subject}의 역사적 역할을 알려줘.",)
        else:
            limitation = "그 내용은 지금 확인되는 기록만으로 정확히 설명하기 어려워요. 장소나 사건, 인물 이름을 조금 더 구체적으로 알려주면 다시 찾아볼게요."
            suggestions = ("목포의 철도와 항만 이야기를 알려줘.",)

        examples = "\n".join(f"- {item}" for item in suggestions)
        if domain == OutputDomain.CHARACTER_DIALOGUE:
            answer = f"{limitation}\n\n{examples}"
        else:
            answer = f"{limitation}\n\n{examples}"
        return answer, suggestions
