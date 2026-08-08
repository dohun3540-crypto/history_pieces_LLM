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
        contextual_query = self._contextualize_query(
            self._journey_scoped_query(
                query, classification.primary_situation_id.value, visited_piece_ids
            ),
            current_place_id=current_place_id,
            current_piece_id=current_piece_id,
            completed_place_ids=completed_place_ids,
            completed_piece_ids=visited_piece_ids,
        )
        prompt = build_prompt(
            user_query=contextual_query,
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
            conversation_mode=chat_mode,
            output_domain=output_domain,
            situation=classification.primary_situation_id,
            conversation_stage=stage,
        )
        if not chunks:
            insufficient_common = dict(common)
            insufficient_common.update(
                request_state="insufficient_evidence", ui_state="insufficient_evidence",
                rag_used=True, source_sufficiency=SourceSufficiency.INSUFFICIENT.value,
            )
            response = ChatResponse(
                self._insufficient_text(output_domain, locale),
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
                contextual_query, prompt, session, chunks, is_fixture, budget,
                locale=locale, conversation_mode=chat_mode,
                output_domain=output_domain,
                situation=classification.primary_situation_id,
                stage=stage,
            )
            try:
                completion = self.llm.complete(request)
                citations = build_citations(chunks)
                answer = self._apply_hackathon_policy(
                    completion.generated_text, chunks
                )
                answer = self._apply_repetition_guard(
                    answer, output_domain=output_domain
                )
                if self.llm.backend_name == "mock":
                    answer = render_mock_grounded(answer, domain=output_domain, locale=locale)
                    if self._source_sufficiency(chunks) == SourceSufficiency.CONFLICTING:
                        answer = "자료마다 설명이 달라. 확인된 차이를 나눠서 볼게. " + answer
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
                    "source_sufficiency": self._source_sufficiency(chunks).value,
                    "warnings": tuple(dict.fromkeys(common["warnings"] + style_warnings)),
                }
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
                    **grounded_common,
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
        request = self._llm_request(
            query, prompt, session, chunks, is_fixture, budget,
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
    def _journey_scoped_query(query: str, situation_id: str, visited_piece_ids: tuple[str, ...]) -> str:
        if situation_id != "JOURNEY_CONTEXT_QUESTION":
            return query
        completed = ", ".join(visited_piece_ids) if visited_piece_ids else "없음"
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
        if current_place_id:
            context.append(f"현재 장소 ID: {current_place_id}")
        if current_piece_id:
            context.append(f"현재 조각 ID: {current_piece_id}")
        if completed_place_ids:
            context.append("완료 장소 ID: " + ", ".join(completed_place_ids))
        if completed_piece_ids:
            context.append("완료 조각 ID: " + ", ".join(completed_piece_ids))
        if not context:
            return query
        return (
            query
            + "\n[관광 여정 문맥 | 역사적 사실의 근거가 아님] "
            + "; ".join(context)
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

    def _llm_request(
        self, query, prompt, session, chunks, is_fixture, budget, *,
        locale, conversation_mode, output_domain, situation, stage,
    ):
        remote_config = getattr(self.llm, "config", None)
        messages = tuple(
            message
            for turn in session.turns[-3:]
            for message in (
                LLMMessage("user", turn.user),
                LLMMessage("assistant", turn.assistant),
            )
        )
        system_prompt = SYSTEM_INSTRUCTIONS
        user_prompt = prompt
        if self.llm.backend_name == "remote" and remote_config is not None:
            safe = serialize_remote_prompt(
                system_prompt=(
                    SYSTEM_INSTRUCTIONS
                    + "\n"
                    + build_persona_prompt(
                        domain=output_domain, locale=locale,
                        mode=conversation_mode, situation=situation, stage=stage,
                    )
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
    def _insufficient_text(domain: OutputDomain, locale: str) -> str:
        if locale.lower() == "zh-cn":
            return ConversationalRagOrchestrator._pending_zh_text(domain)
        if domain == OutputDomain.CHARACTER_DIALOGUE:
            return "지금 확인할 수 있는 자료가 부족해. 추측해서 말하지 않을게."
        if domain == OutputDomain.HISTORICAL_DOCENT:
            return "현재 검수된 자료만으로는 확인할 수 없습니다."
        return "현재 확인 가능한 자료가 부족합니다."
