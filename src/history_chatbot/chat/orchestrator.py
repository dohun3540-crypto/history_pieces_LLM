"""세션부터 검색·프롬프트·답변·출처까지 연결하는 grounded RAG."""

from __future__ import annotations

import re
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

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["sources"] = [asdict(source) for source in self.sources]
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

    def ask(
        self,
        user_query: str,
        *,
        session_id: str | None = None,
        locale: str = "ko",
        top_k: int = 3,
    ) -> ChatResponse:
        query = self._validate(user_query, locale, top_k)
        session = self.sessions.get_or_create(session_id, locale)
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
            user_query=query,
            conversation_summary="\n".join(budget.conversation),
            chunks=chunks,
            locale=locale,
        )
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
                response = ChatResponse(
                    answer,
                    "ok",
                    citations,
                    len(chunks),
                    session.session_id,
                    locale,
                    PROMPT_VERSION,
                    context_metadata=request.metadata.get("context_budget"),  # type: ignore[arg-type]
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
