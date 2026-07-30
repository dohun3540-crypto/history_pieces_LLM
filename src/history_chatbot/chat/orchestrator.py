"""세션부터 검색·프롬프트·답변·출처까지 연결하는 grounded RAG."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterator

from history_chatbot.chat.citation_builder import build_citations
from history_chatbot.chat.interfaces import Citation
from history_chatbot.chat.prompt_builder import PROMPT_VERSION, build_prompt
from history_chatbot.chat.session import ChatSession, SessionStore
from history_chatbot.models.mock_llm import MockLLM
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
        llm: MockLLM,
        sessions: SessionStore,
        *,
        mode: RuntimeMode,
        max_chunks_per_document: int = 2,
    ) -> None:
        if mode == RuntimeMode.PRODUCTION:
            raise ValueError("production 모드에서는 MockLLM 기반 오케스트레이터를 사용할 수 없습니다.")
        self.retrieval = retrieval
        self.llm = llm
        self.sessions = sessions
        self.mode = mode
        self.max_chunks_per_document = max_chunks_per_document

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
        prompt = build_prompt(
            user_query=query,
            conversation_summary=self._conversation_context(session),
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
            )
        else:
            is_fixture = all(
                item.chunk.payload.get("data_classification") == "fictional_fixture"
                for item in chunks
            )
            answer = self.llm.generate_grounded(
                prompt=prompt,
                evidence=tuple(item.chunk.text for item in chunks),
                is_fixture=is_fixture,
            )
            response = ChatResponse(
                answer,
                "ok",
                build_citations(chunks),
                len(chunks),
                session.session_id,
                locale,
                PROMPT_VERSION,
            )
        self.sessions.add_turn(session.session_id, query, response.answer)
        return response

    def stream(self, *args, **kwargs) -> Iterator[StreamEvent]:
        response = self.ask(*args, **kwargs)
        # 현재 Mock은 문장/토큰 스트리밍을 흉내 낸다. complete는 별도 메타 이벤트다.
        for token in response.answer.split(" "):
            yield StreamEvent("token", {"text": token + " "})
        yield StreamEvent("complete", response.to_dict())

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
    def _conversation_context(session: ChatSession) -> str:
        recent = "\n".join(
            f"사용자: {turn.user}\n응답: {turn.assistant}" for turn in session.turns[-3:]
        )
        return "\n".join(part for part in (session.summary, recent) if part)
