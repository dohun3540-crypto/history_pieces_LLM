"""후속 구현이 교체 가능하도록 정의한 대화·출처·웹 서비스 인터페이스."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from history_chatbot.retrieval.base import RankedChunk


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: str
    document_id: str
    title: str
    institution: str
    source_url: str
    chunk_id: str
    excerpt: str
    retrieval_score: float
    license_status: str
    is_fixture: bool
    usage_status: str = ""
    rights_status: str = ""
    usage_scope: str = ""
    provisional_notice: str = ""
    source_status: str = ""
    approval_tier: str = ""
    production_approved: bool | None = None
    badge_label: str = ""
    usage_notice: str = ""


class ConversationSession(Protocol):
    def add(self, role: str, text: str) -> None: ...
    def clear(self) -> None: ...


class CitationBuilder(Protocol):
    def build(self, chunks: Sequence[RankedChunk]) -> tuple[Citation, ...]: ...


class RagOrchestrator(Protocol):
    def answer(self, query: str) -> tuple[str, tuple[Citation, ...]]: ...
    def stream(self, query: str) -> Iterator[str]: ...


class ChatService(Protocol):
    """로컬 웹 UI가 HTTP/WebSocket 어댑터를 통해 호출할 서비스 계층."""

    def ask(self, session_id: str, query: str) -> dict[str, object]: ...
