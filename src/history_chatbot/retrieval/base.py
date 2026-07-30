"""하이브리드 검색 계층의 저장소 독립 인터페이스와 자료형."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RetrievalChunk:
    document_id: str
    chunk_id: str
    text: str
    title: str
    publisher: str
    source_url: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RetrievalChunk":
        required = (
            "document_id",
            "chunk_id",
            "text",
            "title",
            "publisher",
            "source_url",
        )
        missing = [name for name in required if not str(record.get(name, "")).strip()]
        if missing:
            raise ValueError(f"index_ready 청크 필수 필드 누락: {', '.join(missing)}")
        return cls(
            document_id=str(record["document_id"]),
            chunk_id=str(record["chunk_id"]),
            text=str(record["text"]),
            title=str(record["title"]),
            publisher=str(record["publisher"]),
            source_url=str(record["source_url"]),
            payload=dict(record),
        )


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: RetrievalChunk
    score: float
    methods: tuple[str, ...]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    reranker_score: float | None = None


class DenseEncoder(ABC):
    model_id: str
    revision: str
    dimension: int

    @abstractmethod
    def encode(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        """텍스트를 정규화된 벡터로 변환한다."""


class VectorStore(ABC):
    @abstractmethod
    def replace(
        self,
        entries: Iterable[tuple[RetrievalChunk, Sequence[float]]],
        *,
        model_id: str,
        revision: str,
        source_snapshot: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """현재 활성 청크 전체를 원자적으로 교체한다."""

    @abstractmethod
    def search(self, vector: Sequence[float], limit: int) -> list[RankedChunk]:
        """벡터 유사도 순으로 검색한다."""

    @abstractmethod
    def chunks(self) -> list[RetrievalChunk]:
        """현재 활성 청크를 반환한다."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """인덱스 버전 정보를 반환한다."""


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, results: Sequence[RankedChunk]
    ) -> list[RankedChunk]:
        """후보 순서를 다시 계산한다."""
