"""검색 문서와 결과의 공통 자료형."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    source: str
    content: str
    language: str = "ko"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    score: float
