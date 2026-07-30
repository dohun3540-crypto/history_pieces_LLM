"""생성 모델 교체를 위한 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from history_chatbot.retrieval.document import SearchResult


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    original_query: str
    normalized_query: str
    contexts: tuple[SearchResult, ...]


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """검색 문맥을 바탕으로 답변을 생성한다."""
