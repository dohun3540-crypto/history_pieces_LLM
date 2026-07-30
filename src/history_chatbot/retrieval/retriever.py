"""교체 가능한 검색 인터페이스와 메모리 키워드 구현."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from history_chatbot.preprocessing.normalize_korean import normalize_korean
from history_chatbot.retrieval.document import Document, SearchResult


class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """관련 문서를 점수순으로 반환한다."""


class KeywordRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[0-9A-Za-z가-힣]+", normalize_korean(text).lower())
            if len(token) > 1
        }

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        results: list[SearchResult] = []
        for document in self.documents:
            searchable = f"{document.title} {document.content}"
            document_tokens = self._tokens(searchable)
            matches = query_tokens & document_tokens
            if matches:
                score = len(matches) / len(query_tokens)
                results.append(SearchResult(document=document, score=score))
        return sorted(results, key=lambda item: (-item.score, item.document.id))[:top_k]
