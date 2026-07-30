"""reranker 선택 지점. 기본값은 점수를 변경하지 않는다."""

from __future__ import annotations

from collections.abc import Sequence

from history_chatbot.retrieval.base import RankedChunk, Reranker


class NoOpReranker(Reranker):
    def rerank(self, query: str, results: Sequence[RankedChunk]) -> list[RankedChunk]:
        del query
        return list(results)
