"""외부 의존성 없는 BM25 어휘 검색."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.retrieval.query_normalizer import normalize_query, tokenize


class BM25Searcher:
    def __init__(self, chunks: Sequence[RetrievalChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [
            tokenize(f"{chunk.title} {chunk.text} {' '.join(chunk.payload.get('keywords', []))}")
            for chunk in self.chunks
        ]
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        )
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        query_tokens = set(normalize_query(query).informative_tokens)
        if not query_tokens or limit <= 0 or not self.chunks:
            return []
        scored: list[RankedChunk] = []
        count = len(self.chunks)
        for chunk, tokens in zip(self.chunks, self._tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = len(tokens) / self._average_length if self._average_length else 0
                score += inverse_frequency * (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * (1 - self.b + self.b * length_ratio))
                )
            if score > 0:
                scored.append(
                    RankedChunk(chunk, score, ("sparse",), sparse_score=score)
                )
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]
