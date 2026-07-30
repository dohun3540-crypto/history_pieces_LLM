"""공통 단어 오탐을 막고 근거가 약한 검색 결과를 제거한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from history_chatbot.retrieval.base import RankedChunk
from history_chatbot.retrieval.query_normalizer import NormalizedQuery, tokenize


def apply_thresholds(
    query: NormalizedQuery,
    results: Sequence[RankedChunk],
    *,
    minimum_score: float,
    minimum_dense_score: float,
    max_chunks_per_document: int,
    final_top_k: int,
) -> list[RankedChunk]:
    selected: list[RankedChunk] = []
    per_document: Counter[str] = Counter()
    informative = set(query.informative_tokens)
    for result in results:
        searchable = set(tokenize(f"{result.chunk.title} {result.chunk.text}"))
        lexical_evidence = bool(informative & searchable)
        semantic_evidence = result.dense_score >= minimum_dense_score
        if result.score < minimum_score or not (lexical_evidence or semantic_evidence):
            continue
        if per_document[result.chunk.document_id] >= max_chunks_per_document:
            continue
        selected.append(result)
        per_document[result.chunk.document_id] += 1
        if len(selected) >= final_top_k:
            break
    return selected
