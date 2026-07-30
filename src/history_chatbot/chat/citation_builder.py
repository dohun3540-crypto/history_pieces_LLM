"""실제로 선택된 청크에서만 출처를 생성한다."""

from __future__ import annotations

from collections.abc import Sequence

from history_chatbot.chat.interfaces import Citation
from history_chatbot.retrieval.base import RankedChunk


def build_citations(chunks: Sequence[RankedChunk]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen_documents: set[str] = set()
    for item in chunks:
        if item.chunk.document_id in seen_documents:
            continue
        seen_documents.add(item.chunk.document_id)
        payload = item.chunk.payload
        citations.append(
            Citation(
                source_id=str(payload.get("source_id") or item.chunk.document_id),
                document_id=item.chunk.document_id,
                title=item.chunk.title,
                institution=item.chunk.publisher,
                source_url=item.chunk.source_url,
                chunk_id=item.chunk.chunk_id,
                excerpt=item.chunk.text[:300],
                retrieval_score=round(item.score, 6),
                license_status=str(payload.get("copyright_status", "unknown")),
                is_fixture=payload.get("data_classification") == "fictional_fixture",
            )
        )
    return tuple(citations)
