"""문단과 문장 경계를 우선하는 보수적 문자 청커."""

from __future__ import annotations

import re
from dataclasses import dataclass

from history_chatbot.ingestion.models import IngestionChunk, SourceDocument


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int


class DocumentChunker:
    def __init__(self, max_chars: int = 800, overlap: int = 80) -> None:
        if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
            raise ValueError("max_chars는 양수이고 overlap은 0 이상 max_chars 미만이어야 합니다.")
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, text: str, document: SourceDocument) -> list[IngestionChunk]:
        spans = self._paragraph_spans(text)
        bounded = [
            subspan
            for span in spans
            for subspan in self._split_long_span(text, span.start, span.end)
        ]
        overlapped = self._apply_overlap(text, bounded)
        metadata = self._metadata(document)
        return [
            IngestionChunk(
                chunk_id=f"{document.document_id}::chunk-{index:04d}",
                document_id=document.document_id,
                chunk_index=index,
                start_char=span.start,
                end_char=span.end,
                text=text[span.start : span.end],
                title=document.title,
                source=document.source_url or document.publisher,
                metadata=metadata,
            )
            for index, span in enumerate(overlapped)
            if text[span.start : span.end].strip()
        ]

    @staticmethod
    def _paragraph_spans(text: str) -> list[_Span]:
        return [
            _Span(match.start(), match.end())
            for match in re.finditer(r"\S(?:.*?\S)?(?=\n{2,}|\Z)", text, re.DOTALL)
        ]

    def _split_long_span(self, text: str, start: int, end: int) -> list[_Span]:
        spans: list[_Span] = []
        cursor = start
        while end - cursor > self.max_chars:
            limit = cursor + self.max_chars
            candidates = [
                match.end()
                for match in re.finditer(r"(?:[.!?。！？]\s+|\n)", text[cursor:limit])
                if match.end() >= self.max_chars // 2
            ]
            cut = cursor + (candidates[-1] if candidates else self.max_chars)
            spans.append(_Span(cursor, cut))
            cursor = cut
            while cursor < end and text[cursor].isspace():
                cursor += 1
        if cursor < end:
            spans.append(_Span(cursor, end))
        return spans

    def _apply_overlap(self, text: str, spans: list[_Span]) -> list[_Span]:
        if not self.overlap:
            return spans
        result: list[_Span] = []
        for index, span in enumerate(spans):
            start = span.start
            if index:
                start = max(spans[index - 1].start, start - self.overlap)
                while start < span.start and start > 0 and not text[start - 1].isspace():
                    start += 1
            result.append(_Span(start, span.end))
        return result

    @staticmethod
    def _metadata(document: SourceDocument) -> dict[str, object]:
        return {
            "document_id": document.document_id,
            "title": document.title,
            "source_name": document.publisher,
            "source_url": document.source_url,
            "author": document.author,
            "published_date": document.published_date,
            "review_status": document.review_status.value,
            "language": document.language,
            "period_start": document.period_start,
            "period_end": document.period_end,
            "historical_period": document.historical_period,
            "people": document.people,
            "places": document.places,
            "organizations": document.organizations,
            "events": document.events,
            "keywords": document.keywords,
            "source_reliability": document.source_reliability,
            "verification_notes": document.verification_notes,
            "copyright_status": document.copyright_status.value,
            "license_name": document.license_name,
            "license": document.license_name,
            "license_url": document.license_url,
            "attribution_text": document.attribution_text,
            "redistribution_allowed": document.redistribution_allowed,
        }
