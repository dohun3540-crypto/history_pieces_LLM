"""검수 manifest와 처리 청크를 대조해 추적 가능한 입력을 로딩한다."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from history_chatbot.indexing.eligibility import RagEligibilityPolicy
from history_chatbot.indexing.snapshot import sha256_file, stable_json_hash
from history_chatbot.ingestion.models import SourceDocument
from history_chatbot.ingestion.source_registry import SourceRegistry
from history_chatbot.ingestion.validator import validate_local_path


@dataclass(frozen=True, slots=True)
class IndexChunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    title: str
    publisher: str
    source_url: str
    author: str
    published_date: str
    language: str
    copyright_status: str
    license_name: str
    license_url: str
    attribution_required: bool
    attribution_text: str
    source_reliability: str
    review_status: str
    allowed_for_rag: bool
    reviewed_by: str
    reviewed_at: str
    period_start: int | None
    period_end: int | None
    historical_period: str
    people: tuple[str, ...]
    places: tuple[str, ...]
    organizations: tuple[str, ...]
    events: tuple[str, ...]
    keywords: tuple[str, ...]
    start_char: int | None
    end_char: int | None
    page: int | None
    section: str | None
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("people", "places", "organizations", "events", "keywords"):
            value[name] = list(value[name])
        value.update(
            {
                "approval_tier": "production_approved",
                "data_classification": "real_historical_source",
                "is_fixture": False,
                "development_only": False,
                "production_approved": True,
                "source_status": "production",
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    document: SourceDocument
    chunks: tuple[IndexChunk, ...]
    raw_sha256: str
    processed_sha256: str
    snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class RejectedDocument:
    document_id: str
    title: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadReport:
    eligible: tuple[LoadedDocument, ...]
    rejected: tuple[RejectedDocument, ...]


class ReviewedChunkLoader:
    def __init__(
        self,
        registry: SourceRegistry,
        raw_root: Path,
        processed_dir: Path,
        policy: RagEligibilityPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.raw_root = raw_root
        self.processed_dir = processed_dir
        self.policy = policy or RagEligibilityPolicy()

    def load(self) -> LoadReport:
        eligible: list[LoadedDocument] = []
        rejected: list[RejectedDocument] = []
        for document in self.registry.list():
            decision = self.policy.evaluate(document)
            if not decision.eligible:
                rejected.append(
                    RejectedDocument(document.document_id, document.title, decision.reasons)
                )
                continue
            try:
                eligible.append(self._load_document(document))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                rejected.append(
                    RejectedDocument(document.document_id, document.title, (str(error),))
                )
        return LoadReport(tuple(eligible), tuple(rejected))

    def _load_document(self, document: SourceDocument) -> LoadedDocument:
        path_errors = validate_local_path(document, self.raw_root)
        if path_errors:
            raise ValueError("; ".join(path_errors))
        raw_path = Path(document.local_path)
        processed_path = self.processed_dir / f"{document.document_id}.jsonl"
        if not processed_path.is_file():
            raise ValueError(f"처리된 청크 파일이 없습니다: {processed_path}")

        chunks: list[IndexChunk] = []
        with processed_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(self._to_index_chunk(document, record, line_number))
        if not chunks:
            raise ValueError("처리된 청크가 없습니다.")

        raw_hash = sha256_file(raw_path)
        processed_hash = sha256_file(processed_path)
        snapshot_hash = stable_json_hash(
            {
                "document": document.to_dict(),
                "raw_sha256": raw_hash,
                "processed_sha256": processed_hash,
            }
        )
        return LoadedDocument(
            document,
            tuple(chunks),
            raw_hash,
            processed_hash,
            snapshot_hash,
        )

    @staticmethod
    def _to_index_chunk(
        document: SourceDocument,
        record: dict[str, Any],
        line_number: int,
    ) -> IndexChunk:
        document_id = str(record.get("document_id", "")).strip()
        chunk_id = str(record.get("chunk_id", "")).strip()
        text = str(record.get("text", "")).strip()
        if document_id != document.document_id:
            raise ValueError(
                f"청크 {line_number}의 document_id 추적 관계가 일치하지 않습니다."
            )
        if not chunk_id:
            raise ValueError(f"청크 {line_number}에 chunk_id가 없습니다.")
        if not text:
            raise ValueError(f"청크 {line_number}에 본문이 없습니다.")
        if not chunk_id.startswith(f"{document.document_id}::"):
            raise ValueError(f"청크 {line_number}의 chunk_id 추적 관계가 유효하지 않습니다.")

        normalized_text = re.sub(r"\s+", " ", text).strip()
        metadata = record.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"청크 {line_number}의 metadata가 객체가 아닙니다.")
        return IndexChunk(
            document_id=document.document_id,
            chunk_id=chunk_id,
            chunk_index=int(record.get("chunk_index", line_number - 1)),
            text=text,
            title=document.title,
            publisher=document.publisher,
            source_url=document.source_url,
            author=document.author,
            published_date=document.published_date,
            language=document.language,
            copyright_status=document.copyright_status.value,
            license_name=document.license_name,
            license_url=document.license_url,
            attribution_required=document.attribution_required,
            attribution_text=document.attribution_text,
            source_reliability=document.source_reliability,
            review_status=document.review_status.value,
            allowed_for_rag=document.allowed_for_rag,
            reviewed_by=document.reviewed_by,
            reviewed_at=document.reviewed_at,
            period_start=document.period_start,
            period_end=document.period_end,
            historical_period=document.historical_period,
            people=tuple(document.people),
            places=tuple(document.places),
            organizations=tuple(document.organizations),
            events=tuple(document.events),
            keywords=tuple(document.keywords),
            start_char=record.get("start_char"),
            end_char=record.get("end_char"),
            page=record.get("page"),
            section=record.get("section"),
            content_sha256=stable_json_hash(normalized_text),
        )
