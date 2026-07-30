"""수집 파이프라인에서 공유하는 자료형."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class CopyrightStatus(StrEnum):
    PUBLIC_DOMAIN = "public_domain"
    OPEN_LICENSE = "open_license"
    PERMISSION_GRANTED = "permission_granted"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    EXTRACTED = "extracted"
    CLEANED = "cleaned"
    METADATA_ADDED = "metadata_added"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


@dataclass(slots=True)
class SourceDocument:
    document_id: str
    title: str
    source_type: str
    publisher: str
    author: str
    source_url: str
    local_path: str
    published_date: str
    accessed_date: str
    language: str
    license_name: str
    license_url: str
    copyright_status: CopyrightStatus
    allowed_for_rag: bool
    allowed_for_training: bool
    redistribution_allowed: bool
    attribution_required: bool
    attribution_text: str
    notes: str
    review_status: ReviewStatus
    reviewed_by: str
    reviewed_at: str
    period_start: int | None = None
    period_end: int | None = None
    historical_period: str = ""
    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source_reliability: str = ""
    verification_notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceDocument":
        values = dict(data)
        values["copyright_status"] = CopyrightStatus(values["copyright_status"])
        values["review_status"] = ReviewStatus(values["review_status"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    original_text: str
    extractor: str
    source_path: str


@dataclass(frozen=True, slots=True)
class CleanedText:
    original_text: str
    cleaned_text: str
    cleaning_log: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngestionChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str
    title: str
    source: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    document: SourceDocument
    cleaned: CleanedText
    chunks: tuple[IngestionChunk, ...]
    output_path: str
