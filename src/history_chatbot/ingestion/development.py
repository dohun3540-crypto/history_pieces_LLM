"""Isolated development-only metadata and loading policy for real sources."""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from history_chatbot.runtime import RuntimeMode


class ApprovalTier(StrEnum):
    DEVELOPMENT_PENDING_REVIEW = "development_pending_review"
    DEVELOPMENT_APPROVED = "development_approved"
    PRODUCTION_APPROVED = "production_approved"


class DataClassification(StrEnum):
    FICTIONAL_FIXTURE = "fictional_fixture"
    REAL_HISTORICAL_SOURCE = "real_historical_source"


@dataclass(frozen=True, slots=True)
class DevelopmentSourceDocument:
    document_id: str
    title: str
    publisher: str
    canonical_source_url: str
    accessed_date: str
    language: str
    primary_topic: str
    factual_summary: str
    evidence_quote: str
    citation_title: str
    citation_url: str
    approval_tier: ApprovalTier
    data_classification: DataClassification
    is_fixture: bool
    development_only: bool
    production_approved: bool
    public_release_allowed: bool
    license_review_status: str
    raw_source_status: str
    review_status: str
    source_status: str
    source_reliability: str
    development_approved_by: str = ""
    development_approved_at: str = ""
    development_approval_notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DevelopmentSourceDocument":
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.name in data:
                values[field.name] = data[field.name]
            elif field.default is not MISSING:
                values[field.name] = field.default
            else:
                values[field.name] = ""
        values["approval_tier"] = ApprovalTier(values["approval_tier"])
        values["data_classification"] = DataClassification(values["data_classification"])
        return cls(**values)

    def validation_errors(self, *, allow_pending: bool = False) -> tuple[str, ...]:
        errors: list[str] = []
        required = (
            "document_id", "title", "publisher", "canonical_source_url",
            "accessed_date", "language", "primary_topic", "factual_summary",
            "evidence_quote", "citation_title", "citation_url",
            "license_review_status", "raw_source_status", "review_status",
            "source_reliability",
        )
        for name in required:
            if not str(getattr(self, name)).strip():
                errors.append(f"missing_required_metadata:{name}")
        for name in ("canonical_source_url", "citation_url"):
            parsed = urlsplit(str(getattr(self, name)))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                errors.append(f"invalid_url:{name}")
        if self.data_classification != DataClassification.REAL_HISTORICAL_SOURCE:
            errors.append("data_classification_must_be_real_historical_source")
        if self.is_fixture is not False:
            errors.append("is_fixture_must_be_false")
        if self.development_only is not True:
            errors.append("development_only_must_be_true")
        if self.production_approved is not False:
            errors.append("production_approved_must_be_false")
        if self.public_release_allowed is not False:
            errors.append("public_release_allowed_must_be_false")
        if self.source_status != "development_only":
            errors.append("source_status_must_be_development_only")
        if self.review_status != "verified_pending_production_review":
            errors.append("review_status_not_development_safe")
        if self.source_reliability not in {"A", "B"}:
            errors.append("source_reliability_must_be_A_or_B")
        if self.license_review_status not in {"pending_review", "reviewed"}:
            errors.append("invalid_license_review_status")
        if self.raw_source_status not in {"remote_only", "pending_archive", "archived"}:
            errors.append("invalid_raw_source_status")
        if self.approval_tier == ApprovalTier.PRODUCTION_APPROVED:
            errors.append("production_approved_tier_forbidden")
        elif self.approval_tier == ApprovalTier.DEVELOPMENT_PENDING_REVIEW:
            if not allow_pending:
                errors.append("development_pending_review")
        elif self.approval_tier == ApprovalTier.DEVELOPMENT_APPROVED:
            for name in (
                "development_approved_by", "development_approved_at",
                "development_approval_notes",
            ):
                if not str(getattr(self, name)).strip():
                    errors.append(f"explicit_development_approval_missing:{name}")
        return tuple(dict.fromkeys(errors))

    def citation_metadata(self) -> dict[str, object]:
        return {
            "source_status": "development_only",
            "approval_tier": self.approval_tier.value,
            "production_approved": False,
            "badge_label": "개발 검증용 자료",
            "usage_notice": "실제 역사 자료이나 production 공개 승인을 받지 않았습니다.",
        }


@dataclass(frozen=True, slots=True)
class DevelopmentLoadReport:
    approved: tuple[DevelopmentSourceDocument, ...]
    rejected: tuple[tuple[str, tuple[str, ...]], ...]


class DevelopmentManifestLoader:
    """Read-only loader; pending records require an explicit per-document opt-in."""

    def __init__(
        self,
        path: Path,
        *,
        runtime_mode: RuntimeMode,
        explicitly_selected_pending_ids: Iterable[str] = (),
    ) -> None:
        if runtime_mode not in {RuntimeMode.DEVELOPMENT, RuntimeMode.TEST}:
            raise ValueError("development manifest는 development/test에서만 로드할 수 있습니다.")
        self.path = path
        self.explicitly_selected_pending_ids = frozenset(explicitly_selected_pending_ids)

    def load(self) -> DevelopmentLoadReport:
        if not self.path.is_file():
            return DevelopmentLoadReport((), ())
        records = (
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return self.evaluate(records)

    def evaluate(self, records: Iterable[dict[str, Any]]) -> DevelopmentLoadReport:
        approved: list[DevelopmentSourceDocument] = []
        rejected: list[tuple[str, tuple[str, ...]]] = []
        for record in records:
            document_id = str(record.get("document_id", ""))
            try:
                document = DevelopmentSourceDocument.from_dict(record)
            except (TypeError, ValueError) as error:
                rejected.append((document_id, (f"invalid_schema:{error}",)))
                continue
            errors = document.validation_errors(
                allow_pending=document_id in self.explicitly_selected_pending_ids
            )
            if errors:
                rejected.append((document_id, errors))
            else:
                approved.append(document)
        return DevelopmentLoadReport(tuple(approved), tuple(rejected))
