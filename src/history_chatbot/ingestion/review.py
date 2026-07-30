"""사람 검수 승인·거절과 append-only 감사 로그."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from history_chatbot.ingestion.models import (
    CopyrightStatus,
    ReviewStatus,
    SourceDocument,
)
from history_chatbot.ingestion.source_registry import SourceRegistry
from history_chatbot.ingestion.validator import (
    can_index_for_service,
    validate_local_path,
    validate_source_document,
)


REVIEW_REQUIRED_FIELDS = (
    "document_id",
    "title",
    "source_type",
    "publisher",
    "source_url",
    "local_path",
    "accessed_date",
    "language",
    "license_name",
    "license_url",
)


class ReviewError(ValueError):
    """검수 명령을 안전하게 완료할 수 없을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class ReviewAuditEntry:
    document_id: str
    action: str
    reviewer: str
    occurred_at: str
    previous_status: str
    new_status: str
    reason: str
    copyright_status: str
    allowed_for_rag: bool
    allowed_for_training: bool
    rag_index_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReviewAuditLog:
    """기존 항목을 수정하지 않고 한 줄씩 추가하는 JSONL 감사 로그."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, entry: ReviewAuditEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def list(self) -> list[ReviewAuditEntry]:
        if not self.path.exists():
            return []
        entries: list[ReviewAuditEntry] = []
        with self.path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    entries.append(ReviewAuditEntry(**json.loads(line)))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"감사 로그 {line_number}번째 줄이 유효하지 않습니다: {error}"
                    ) from error
        return entries


class ReviewService:
    def __init__(
        self,
        registry: SourceRegistry,
        audit_log: ReviewAuditLog,
        raw_root: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.registry = registry
        self.audit_log = audit_log
        self.raw_root = raw_root
        self.now = now or (lambda: datetime.now(timezone.utc).astimezone())

    def show(self, document_id: str) -> SourceDocument:
        return self.registry.get(document_id)

    def approve(
        self,
        document_id: str,
        reviewer: str,
        verification_notes: str = "",
    ) -> SourceDocument:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ReviewError("reviewer는 비어 있을 수 없습니다.")

        document = self.registry.get(document_id)
        errors = self._approval_errors(document)
        if errors:
            raise ReviewError("승인할 수 없습니다: " + "; ".join(errors))

        occurred_at = self.now().isoformat()
        notes = verification_notes.strip() or (
            "필수 메타데이터, 원본 URL, 출처 기관, 저작권 및 이용 조건을 사람이 검수함."
        )
        updated = replace(
            document,
            review_status=ReviewStatus.REVIEWED,
            reviewed_by=reviewer,
            reviewed_at=occurred_at,
            verification_notes=notes,
        )
        post_errors = validate_source_document(updated)
        if post_errors:
            raise ReviewError("승인 결과가 유효하지 않습니다: " + "; ".join(post_errors))

        self.registry.update(updated)
        self._record(document, updated, "approve", reviewer, notes, occurred_at)
        return updated

    def reject(self, document_id: str, reviewer: str, reason: str) -> SourceDocument:
        reviewer = reviewer.strip()
        reason = reason.strip()
        if not reviewer:
            raise ReviewError("reviewer는 비어 있을 수 없습니다.")
        if not reason:
            raise ReviewError("거절 사유는 비어 있을 수 없습니다.")

        document = self.registry.get(document_id)
        occurred_at = self.now().isoformat()
        updated = replace(
            document,
            review_status=ReviewStatus.REJECTED,
            reviewed_by=reviewer,
            reviewed_at=occurred_at,
            verification_notes=reason,
        )
        self.registry.update(updated)
        self._record(document, updated, "reject", reviewer, reason, occurred_at)
        return updated

    def _approval_errors(self, document: SourceDocument) -> list[str]:
        errors = validate_source_document(document)
        for field_name in REVIEW_REQUIRED_FIELDS:
            if not str(getattr(document, field_name)).strip():
                errors.append(f"승인 필수 메타데이터가 없습니다: {field_name}")

        source_url = urlsplit(document.source_url)
        if source_url.scheme not in {"http", "https"} or not source_url.hostname:
            errors.append("원본 URL은 유효한 HTTP(S) 기관 URL이어야 합니다.")
        if not document.publisher.strip():
            errors.append("출처 기관을 확인할 수 없습니다.")

        errors.extend(validate_local_path(document, self.raw_root))
        if document.copyright_status in {
            CopyrightStatus.UNKNOWN,
            CopyrightStatus.RESTRICTED,
        }:
            errors.append(
                "copyright_status가 unknown 또는 restricted인 문서는 승인할 수 없습니다."
            )
        if document.attribution_required and not document.attribution_text.strip():
            errors.append("출처 표시가 필요한 문서에는 attribution_text가 필요합니다.")
        return list(dict.fromkeys(errors))

    def _record(
        self,
        previous: SourceDocument,
        updated: SourceDocument,
        action: str,
        reviewer: str,
        reason: str,
        occurred_at: str,
    ) -> None:
        self.audit_log.append(
            ReviewAuditEntry(
                document_id=updated.document_id,
                action=action,
                reviewer=reviewer,
                occurred_at=occurred_at,
                previous_status=previous.review_status.value,
                new_status=updated.review_status.value,
                reason=reason,
                copyright_status=updated.copyright_status.value,
                allowed_for_rag=updated.allowed_for_rag,
                allowed_for_training=updated.allowed_for_training,
                rag_index_eligible=can_index_for_service(updated),
            )
        )
