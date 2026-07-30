"""등록 메타데이터와 서비스 색인 가능 여부 검증."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from history_chatbot.ingestion.license_policy import license_policy_errors
from history_chatbot.ingestion.models import ReviewStatus, SourceDocument


def validate_source_document(document: SourceDocument) -> list[str]:
    errors: list[str] = []
    for name in (
        "document_id",
        "title",
        "source_type",
        "publisher",
        "local_path",
        "language",
    ):
        if not str(getattr(document, name)).strip():
            errors.append(f"{name}은(는) 필수입니다.")
    if not document.source_url.strip() and not document.publisher.strip():
        errors.append("source_url 또는 publisher가 필요합니다.")
    for field_name in ("published_date", "accessed_date"):
        value = getattr(document, field_name)
        if value:
            try:
                date.fromisoformat(value)
            except ValueError:
                errors.append(f"{field_name}은 YYYY-MM-DD 형식이어야 합니다.")
    if document.period_start and document.period_end:
        if document.period_start > document.period_end:
            errors.append("period_start는 period_end보다 클 수 없습니다.")
    if document.review_status == ReviewStatus.REVIEWED:
        if not document.reviewed_by.strip():
            errors.append("reviewed 상태에는 reviewed_by가 필요합니다.")
        if not document.reviewed_at.strip():
            errors.append("reviewed 상태에는 reviewed_at이 필요합니다.")
        else:
            try:
                datetime.fromisoformat(document.reviewed_at)
            except ValueError:
                errors.append("reviewed_at은 ISO 8601 형식이어야 합니다.")
    errors.extend(license_policy_errors(document))
    return errors


def validate_local_path(document: SourceDocument, raw_root: Path) -> list[str]:
    candidate = Path(document.local_path)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(raw_root.resolve())
    except ValueError:
        return ["local_path는 data/raw 아래에 있어야 합니다."]
    return [] if resolved.is_file() else [f"원문 파일을 찾을 수 없습니다: {candidate}"]


def can_index_for_service(document: SourceDocument) -> bool:
    return (
        document.review_status == ReviewStatus.REVIEWED
        and document.allowed_for_rag
        and document.source_reliability in {"", "A", "B"}
        and not validate_source_document(document)
    )
