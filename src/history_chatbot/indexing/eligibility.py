"""문서 메타데이터만으로 판단하는 보수적인 RAG 허용 정책."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from history_chatbot.ingestion.license_policy import license_policy_errors
from history_chatbot.ingestion.models import (
    CopyrightStatus,
    ReviewStatus,
    SourceDocument,
)


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]


class RagEligibilityPolicy:
    """사람 검수와 권리 허용이 모두 확인된 문서만 통과시킨다."""

    def evaluate(self, document: SourceDocument) -> EligibilityDecision:
        reasons: list[str] = []
        if document.review_status != ReviewStatus.REVIEWED:
            reasons.append(f"review_status={document.review_status.value}")
        if not document.allowed_for_rag:
            reasons.append("allowed_for_rag=false")
        if document.source_reliability not in {"A", "B"}:
            reasons.append(f"source_reliability={document.source_reliability or 'missing'}")
        for field_name in ("document_id", "title", "publisher", "source_url"):
            if not str(getattr(document, field_name)).strip():
                reasons.append(f"필수 출처 정보 누락: {field_name}")

        source_url = urlsplit(document.source_url)
        if document.source_url and (
            source_url.scheme not in {"http", "https"} or not source_url.hostname
        ):
            reasons.append("유효하지 않은 source_url")
        if document.copyright_status in {
            CopyrightStatus.UNKNOWN,
            CopyrightStatus.RESTRICTED,
        }:
            reasons.append(f"copyright_status={document.copyright_status.value}")
        reasons.extend(license_policy_errors(document))
        if document.attribution_required and not document.attribution_text.strip():
            reasons.append("attribution_text 누락")
        if not document.reviewed_by.strip():
            reasons.append("reviewed_by 누락")
        if not document.reviewed_at.strip():
            reasons.append("reviewed_at 누락")
        if not document.verification_notes.strip():
            reasons.append("verification_notes 누락")
        return EligibilityDecision(not reasons, tuple(dict.fromkeys(reasons)))
