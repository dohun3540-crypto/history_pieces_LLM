"""명시적인 저작권·이용 조건 기본 정책."""

from history_chatbot.ingestion.models import CopyrightStatus, SourceDocument


def license_policy_errors(document: SourceDocument) -> list[str]:
    errors: list[str] = []
    if document.copyright_status in {
        CopyrightStatus.UNKNOWN,
        CopyrightStatus.RESTRICTED,
    }:
        if document.allowed_for_rag:
            errors.append("unknown 또는 restricted 자료는 RAG 사용을 허용할 수 없습니다.")
        if document.allowed_for_training:
            errors.append("unknown 또는 restricted 자료는 학습 사용을 허용할 수 없습니다.")
    if document.attribution_required and not document.attribution_text.strip():
        errors.append("출처 표시가 필요한 자료에는 attribution_text가 필요합니다.")
    return errors


def can_use_for_rag(document: SourceDocument) -> bool:
    return document.allowed_for_rag and not license_policy_errors(document)


def can_use_for_training(document: SourceDocument) -> bool:
    return document.allowed_for_training and not license_policy_errors(document)
