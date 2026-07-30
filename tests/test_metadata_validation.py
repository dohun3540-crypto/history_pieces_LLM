import pytest

from history_chatbot.ingestion.models import CopyrightStatus, ReviewStatus
from history_chatbot.ingestion.pipeline import copy_reviewed_output
from history_chatbot.ingestion.validator import (
    can_index_for_service,
    validate_source_document,
)


def test_non_reviewed_document_cannot_be_indexed(tmp_path, source_factory) -> None:
    source = source_factory(tmp_path / "virtual.txt", review_status=ReviewStatus.CLEANED)
    assert not can_index_for_service(source)


def test_invalid_metadata_fails_validation(tmp_path, source_factory) -> None:
    source = source_factory(
        tmp_path / "virtual.txt",
        title="",
        accessed_date="30-07-2026",
        copyright_status=CopyrightStatus.UNKNOWN,
        allowed_for_rag=True,
    )
    errors = validate_source_document(source)
    assert any("title" in error for error in errors)
    assert any("accessed_date" in error for error in errors)
    assert any("RAG" in error for error in errors)


def test_reviewed_valid_document_can_be_indexed(tmp_path, source_factory) -> None:
    source = source_factory(
        tmp_path / "virtual.txt",
        review_status=ReviewStatus.REVIEWED,
        reviewed_by="검수자",
        reviewed_at="2026-07-30T15:00:00+09:00",
    )
    assert can_index_for_service(source)


def test_non_reviewed_document_cannot_be_promoted(tmp_path, source_factory) -> None:
    processed = tmp_path / "processed.jsonl"
    processed.write_text("{}\n", encoding="utf-8")
    source = source_factory(tmp_path / "virtual.txt")
    with pytest.raises(ValueError, match="검수 완료"):
        copy_reviewed_output(processed, tmp_path / "reviewed", source)
