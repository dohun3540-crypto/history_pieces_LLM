import hashlib
import json
import sys
from datetime import datetime, timezone

import pytest

from history_chatbot.ingestion import cli as ingestion_cli
from history_chatbot.ingestion.models import CopyrightStatus, ReviewStatus
from history_chatbot.ingestion.review import (
    ReviewAuditLog,
    ReviewError,
    ReviewService,
)
from history_chatbot.ingestion.source_registry import SourceRegistry
from history_chatbot.ingestion.validator import can_index_for_service


FIXED_TIME = datetime(2026, 7, 30, 15, 30, tzinfo=timezone.utc)


def review_service(tmp_path, source_factory, **overrides):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_path = raw_root / "source.txt"
    raw_path.write_text("원본은 변경하지 않습니다.", encoding="utf-8")
    source = source_factory(raw_path, **overrides)
    registry = SourceRegistry(tmp_path / "sources.jsonl")
    registry.register(source)
    service = ReviewService(
        registry,
        ReviewAuditLog(tmp_path / "review_audit.jsonl"),
        raw_root,
        now=lambda: FIXED_TIME,
    )
    return service, registry, raw_path, tmp_path / "review_audit.jsonl"


def test_approve_records_reviewer_time_notes_and_audit(tmp_path, source_factory) -> None:
    service, registry, raw_path, audit_path = review_service(tmp_path, source_factory)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    approved = service.approve("test-virtual-001", "검수자", "출처와 권리 확인 완료")

    assert approved.review_status == ReviewStatus.REVIEWED
    assert approved.reviewed_by == "검수자"
    assert approved.reviewed_at == FIXED_TIME.isoformat()
    assert approved.verification_notes == "출처와 권리 확인 완료"
    assert registry.get(approved.document_id) == approved
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before_hash

    entries = ReviewAuditLog(audit_path).list()
    assert len(entries) == 1
    assert entries[0].action == "approve"
    assert entries[0].previous_status == "draft"
    assert entries[0].new_status == "reviewed"
    assert entries[0].reviewer == "검수자"


@pytest.mark.parametrize(
    "copyright_status",
    [CopyrightStatus.UNKNOWN, CopyrightStatus.RESTRICTED],
)
def test_unknown_or_restricted_document_cannot_be_approved(
    tmp_path, source_factory, copyright_status
) -> None:
    service, registry, _, audit_path = review_service(
        tmp_path,
        source_factory,
        copyright_status=copyright_status,
        allowed_for_rag=False,
        allowed_for_training=False,
    )

    with pytest.raises(ReviewError, match="승인할 수 없습니다"):
        service.approve("test-virtual-001", "검수자")

    assert registry.get("test-virtual-001").review_status == ReviewStatus.DRAFT
    assert not audit_path.exists()


def test_required_attribution_text_blocks_approval(tmp_path, source_factory) -> None:
    service, _, _, _ = review_service(
        tmp_path,
        source_factory,
        attribution_required=True,
        attribution_text="",
    )

    with pytest.raises(ReviewError, match="attribution_text"):
        service.approve("test-virtual-001", "검수자")


def test_approval_does_not_enable_rag_flag(tmp_path, source_factory) -> None:
    service, _, _, audit_path = review_service(
        tmp_path,
        source_factory,
        allowed_for_rag=False,
    )

    approved = service.approve("test-virtual-001", "검수자")

    assert not approved.allowed_for_rag
    assert not can_index_for_service(approved)
    assert not ReviewAuditLog(audit_path).list()[0].rag_index_eligible


def test_missing_required_metadata_blocks_approval(tmp_path, source_factory) -> None:
    service, _, _, _ = review_service(
        tmp_path,
        source_factory,
        license_name="",
    )

    with pytest.raises(ReviewError, match="license_name"):
        service.approve("test-virtual-001", "검수자")


def test_reject_records_reason_and_audit_without_touching_raw(
    tmp_path, source_factory
) -> None:
    service, registry, raw_path, audit_path = review_service(
        tmp_path,
        source_factory,
        copyright_status=CopyrightStatus.UNKNOWN,
        allowed_for_rag=False,
        allowed_for_training=False,
    )
    before = raw_path.read_bytes()

    rejected = service.reject("test-virtual-001", "검수자", "역사 본문이 없음")

    assert rejected.review_status == ReviewStatus.REJECTED
    assert rejected.reviewed_by == "검수자"
    assert rejected.reviewed_at == FIXED_TIME.isoformat()
    assert rejected.verification_notes == "역사 본문이 없음"
    assert registry.get(rejected.document_id) == rejected
    assert raw_path.read_bytes() == before
    entry = ReviewAuditLog(audit_path).list()[0]
    assert entry.action == "reject"
    assert entry.reason == "역사 본문이 없음"
    assert entry.new_status == "rejected"


def test_review_show_cli_outputs_document_without_mutation(
    tmp_path, source_factory, monkeypatch, capsys
) -> None:
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True)
    raw_path = raw_root / "source.txt"
    raw_path.write_text("원본", encoding="utf-8")
    source = source_factory(raw_path)
    manifest = tmp_path / "sources.jsonl"
    SourceRegistry(manifest).register(source)
    before = manifest.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingestion",
            "review",
            "show",
            "--manifest",
            str(manifest),
            "--document-id",
            source.document_id,
        ],
    )
    ingestion_cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["document_id"] == source.document_id
    assert output["review_status"] == "draft"
    assert not output["rag_index_eligible"]
    assert manifest.read_bytes() == before


def test_review_approve_cli_updates_manifest_and_audit(
    tmp_path, source_factory, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "data"
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True)
    raw_path = raw_root / "source.txt"
    raw_path.write_text("원본", encoding="utf-8")
    source = source_factory(raw_path, allowed_for_rag=False)
    manifest = tmp_path / "sources.jsonl"
    audit = tmp_path / "audit.jsonl"
    SourceRegistry(manifest).register(source)
    monkeypatch.setattr(ingestion_cli, "_project_data_root", lambda: data_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingestion",
            "review",
            "approve",
            "--manifest",
            str(manifest),
            "--audit-log",
            str(audit),
            "--document-id",
            source.document_id,
            "--reviewer",
            "검수자",
        ],
    )

    ingestion_cli.main()

    assert "서비스 RAG 색인: 불가" in capsys.readouterr().out
    approved = SourceRegistry(manifest).get(source.document_id)
    assert approved.review_status == ReviewStatus.REVIEWED
    assert not approved.allowed_for_rag
    assert ReviewAuditLog(audit).list()[0].action == "approve"


def test_review_reject_cli_records_reason(
    tmp_path, source_factory, monkeypatch, capsys
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_path = raw_root / "source.txt"
    raw_path.write_text("원본", encoding="utf-8")
    source = source_factory(raw_path)
    manifest = tmp_path / "sources.jsonl"
    audit = tmp_path / "audit.jsonl"
    SourceRegistry(manifest).register(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingestion",
            "review",
            "reject",
            "--manifest",
            str(manifest),
            "--audit-log",
            str(audit),
            "--document-id",
            source.document_id,
            "--reviewer",
            "검수자",
            "--reason",
            "출처 확인 실패",
        ],
    )

    ingestion_cli.main()

    assert "검수 거절 완료" in capsys.readouterr().out
    rejected = SourceRegistry(manifest).get(source.document_id)
    assert rejected.review_status == ReviewStatus.REJECTED
    assert rejected.verification_notes == "출처 확인 실패"
    assert ReviewAuditLog(audit).list()[0].action == "reject"
