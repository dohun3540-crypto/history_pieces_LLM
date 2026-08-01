from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from history_chatbot.chat.citation_builder import build_citations
from history_chatbot.indexing.eligibility import RagEligibilityPolicy
from history_chatbot.ingestion.development import DevelopmentManifestLoader
from history_chatbot.ingestion.models import CopyrightStatus, ReviewStatus, SourceDocument
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.retrieval.service import (
    DevelopmentRealReader,
    IndexReadyReader,
    RetrievalConfig,
)
from history_chatbot.runtime import RuntimeMode


def development_record(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "document_id": "dev-real-001",
        "chunk_id": "dev-real-001::0000",
        "text": "공식 원문에서 확인한 개발 검증용 역사 문장",
        "title": "공식 역사 자료",
        "publisher": "공식기관",
        "source_url": "https://example.go.kr/history/1",
        "canonical_source_url": "https://example.go.kr/history/1",
        "accessed_date": "2026-08-02",
        "language": "ko",
        "primary_topic": "urban_formation",
        "factual_summary": "공식 원문의 단일 사실을 요약했다.",
        "evidence_quote": "공식 원문에서 확인한 근거 문장",
        "citation_title": "공식 역사 자료",
        "citation_url": "https://example.go.kr/history/1",
        "approval_tier": "development_approved",
        "data_classification": "real_historical_source",
        "is_fixture": False,
        "development_only": True,
        "production_approved": False,
        "public_release_allowed": False,
        "license_review_status": "pending_review",
        "raw_source_status": "remote_only",
        "review_status": "verified_pending_production_review",
        "source_status": "development_only",
        "source_reliability": "A",
        "retrieval_subjects": ["공식 역사 자료"],
        "development_approved_by": "human-reviewer",
        "development_approved_at": "2026-08-02",
        "development_approval_notes": "격리된 development 통합 검증만 승인",
        "badge_label": "개발 검증용 자료",
        "usage_notice": "실제 역사 자료이나 production 공개 승인을 받지 않았습니다.",
    }
    record.update(updates)
    return record


def manifest_report(
    record: dict[str, object], *, mode: RuntimeMode = RuntimeMode.TEST
):
    loader = DevelopmentManifestLoader(Path("unused.jsonl"), runtime_mode=mode)
    return loader.evaluate([record])


def test_pending_review_is_blocked_by_default() -> None:
    report = manifest_report(
        development_record(
            approval_tier="development_pending_review",
            development_approved_by="",
            development_approved_at="",
            development_approval_notes="",
        )
    )
    assert not report.approved
    assert "development_pending_review" in report.rejected[0][1]


def test_pending_review_requires_explicit_document_selection() -> None:
    loader = DevelopmentManifestLoader(
        Path("unused.jsonl"),
        runtime_mode=RuntimeMode.TEST,
        explicitly_selected_pending_ids=("dev-real-001",),
    )
    report = loader.evaluate(
        [
            development_record(
                approval_tier="development_pending_review",
                development_approved_by="",
                development_approved_at="",
                development_approval_notes="",
            )
        ]
    )
    assert [item.document_id for item in report.approved] == ["dev-real-001"]


def test_approved_document_loads_in_development() -> None:
    report = manifest_report(development_record(), mode=RuntimeMode.DEVELOPMENT)
    assert [item.document_id for item in report.approved] == ["dev-real-001"]


def test_development_loader_is_blocked_in_production() -> None:
    with pytest.raises(ValueError, match="development/test"):
        DevelopmentManifestLoader(
            Path("unused.jsonl"), runtime_mode=RuntimeMode.PRODUCTION
        )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"data_classification": "fictional_fixture"}, "data_classification"),
        ({"is_fixture": True}, "is_fixture"),
        ({"canonical_source_url": ""}, "canonical_source_url"),
        ({"evidence_quote": ""}, "evidence_quote"),
        ({"public_release_allowed": True}, "public_release_allowed"),
        ({"production_approved": True}, "production_approved"),
        ({"source_reliability": "C"}, "source_reliability"),
        ({"is_fixture": 0}, "is_fixture"),
    ],
)
def test_manifest_rejects_unsafe_metadata(
    updates: dict[str, object], reason: str
) -> None:
    report = manifest_report(development_record(**updates))
    assert not report.approved
    assert any(reason in item for item in report.rejected[0][1])


def test_retrieval_config_rejects_development_lane_in_production() -> None:
    with pytest.raises(ValueError, match="development_real"):
        RetrievalConfig(
            runtime_mode="production",
            development_chunks_path=Path(
                "data/development_real/index_ready/chunks.jsonl"
            ),
        ).validate()


def test_retrieval_config_rejects_fixture_and_real_lane_mix() -> None:
    with pytest.raises(ValueError, match="혼합"):
        RetrievalConfig(
            runtime_mode="development",
            fixture_chunks_path=Path("fixture.jsonl"),
            development_chunks_path=Path("real.jsonl"),
        ).validate()


def test_development_retrieval_config_loads_isolated_paths() -> None:
    config = RetrievalConfig.load(Path("configs/retrieval.development-real.yaml"))
    assert config.runtime_mode == "development"
    assert config.development_chunks_path == Path(
        "data/development_real/index_ready/chunks.jsonl"
    )
    assert config.local_storage_path == Path("data/development_real/retrieval_index")


def test_development_reader_rejects_pending_and_fixture_records() -> None:
    pending = DevelopmentRealReader._validation_errors(
        development_record(approval_tier="development_pending_review")
    )
    fixture = DevelopmentRealReader._validation_errors(
        development_record(data_classification="fictional_fixture", is_fixture=True)
    )
    assert "development_pending_review" in pending
    assert any("fictional_fixture" in item or "is_fixture" in item for item in fixture)


def test_development_reader_loads_approved_chunk_and_forces_warning() -> None:
    record = development_record()
    record.pop("badge_label")
    record.pop("usage_notice")

    class MemoryPath:
        def is_file(self) -> bool:
            return True

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps(record, ensure_ascii=False)

    chunks, snapshot = DevelopmentRealReader(
        MemoryPath(), RuntimeMode.TEST  # type: ignore[arg-type]
    ).load()
    assert snapshot
    assert chunks[0].payload["badge_label"] == "개발 검증용 자료"
    assert chunks[0].payload["production_approved"] is False


def test_missing_required_field_is_rejected() -> None:
    record = development_record()
    del record["title"]
    report = manifest_report(record)
    assert "missing_required_metadata:title" in report.rejected[0][1]


def test_production_policy_still_rejects_non_reviewed_documents() -> None:
    document = SourceDocument(
        document_id="prod-check",
        title="title",
        source_type="webpage",
        publisher="publisher",
        author="",
        source_url="https://example.go.kr/1",
        local_path="data/raw/prod-check.html",
        published_date="",
        accessed_date="2026-08-02",
        language="ko",
        license_name="review pending",
        license_url="https://example.go.kr/license",
        copyright_status=CopyrightStatus.UNKNOWN,
        allowed_for_rag=False,
        allowed_for_training=False,
        redistribution_allowed=False,
        attribution_required=False,
        attribution_text="",
        notes="",
        review_status=ReviewStatus.METADATA_ADDED,
        reviewed_by="",
        reviewed_at="",
    )
    decision = RagEligibilityPolicy().evaluate(document)
    assert not decision.eligible
    assert "allowed_for_rag=false" in decision.reasons


def test_production_reader_rejects_development_metadata() -> None:
    errors = IndexReadyReader._development_lane_errors(development_record())
    assert "development_only=true" in errors
    assert "production_approved=false" in errors


def test_citation_exposes_optional_development_warning_metadata() -> None:
    record = development_record(
        source_id="source-dev-1",
        copyright_status="unknown",
    )
    ranked = RankedChunk(
        RetrievalChunk.from_record(record), score=0.9, methods=("bm25",)
    )
    citation = asdict(build_citations([ranked])[0])
    assert citation["source_status"] == "development_only"
    assert citation["approval_tier"] == "development_approved"
    assert citation["production_approved"] is False
    assert citation["badge_label"] == "개발 검증용 자료"
    assert "production 공개 승인" in citation["usage_notice"]
