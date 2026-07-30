import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from history_chatbot.indexing import cli as indexing_cli
from history_chatbot.indexing.builder import IndexBuilder
from history_chatbot.indexing.loader import ReviewedChunkLoader
from history_chatbot.indexing.manifest import IndexManifest
from history_chatbot.ingestion.models import CopyrightStatus, ReviewStatus
from history_chatbot.ingestion.source_registry import SourceRegistry


FIXED_TIME = datetime(2026, 7, 30, 16, 0, tzinfo=timezone.utc)


def make_index_fixture(tmp_path, source_factory, *, chunks=None, **overrides):
    raw_root = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "data" / "index_ready"
    raw_root.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    raw_path = raw_root / "source.txt"
    raw_path.write_text("테스트용 가상 원문", encoding="utf-8")
    defaults = {
        "review_status": ReviewStatus.REVIEWED,
        "reviewed_by": "검수자",
        "reviewed_at": FIXED_TIME.isoformat(),
        "verification_notes": "원본 URL, 기관, 권리 조건 검수 완료",
        "source_reliability": "A",
        "allowed_for_rag": True,
    }
    defaults.update(overrides)
    source = source_factory(raw_path, **defaults)
    manifest_path = tmp_path / "data" / "manifests" / "sources.jsonl"
    registry = SourceRegistry(manifest_path)
    registry.register(source)
    if chunks is None:
        chunks = [
            {
                "chunk_id": f"{source.document_id}::chunk-0000",
                "document_id": source.document_id,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 10,
                "text": "테스트용 가상 역사 자료 청크",
                "title": source.title,
                "source": source.source_url,
                "page": None,
                "section": None,
                "metadata": {},
            }
        ]
    processed_path = processed_dir / f"{source.document_id}.jsonl"
    with processed_path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    loader = ReviewedChunkLoader(registry, raw_root, processed_dir)
    builder = IndexBuilder(
        loader,
        manifest_path,
        output_dir,
        now=lambda: FIXED_TIME,
    )
    return source, registry, loader, builder, processed_path


def test_only_reviewed_and_rag_allowed_document_passes(tmp_path, source_factory) -> None:
    source, _, loader, _, _ = make_index_fixture(tmp_path, source_factory)

    report = loader.load()

    assert [item.document.document_id for item in report.eligible] == [
        source.document_id
    ]
    assert not report.rejected


@pytest.mark.parametrize("status", [ReviewStatus.DRAFT, ReviewStatus.REJECTED])
def test_draft_and_rejected_are_blocked(tmp_path, source_factory, status) -> None:
    source, _, loader, _, _ = make_index_fixture(
        tmp_path,
        source_factory,
        review_status=status,
    )

    report = loader.load()

    assert not report.eligible
    assert report.rejected[0].document_id == source.document_id
    assert any("review_status" in reason for reason in report.rejected[0].reasons)


@pytest.mark.parametrize(
    "copyright_status",
    [CopyrightStatus.UNKNOWN, CopyrightStatus.RESTRICTED],
)
def test_unknown_and_restricted_copyright_are_blocked(
    tmp_path, source_factory, copyright_status
) -> None:
    _, _, loader, _, _ = make_index_fixture(
        tmp_path,
        source_factory,
        copyright_status=copyright_status,
        allowed_for_rag=False,
        allowed_for_training=False,
    )

    report = loader.load()

    assert not report.eligible
    assert any(
        "copyright_status" in reason for reason in report.rejected[0].reasons
    )


@pytest.mark.parametrize("grade", ["C", "D"])
def test_low_reliability_is_blocked(tmp_path, source_factory, grade) -> None:
    _, _, loader, _, _ = make_index_fixture(
        tmp_path,
        source_factory,
        source_reliability=grade,
    )

    report = loader.load()

    assert not report.eligible
    assert any("source_reliability" in reason for reason in report.rejected[0].reasons)


def test_missing_attribution_is_blocked(tmp_path, source_factory) -> None:
    _, _, loader, _, _ = make_index_fixture(
        tmp_path,
        source_factory,
        attribution_required=True,
        attribution_text="",
    )

    report = loader.load()

    assert not report.eligible
    assert any("attribution_text" in reason for reason in report.rejected[0].reasons)


def test_document_and_chunk_trace_and_metadata_are_preserved(
    tmp_path, source_factory
) -> None:
    source, _, _, builder, _ = make_index_fixture(
        tmp_path,
        source_factory,
        people=["테스트 인물"],
        places=["테스트 장소"],
        period_start=1900,
        period_end=1910,
    )

    result = builder.prepare()
    record = json.loads(Path(result.chunks_path).read_text(encoding="utf-8"))

    assert record["document_id"] == source.document_id
    assert record["chunk_id"] == f"{source.document_id}::chunk-0000"
    assert record["title"] == source.title
    assert record["publisher"] == source.publisher
    assert record["source_url"] == source.source_url
    assert record["people"] == ["테스트 인물"]
    assert record["places"] == ["테스트 장소"]
    assert record["period_start"] == 1900
    assert record["period_end"] == 1910
    assert not builder.validate()


def test_invalid_chunk_document_trace_is_rejected(tmp_path, source_factory) -> None:
    source, _, loader, _, _ = make_index_fixture(tmp_path, source_factory)
    processed_path = (
        tmp_path / "data" / "processed" / f"{source.document_id}.jsonl"
    )
    record = json.loads(processed_path.read_text(encoding="utf-8"))
    record["document_id"] = "another-document"
    processed_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = loader.load()

    assert not report.eligible
    assert any("추적 관계" in reason for reason in report.rejected[0].reasons)


def test_duplicate_chunk_content_is_removed(tmp_path, source_factory) -> None:
    document_id = "test-virtual-001"
    chunks = [
        {
            "chunk_id": f"{document_id}::chunk-0000",
            "document_id": document_id,
            "chunk_index": 0,
            "text": "동일한 청크",
        },
        {
            "chunk_id": f"{document_id}::chunk-0001",
            "document_id": document_id,
            "chunk_index": 1,
            "text": "동일한   청크",
        },
    ]
    _, _, _, builder, _ = make_index_fixture(
        tmp_path,
        source_factory,
        chunks=chunks,
    )

    result = builder.prepare()

    assert result.chunk_count == 1
    assert result.duplicate_chunks == 1


def test_empty_data_is_successful_and_cli_reports_clear_message(
    tmp_path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "data"
    manifest = data_root / "manifests" / "sources.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "indexing",
            "prepare",
            "--manifest",
            str(manifest),
            "--raw-root",
            str(data_root / "raw"),
            "--processed-dir",
            str(data_root / "processed"),
            "--output-dir",
            str(data_root / "index_ready"),
        ],
    )

    indexing_cli.main()

    output = capsys.readouterr().out
    assert "현재 인덱싱 가능한 검수 완료 문서가 없습니다" in output
    assert (data_root / "index_ready" / "chunks.jsonl").read_text() == ""
    saved = IndexManifest.load(data_root / "index_ready" / "index_manifest.json")
    assert saved.stats["chunks"] == 0


def test_incremental_state_and_tombstone(tmp_path, source_factory) -> None:
    source, registry, _, builder, _ = make_index_fixture(tmp_path, source_factory)

    first = builder.prepare()
    second = builder.prepare()
    registry.update(replace(source, review_status=ReviewStatus.REJECTED))
    third = builder.prepare()
    manifest = IndexManifest.load(builder.manifest_path)

    assert first.changed_document_ids == (source.document_id,)
    assert second.unchanged_document_ids == (source.document_id,)
    assert third.tombstone_document_ids == (source.document_id,)
    assert [item.document_id for item in manifest.tombstones] == [source.document_id]
    assert builder.chunks_path.read_text(encoding="utf-8") == ""
