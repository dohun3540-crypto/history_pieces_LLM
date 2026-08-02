import ast
import hashlib
import json
from pathlib import Path, PureWindowsPath

from scripts.audit_history_data import TOPICS, audit_repository, main


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")


def record(document_id="doc-1", **overrides):
    item = {
        "document_id": document_id,
        "title": "목포 개항과 해관",
        "publisher": "기관 A",
        "source_url": "https://example.invalid/1",
        "local_path": "data/raw/doc-1.txt",
        "copyright_status": "unknown",
        "review_status": "draft",
        "production_approved": False,
    }
    item.update(overrides)
    return item


def prepare(tmp_path, records=None, chunks=None):
    write_jsonl(tmp_path / "data/manifests/sources.jsonl", records or [record()])
    write_jsonl(tmp_path / "data/index_ready/chunks.jsonl", chunks or [])
    return audit_repository(tmp_path)


def test_fixture_is_excluded_from_actual_history_count(tmp_path):
    write_jsonl(tmp_path / "data/manifests/sources.jsonl", [record()])
    write_jsonl(tmp_path / "tests/fixtures/rag/fictional_documents.jsonl", [{"document_id": "fixture-1", "data_classification": "fictional_fixture"}])
    report = audit_repository(tmp_path)
    assert report["counts"]["actual_documents_excluding_fixtures"] == 1
    assert report["counts"]["fixture_documents"] == 1


def test_status_counts_are_conservative(tmp_path):
    records = [record("draft"), record("rejected", review_status="rejected"), record("approved", production_approved=True, copyright_status="open_license")]
    report = prepare(tmp_path, records)
    assert report["status_counts"]["draft"] == 1
    assert report["status_counts"]["production_rejected"] == 1
    assert report["status_counts"]["production_approved"] == 1


def test_missing_fields_are_detected(tmp_path):
    report = prepare(tmp_path, [{"document_id": "only-id"}])
    assert report["missing_field_counts"]["title"] == 1
    assert report["missing_field_counts"]["license_status"] == 1


def test_duplicate_document_id_is_detected(tmp_path):
    report = prepare(tmp_path, [record(), record(source_url="https://example.invalid/2")])
    assert report["duplicates"]["document_id"][0]["value"] == "doc-1"


def test_duplicate_source_url_is_detected(tmp_path):
    report = prepare(tmp_path, [record("a"), record("b")])
    assert report["duplicates"]["source_url"][0]["count"] == 2


def test_raw_manifest_mismatch_is_detected(tmp_path):
    raw = tmp_path / "data/raw/orphan.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("orphan", encoding="utf-8")
    report = prepare(tmp_path)
    assert "doc-1" in report["mismatches"]["manifest_missing_raw_file"]
    assert "data/raw/orphan.txt" in report["mismatches"]["raw_without_manifest"]


def test_chunk_document_mismatch_is_detected(tmp_path):
    report = prepare(tmp_path, chunks=[{"document_id": "orphan", "chunk_id": "orphan::1", "text": "text"}])
    assert report["mismatches"]["chunks_without_manifest_document"] == ["orphan"]


def test_topic_classification_uses_metadata_and_text(tmp_path):
    chunks = [{"document_id": "doc-1", "chunk_id": "doc-1::1", "text": "목포 해관과 외국인 거류지를 설명한다."}]
    report = prepare(tmp_path, [record(title="관련 자료", keywords=["목포 개항"])], chunks)
    coverage = {item["topic"]: item for item in report["topic_coverage"]}
    assert coverage["목포 개항"]["document_count"] == 1
    assert coverage["목포 해관"]["chunk_count"] == 1
    assert len(coverage) == len(TOPICS)


def test_empty_and_short_body_are_detected(tmp_path):
    report = prepare(tmp_path, [record("empty"), record("short")], [{"document_id": "short", "chunk_id": "short::1", "text": "짧은 글"}])
    warnings = {item["document_id"]: item["warnings"] for item in report["quality_warnings"]}
    assert "empty_body" in warnings["empty"]
    assert "short_body" in warnings["short"]


def test_cli_does_not_modify_source_data(tmp_path):
    manifest = tmp_path / "data/manifests/sources.jsonl"
    write_jsonl(manifest, [record()])
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert main(["--root", str(tmp_path), "--json-output", "reports/a.json", "--markdown-output", "reports/a.md"]) == 0
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before


def test_windows_paths_are_normalized_for_lane_detection(tmp_path):
    win = PureWindowsPath("tests", "fixtures", "rag", "fictional_documents.jsonl")
    assert "tests\\fixtures" in str(win)
    write_jsonl(tmp_path / Path(*win.parts), [{"document_id": "fixture-win", "is_fixture": True}])
    report = audit_repository(tmp_path)
    assert report["counts"]["fixture_documents"] == 1


def test_script_parses_as_python_38():
    source = (Path(__file__).parents[1] / "scripts/audit_history_data.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))
