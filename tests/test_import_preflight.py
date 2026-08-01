import json
from pathlib import Path

from history_chatbot.indexing.preflight import (
    PreflightStatus,
    evaluate_record,
    parse_jsonl,
    render_summary,
    PreflightReport,
)


def complete_record(**overrides) -> dict[str, object]:
    value: dict[str, object] = {
        "document_id": "mokpo-reviewed-001",
        "title": "Reviewed source",
        "source_type": "official_webpage",
        "publisher": "Official institution",
        "author": "",
        "source_url": "https://official.example/source",
        "local_path": "data/raw/source.html",
        "published_date": "",
        "accessed_date": "2026-08-02",
        "language": "ko",
        "license_name": "KOGL Type 1",
        "license_url": "https://license.example/type1",
        "copyright_status": "open_license",
        "allowed_for_rag": True,
        "allowed_for_training": False,
        "redistribution_allowed": True,
        "attribution_required": True,
        "attribution_text": "Official institution",
        "notes": "reviewed",
        "review_status": "reviewed",
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-08-02T00:00:00+09:00",
        "source_reliability": "A",
        "verification_notes": "Original source checked",
    }
    value.update(overrides)
    return value


def assess(record: dict[str, object], *, raw_exists: bool = True):
    return evaluate_record(
        record,
        source_file="fixture.jsonl",
        raw_root=Path("data/raw"),
        path_is_file=lambda _path: raw_exists,
    )


def test_complete_reviewed_document_is_eligible() -> None:
    result = assess(complete_record())
    assert result.eligible
    assert result.statuses == (PreflightStatus.ELIGIBLE.value,)


def test_missing_rag_and_review_metadata_are_reported_without_filling() -> None:
    record = complete_record()
    for field in ("allowed_for_rag", "reviewed_by", "reviewed_at", "verification_notes"):
        record.pop(field)
    result = assess(record)
    assert PreflightStatus.MISSING_REQUIRED_METADATA.value in result.statuses
    assert set(result.missing_fields) >= {
        "allowed_for_rag", "reviewed_by", "reviewed_at", "verification_notes"
    }
    assert "allowed_for_rag" in result.invalid_fields


def test_missing_raw_source_is_blocked() -> None:
    result = assess(complete_record(), raw_exists=False)
    assert PreflightStatus.MISSING_RAW_SOURCE.value in result.statuses
    assert not result.eligible


def test_missing_copyright_and_kogl_type_four_require_license_review() -> None:
    missing = assess(complete_record(copyright_status=""))
    assert PreflightStatus.LICENSE_REVIEW_REQUIRED.value in missing.statuses
    kogl4 = assess(complete_record(license_name="KOGL Type 4"))
    assert PreflightStatus.LICENSE_REVIEW_REQUIRED.value in kogl4.statuses
    assert not kogl4.eligible


def test_verified_external_shape_is_not_auto_converted() -> None:
    external = {
        "id": "mokpo_hist_0004",
        "title": "External verified record",
        "review_status": "verified",
        "source": {
            "url": "https://official.example/source",
            "license": "KOGL Type 4",
        },
    }
    result = assess(external, raw_exists=False)
    assert PreflightStatus.SCHEMA_MAPPING_REQUIRED.value in result.statuses
    assert PreflightStatus.INCOMPATIBLE_REVIEW_STATUS.value in result.statuses
    assert result.record_id == "mokpo_hist_0004"
    assert external["review_status"] == "verified"


def test_invalid_source_url_is_reported() -> None:
    result = assess(complete_record(source_url="not-a-url"))
    assert PreflightStatus.MISSING_REQUIRED_METADATA.value in result.statuses
    assert "source_url" in result.invalid_fields


def test_json_and_human_reports_contain_only_structured_findings() -> None:
    records = parse_jsonl(
        [json.dumps(complete_record())], source_file="fixture.jsonl"
    )
    item = assess(records[0])
    report = PreflightReport("fixture", 1, 1, (item,))
    payload = report.to_dict()
    assert payload["dry_run"] is True
    assert payload["eligible_records"] == 1
    assert "eligible: 1" in render_summary(report)
