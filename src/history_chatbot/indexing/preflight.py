"""Non-destructive preflight for mapping an external history corpus to production."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit


class PreflightStatus(StrEnum):
    ELIGIBLE = "eligible"
    MISSING_REQUIRED_METADATA = "missing_required_metadata"
    MISSING_RAW_SOURCE = "missing_raw_source"
    INCOMPATIBLE_REVIEW_STATUS = "incompatible_review_status"
    LICENSE_REVIEW_REQUIRED = "license_review_required"
    SCHEMA_MAPPING_REQUIRED = "schema_mapping_required"


MAIN_REQUIRED_FIELDS = (
    "document_id",
    "title",
    "source_type",
    "publisher",
    "author",
    "source_url",
    "local_path",
    "published_date",
    "accessed_date",
    "language",
    "license_name",
    "license_url",
    "copyright_status",
    "allowed_for_rag",
    "allowed_for_training",
    "redistribution_allowed",
    "attribution_required",
    "attribution_text",
    "notes",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "source_reliability",
    "verification_notes",
)


@dataclass(frozen=True, slots=True)
class DocumentPreflight:
    source_file: str
    record_id: str
    title: str
    statuses: tuple[str, ...]
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.statuses == (PreflightStatus.ELIGIBLE.value,)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    repository: str
    records_scanned: int
    eligible_records: int
    documents: tuple[DocumentPreflight, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dry_run": True,
            "repository": self.repository,
            "records_scanned": self.records_scanned,
            "eligible_records": self.eligible_records,
            "documents": [asdict(item) | {"eligible": item.eligible} for item in self.documents],
        }


def evaluate_record(
    record: dict[str, object],
    *,
    source_file: str,
    raw_root: Path,
    path_is_file: Callable[[Path], bool] | None = None,
) -> DocumentPreflight:
    """Evaluate without mutating or translating the supplied record."""

    is_file = path_is_file or Path.is_file
    statuses: list[str] = []
    invalid: list[str] = []
    notes: list[str] = []
    external_shape = "id" in record or isinstance(record.get("source"), dict)
    if external_shape:
        statuses.append(PreflightStatus.SCHEMA_MAPPING_REQUIRED.value)
        notes.append("external record fields are not auto-mapped to SourceDocument")

    missing = [name for name in MAIN_REQUIRED_FIELDS if name not in record]
    if missing:
        statuses.append(PreflightStatus.MISSING_REQUIRED_METADATA.value)

    record_id = str(record.get("document_id") or record.get("id") or "")
    title = str(record.get("title") or "")
    if not record_id:
        invalid.append("document_id")
    if not title.strip():
        invalid.append("title")

    source_url = str(record.get("source_url") or "")
    parts = urlsplit(source_url)
    if not source_url or parts.scheme not in {"http", "https"} or not parts.hostname:
        invalid.append("source_url")

    if record.get("allowed_for_rag") is not True:
        invalid.append("allowed_for_rag")
    if str(record.get("source_reliability") or "") not in {"A", "B"}:
        invalid.append("source_reliability")
    for field in ("reviewed_by", "reviewed_at", "verification_notes"):
        if not str(record.get(field) or "").strip():
            invalid.append(field)
    if invalid and PreflightStatus.MISSING_REQUIRED_METADATA.value not in statuses:
        statuses.append(PreflightStatus.MISSING_REQUIRED_METADATA.value)

    if record.get("review_status") != "reviewed":
        statuses.append(PreflightStatus.INCOMPATIBLE_REVIEW_STATUS.value)
        notes.append("review status is reported as-is; verified is not converted to reviewed")

    license_name = str(record.get("license_name") or "")
    copyright_status = str(record.get("copyright_status") or "")
    if (
        not license_name
        or copyright_status in {"", "unknown", "restricted"}
        or "KOGL Type 4" in license_name
        or "공공누리 제4유형" in license_name
    ):
        statuses.append(PreflightStatus.LICENSE_REVIEW_REQUIRED.value)
        if "KOGL Type 4" in license_name or "공공누리 제4유형" in license_name:
            notes.append("KOGL Type 4 is never auto-approved for RAG transformation")

    local_path = str(record.get("local_path") or "")
    raw_missing = not local_path
    if local_path:
        candidate = Path(local_path)
        try:
            candidate.resolve().relative_to(raw_root.resolve())
        except ValueError:
            raw_missing = True
            invalid.append("local_path_outside_raw_root")
        else:
            raw_missing = not is_file(candidate)
    if raw_missing:
        statuses.append(PreflightStatus.MISSING_RAW_SOURCE.value)

    resolved_statuses = tuple(dict.fromkeys(statuses))
    if not resolved_statuses:
        resolved_statuses = (PreflightStatus.ELIGIBLE.value,)
    return DocumentPreflight(
        source_file=source_file,
        record_id=record_id,
        title=title,
        statuses=resolved_statuses,
        missing_fields=tuple(missing),
        invalid_fields=tuple(dict.fromkeys(invalid)),
        notes=tuple(notes),
    )


def parse_jsonl(lines: Iterable[str], *, source_file: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source_file}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def inspect_repository(root: Path, *, include_provisional: bool = False) -> PreflightReport:
    resolved_root = root.resolve()
    groups = [resolved_root / "data" / "records"]
    if include_provisional:
        groups.append(resolved_root / "data" / "provisional")
    documents: list[DocumentPreflight] = []
    for directory in groups:
        for path in sorted(directory.glob("*.jsonl")):
            records = parse_jsonl(path.read_text(encoding="utf-8").splitlines(), source_file=str(path))
            documents.extend(
                evaluate_record(
                    record,
                    source_file=str(path),
                    raw_root=resolved_root / "data" / "raw",
                )
                for record in records
            )
    return PreflightReport(
        repository=str(resolved_root),
        records_scanned=len(documents),
        eligible_records=sum(item.eligible for item in documents),
        documents=tuple(documents),
    )


def render_summary(report: PreflightReport) -> str:
    lines = [
        "Production import preflight (dry-run)",
        f"repository: {report.repository}",
        f"records: {report.records_scanned}",
        f"eligible: {report.eligible_records}",
    ]
    for item in report.documents:
        lines.append(
            f"- {item.record_id or '<missing-id>'}: {', '.join(item.statuses)}"
        )
        if item.missing_fields:
            lines.append(f"  missing: {', '.join(item.missing_fields)}")
        if item.invalid_fields:
            lines.append(f"  invalid: {', '.join(item.invalid_fields)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only production import preflight")
    parser.add_argument("repository", type=Path)
    parser.add_argument("--include-provisional", action="store_true")
    parser.add_argument("--format", choices=("summary", "json"), default="summary")
    args = parser.parse_args()
    report = inspect_repository(args.repository, include_provisional=args.include_provisional)
    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_summary(report))


if __name__ == "__main__":
    main()
