"""Read-only audit of historical data lanes, manifests, chunks, and indexes.

The module intentionally uses only the Python standard library and Python 3.8
syntax.  It never rewrites source data or approval metadata.
"""

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOPICS = (
    ("목포 개항", ("목포 개항", "개항장", "개항")),
    ("목포 해관", ("목포 해관", "해관", "세관")),
    ("외국인 거류지·조계지", ("외국인 거류지", "조계지", "조계", "거류지")),
    ("구 일본영사관", ("구 목포 일본영사관", "목포 일본영사관", "일본영사관", "근대역사관 1관")),
    ("동양척식주식회사 목포지점", ("동양척식주식회사 목포지점", "동양척식주식회사", "동척", "근대역사관 2관")),
    ("목포 근대역사문화공간", ("목포 근대역사문화공간", "근대역사문화공간", "근대역사관")),
    ("근대 항만과 철도", ("목포항", "항만", "부두", "철도", "호남선")),
    ("일제강점기 목포의 산업과 도시 변화", ("산업", "도시 변화", "도시변화", "도시 형성", "면화", "미곡", "상업", "공업", "노동")),
)

MISSING_FIELDS = (
    "document_id", "title", "institution", "source_url", "published_date",
    "accessed_at", "license_status", "usage_scope", "review_status",
    "production_approved", "historical_period", "topic_tags", "place_tags",
    "person_tags", "raw_text_path", "clean_text_path",
)


def _jsonl(path):
    # type: (Path) -> List[Dict[str, Any]]
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            item["_audit_file"] = path.as_posix()
            item["_audit_line"] = number
            records.append(item)
    return records


def _value(record, meaning):
    # type: (Mapping[str, Any], str) -> Any
    aliases = {
        "document_id": ("document_id", "source_id"),
        "title": ("title", "source_title", "citation_title"),
        "institution": ("institution", "publisher"),
        "source_url": ("source_url", "canonical_source_url", "citation_url"),
        "accessed_at": ("accessed_at", "accessed_date", "collected_at"),
        "license_status": ("license_status", "license_review_status", "copyright_status", "rights_status"),
        "usage_scope": ("usage_scope",),
        "review_status": ("review_status", "usage_status"),
        "historical_period": ("historical_period", "period"),
        "topic_tags": ("topic_tags", "keywords", "retrieval_subjects", "topic", "primary_topic"),
        "place_tags": ("place_tags", "places"),
        "person_tags": ("person_tags", "people"),
        "raw_text_path": ("raw_text_path", "local_path"),
        "clean_text_path": ("clean_text_path",),
    }
    for key in aliases.get(meaning, (meaning,)):
        value = record.get(key)
        if value is not None and value != "" and value != []:
            return value
    return None


def _document_id(record):
    # type: (Mapping[str, Any]) -> str
    return str(record.get("document_id") or record.get("source_id") or "")


def _title(record):
    # type: (Mapping[str, Any]) -> str
    return str(record.get("title") or record.get("source_title") or record.get("citation_title") or "")


def _lane(path):
    # type: (str) -> str
    text = path.replace("\\", "/")
    if "tests/fixtures/" in text:
        return "fixture"
    if "/provisional_hackathon/" in "/" + text:
        return "provisional"
    if "/development_real/" in "/" + text:
        return "development_real"
    return "production_candidate"


def _status(record):
    # type: (Mapping[str, Any]) -> str
    if _lane(str(record.get("_audit_file", ""))) == "fixture" or record.get("is_fixture") is True:
        return "fixture"
    if record.get("production_approved") is True:
        return "production_approved"
    value = str(record.get("review_status") or record.get("usage_status") or "").lower()
    if value in ("rejected", "production_rejected"):
        return "production_rejected"
    if "provisional" in value or _lane(str(record.get("_audit_file", ""))) == "provisional":
        return "provisional"
    if value == "reference_only" or record.get("allowed_for_rag") is False and value == "reviewed":
        return "reference_only"
    if value == "draft":
        return "draft"
    if not value:
        return "review_status_missing"
    return "other_review_state"


def _license(record):
    # type: (Mapping[str, Any]) -> str
    value = _value(record, "license_status")
    return str(value).lower() if value is not None else "missing"


def _normalize_url(value):
    # type: (Any) -> str
    return str(value or "").strip().rstrip("/").lower()


def _normalize_text(value):
    # type: (Any) -> str
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _group_duplicates(records, key_func):
    # type: (Sequence[Mapping[str, Any]], Any) -> List[Dict[str, Any]]
    grouped = defaultdict(list)  # type: Dict[str, List[Mapping[str, Any]]]
    for record in records:
        key = key_func(record)
        if key:
            grouped[key].append(record)
    result = []
    for key in sorted(grouped):
        items = grouped[key]
        if len(items) > 1:
            result.append({"value": key, "count": len(items), "document_ids": sorted(_document_id(item) for item in items),
                           "locations": sorted("%s:%s" % (item.get("_audit_file"), item.get("_audit_line")) for item in items)})
    return result


def _similar_titles(records):
    # type: (Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]
    result = []
    for index, left in enumerate(records):
        left_title = _normalize_text(_title(left)).lower()
        if not left_title:
            continue
        for right in records[index + 1:]:
            right_title = _normalize_text(_title(right)).lower()
            ratio = SequenceMatcher(None, left_title, right_title).ratio()
            if ratio >= 0.88:
                result.append({"document_ids": [_document_id(left), _document_id(right)], "titles": [_title(left), _title(right)], "similarity": round(ratio, 3)})
    return result


def _similar_bodies(records, texts_by_document):
    # type: (Sequence[Mapping[str, Any]], Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]
    bodies = []
    for record in records:
        body = _normalize_text("\n".join(texts_by_document.get(_document_id(record), [])))
        if body:
            bodies.append((_document_id(record), body))
    result = []
    for index, left in enumerate(bodies):
        for right in bodies[index + 1:]:
            ratio = SequenceMatcher(None, left[1], right[1]).ratio()
            if ratio >= 0.92:
                result.append({"document_ids": [left[0], right[0]], "similarity": round(ratio, 3), "review_required": True})
    return result


def _quality_warnings(record, text):
    # type: (Mapping[str, Any], str) -> List[str]
    warnings = []
    clean = _normalize_text(text)
    if not clean:
        warnings.append("empty_body")
    elif len(clean) < 120:
        warnings.append("short_body")
    if "�" in clean or clean.count("?") > max(10, len(clean) // 20):
        warnings.append("encoding_or_ocr_review_required")
    if re.search(r"(.)\1{7,}", clean):
        warnings.append("repeated_characters_review_required")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]
    if len(paragraphs) != len(set(paragraphs)):
        warnings.append("repeated_paragraph_review_required")
    if re.search(r"<(script|style|nav|footer|header)\b|javascript:", str(text or ""), re.I):
        warnings.append("html_or_script_residue")
    title_terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", _title(record)))
    if clean and title_terms and not any(term in clean for term in title_terms):
        warnings.append("title_body_mismatch_review_required")
    if clean and "목포" not in (_title(record) + clean + str(_value(record, "topic_tags") or "")):
        warnings.append("low_mokpo_relevance_review_required")
    for meaning, warning in (("source_url", "missing_source_url"), ("institution", "missing_institution"), ("license_status", "missing_license_status")):
        if _value(record, meaning) is None:
            warnings.append(warning)
    return warnings


def _topic_matches(records, chunks):
    # type: (Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]
    chunks_by_document = defaultdict(list)  # type: Dict[str, List[Mapping[str, Any]]]
    for chunk in chunks:
        chunks_by_document[_document_id(chunk)].append(chunk)
    coverage = []
    for topic, keywords in TOPICS:
        matched = []
        matched_chunks = []
        confidence = []
        for record in records:
            doc_id = _document_id(record)
            metadata = " ".join((_title(record), str(_value(record, "topic_tags") or ""), str(_value(record, "historical_period") or "")))
            body = " ".join(str(item.get("text") or "") for item in chunks_by_document.get(doc_id, []))
            metadata_hits = [word for word in keywords if word.lower() in metadata.lower()]
            text_hits = [word for word in keywords if word.lower() in body.lower()]
            if metadata_hits or text_hits:
                matched.append(record)
                matched_chunks.extend(item for item in chunks_by_document.get(doc_id, []) if any(word.lower() in str(item.get("text") or "").lower() for word in keywords))
                confidence.append("high" if metadata_hits and text_hits else "medium" if metadata_hits else "low_review_required")
        institutions = Counter(str(_value(item, "institution") or "(missing)") for item in matched)
        total = len(matched)
        dominant = max(institutions.values()) if institutions else 0
        independent = len([name for name in institutions if name != "(missing)"])
        target = 5 if total < 5 else 10
        unknown_count = sum(_license(item) in ("unknown", "unconfirmed", "pending_review", "missing") for item in matched)
        recommended = max(0, target - total)
        if independent < 2:
            recommended = max(5, recommended)
        if total and unknown_count == total:
            recommended = max(3, recommended)
        if total and float(dominant) / total > 0.75:
            recommended = max(3, recommended)
        coverage.append({
            "topic": topic, "classification": "estimated", "document_count": total,
            "chunk_count": len(matched_chunks), "source_institution_count": independent,
            "production_approved_count": sum(_status(item) == "production_approved" for item in matched),
            "provisional_count": sum(_status(item) == "provisional" for item in matched),
            "reference_only_count": sum(_status(item) == "reference_only" for item in matched),
            "license_unknown_count": unknown_count,
            "dominant_source_share": round(float(dominant) / total, 3) if total else 0.0,
            "independent_source_count": independent, "search_keywords": list(keywords),
            "representative_titles": sorted({_title(item) for item in matched if _title(item)})[:5],
            "document_ids": sorted(_document_id(item) for item in matched),
            "confidence": Counter(confidence), "shortage": total < 5 or independent < 2 or not any(_status(item) == "production_approved" for item in matched),
            "recommended_additional_count": recommended,
        })
    return coverage


def _manifest_files(root, data_dir, fixture_dir):
    # type: (Path, Path, Optional[Path]) -> List[Path]
    files = list(data_dir.glob("**/manifests/sources.jsonl"))
    if fixture_dir and fixture_dir.exists():
        files.extend(fixture_dir.glob("**/fictional_documents.jsonl"))
    return sorted(set(files))


def audit_repository(root, data_dir=None, fixture_dir=None):
    # type: (Path, Optional[Path], Optional[Path]) -> Dict[str, Any]
    root = Path(root).resolve()
    data_dir = (Path(data_dir) if data_dir else root / "data").resolve()
    fixture_dir = (Path(fixture_dir) if fixture_dir else root / "tests" / "fixtures").resolve()
    manifest_files = _manifest_files(root, data_dir, fixture_dir)
    records = []
    for path in manifest_files:
        records.extend(_jsonl(path))
    fixtures = [item for item in records if _lane(str(item["_audit_file"])) == "fixture" or item.get("is_fixture") is True]
    actual = [item for item in records if item not in fixtures]

    chunk_files = sorted(data_dir.glob("**/chunks.jsonl")) + sorted(fixture_dir.glob("**/fictional_chunks.jsonl"))
    chunks = []
    for path in sorted(set(chunk_files)):
        chunks.extend(_jsonl(path))
    fixture_chunks = [item for item in chunks if _lane(str(item["_audit_file"])) == "fixture" or item.get("data_classification") == "fictional_fixture"]
    actual_chunks = [item for item in chunks if item not in fixture_chunks]

    raw_files = sorted(path for path in data_dir.glob("**/raw/**/*") if path.is_file() and path.name != ".gitkeep")
    extracted_files = sorted(path for path in data_dir.glob("extracted/**/*") if path.is_file() and path.name != ".gitkeep")
    manifest_ids = {_document_id(item) for item in actual if _document_id(item)}
    chunk_ids = {_document_id(item) for item in actual_chunks if _document_id(item)}

    indexed_documents = set()  # type: set
    indexed_chunk_ids = set()  # type: set
    index_details = []
    for path in sorted(data_dir.glob("**/retrieval_index/*.json")):
        if path.parent.name == "snapshots":
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        entries = payload.get("entries", [])
        entry_records = [item.get("chunk", item.get("payload", item)) for item in entries]
        doc_ids = {_document_id(item) for item in entry_records}
        chunk_entry_ids = {str(item.get("chunk_id") or "") for item in entry_records}
        indexed_documents.update(value for value in doc_ids if value)
        indexed_chunk_ids.update(value for value in chunk_entry_ids if value)
        index_details.append({"path": path.relative_to(root).as_posix(), "document_count": len([x for x in doc_ids if x]), "chunk_count": len(entries)})
    # Index manifests are authoritative even when a vector file is absent or empty.
    for path in sorted(data_dir.glob("**/index_ready/index_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        documents = payload.get("documents", {})
        indexed_documents.update(documents.keys())

    raw_references = {}
    for item in actual:
        local = item.get("local_path")
        if local:
            raw_references[_document_id(item)] = (root / str(local)).resolve()
        elif _lane(str(item.get("_audit_file", ""))) == "provisional":
            raw_references[_document_id(item)] = data_dir / "provisional_hackathon" / "raw" / (_document_id(item) + ".html")
    missing_raw = sorted(doc_id for doc_id, path in raw_references.items() if not path.exists())
    manifest_without_raw_reference = sorted(_document_id(item) for item in actual if _document_id(item) not in raw_references)
    raw_without_manifest = sorted(path.relative_to(root).as_posix() for path in raw_files if not any(path.resolve() == ref.resolve() for ref in raw_references.values()))
    extracted_document_ids = set()
    for doc_id, raw_path in raw_references.items():
        try:
            relative_raw = raw_path.resolve().relative_to((data_dir / "raw").resolve())
        except ValueError:
            continue
        expected_extracted = data_dir / "extracted" / relative_raw.with_suffix(".txt")
        if expected_extracted.exists():
            extracted_document_ids.add(doc_id)

    texts_by_document = defaultdict(list)  # type: Dict[str, List[str]]
    for chunk in actual_chunks:
        texts_by_document[_document_id(chunk)].append(str(chunk.get("text") or ""))
    body_hashes = defaultdict(list)  # type: Dict[str, List[str]]
    quality = []
    for item in actual:
        doc_id = _document_id(item)
        body = "\n".join(texts_by_document.get(doc_id, []))
        if body:
            body_hashes[hashlib.sha256(_normalize_text(body).encode("utf-8")).hexdigest()].append(doc_id)
        warnings = _quality_warnings(item, body)
        if warnings:
            quality.append({"document_id": doc_id, "warnings": warnings, "review_required": any("review_required" in value for value in warnings)})
    duplicate_body_hashes = [{"sha256": key, "document_ids": sorted(value)} for key, value in sorted(body_hashes.items()) if len(value) > 1]

    duplicate_chunk_ids = _group_duplicates(actual_chunks, lambda item: str(item.get("chunk_id") or ""))
    duplicate_chunk_bodies = _group_duplicates(actual_chunks, lambda item: hashlib.sha256(_normalize_text(item.get("text")).encode("utf-8")).hexdigest() if _normalize_text(item.get("text")) else "")
    status_counts = Counter(_status(item) for item in records)
    for status_name in ("fixture", "draft", "provisional", "reference_only", "production_approved", "production_rejected", "review_status_missing", "other_review_state"):
        status_counts.setdefault(status_name, 0)
    license_counts = Counter(_license(item) for item in actual)
    missing = {field: sum(_value(item, field) is None for item in actual) for field in MISSING_FIELDS}
    production_usable = sum(_status(item) == "production_approved" and _license(item) not in ("unknown", "unconfirmed", "pending_review", "missing") for item in actual)

    report = {
        "audit_version": 1,
        "basis": {"root": root.as_posix(), "data_dir": data_dir.relative_to(root).as_posix(), "fixture_policy": "excluded_from_actual_historical_data", "topic_classification": "estimated; low confidence requires manual review"},
        "counts": {"manifest_documents_total": len(records), "actual_documents_excluding_fixtures": len(actual), "fixture_documents": len(fixtures),
                   "raw_files": len(raw_files), "extracted_text_files": len(extracted_files), "chunks_total": len(chunks), "actual_chunks_excluding_fixtures": len(actual_chunks),
                   "fixture_chunks": len(fixture_chunks), "indexed_documents": len(indexed_documents), "indexed_chunks": len(indexed_chunk_ids), "production_usable_documents": production_usable},
        "lane_counts": dict(sorted(Counter(_lane(str(item.get("_audit_file", ""))) for item in records).items())),
        "chunk_lane_counts": dict(sorted(Counter(_lane(str(item.get("_audit_file", ""))) for item in chunks).items())),
        "raw_lane_counts": {"production_candidate": sum("provisional_hackathon" not in path.as_posix() for path in raw_files), "provisional": sum("provisional_hackathon" in path.as_posix() for path in raw_files)},
        "status_counts": dict(sorted(status_counts.items())), "license_counts": dict(sorted(license_counts.items())), "missing_field_counts": missing,
        "physical_field_absence": {"review_status": sum("review_status" not in item for item in actual), "production_approved": sum("production_approved" not in item for item in actual)},
        "duplicates": {"document_id": _group_duplicates(actual, _document_id), "source_url": _group_duplicates(actual, lambda item: _normalize_url(_value(item, "source_url"))),
                       "similar_titles": _similar_titles(actual), "identical_body_hash": duplicate_body_hashes, "similar_body": _similar_bodies(actual, texts_by_document), "duplicate_chunk_id": duplicate_chunk_ids, "identical_chunk_body": duplicate_chunk_bodies},
        "mismatches": {"raw_without_manifest": raw_without_manifest, "manifest_missing_raw_file": missing_raw, "manifest_without_raw_path_or_resolvable_rule": manifest_without_raw_reference,
                       "manifest_without_extracted_text": sorted(manifest_ids - extracted_document_ids),
                       "chunks_without_manifest_document": sorted(chunk_ids - manifest_ids), "manifest_without_chunks": sorted(manifest_ids - chunk_ids),
                       "index_documents_not_in_manifest": sorted(indexed_documents - manifest_ids), "manifest_documents_not_in_index": sorted(manifest_ids - indexed_documents),
                       "index_details": index_details},
        "topic_coverage": _topic_matches(actual, actual_chunks), "quality_warnings": quality,
        "collection_priority": sorted(({"topic": item["topic"], "recommended_additional_count": item["recommended_additional_count"], "reason": "coverage/source diversity/approval gap"} for item in _topic_matches(actual, actual_chunks)), key=lambda item: (-item["recommended_additional_count"], item["topic"])),
        "schema_mapping": {
            "document_id": "document_id (provisional fallback: source_id)", "title": "title (provisional: source_title)", "institution": "institution or publisher",
            "source_url": "source_url or canonical_source_url", "published_date": "published_date", "accessed_at": "accessed_date/collected_at",
            "license_status": "copyright_status/license_review_status/rights_status", "usage_scope": "usage_scope (no general-schema equivalent)",
            "review_status": "review_status or provisional usage_status", "production_approved": "development schema only; general schema uses allowed_for_rag plus reviewed",
            "historical_period": "historical_period or period", "topic_tags": "keywords/retrieval_subjects/topic/primary_topic", "place_tags": "places", "person_tags": "people",
            "raw_text_path": "local_path; provisional deterministic raw path", "clean_text_path": "no manifest field; chunks/processed files are separate artifacts",
        },
        "schema_field_assessment": {
            "currently_present_and_sufficient": ["document_id", "title", "source_url", "published_date", "accessed_at (accessed_date)", "license_status (copyright_status)", "review_status", "historical_period", "person_tags (people)"],
            "present_but_incomplete": ["institution (publisher)", "topic_tags (keywords)", "place_tags (places)", "raw_text_path (local_path: raw/clean meaning is ambiguous)"],
            "not_in_current_general_schema": ["usage_scope", "production_approved", "clean_text_path"],
            "minimal_change_proposal_only": "A future version may add usage_scope, production_approved, raw_text_path, and clean_text_path without changing existing values; no schema change was made by this audit.",
        },
    }
    return report


def render_markdown(report):
    # type: (Mapping[str, Any]) -> str
    counts = report["counts"]
    lines = ["# 목포 역사 데이터 감사 보고서", "", "> 자동 분류는 추정이며 역사성·OCR 오류·production 승인을 확정하지 않습니다.", "", "## 1. 집계 기준", "",
             "- fixture는 실제 역사 데이터에서 제외했습니다.", "- provisional과 개발 검증용 자료는 현황에는 포함하되 production 사용 가능으로 보지 않았습니다.", "- 권리 상태가 unknown/unconfirmed/pending/missing인 자료는 production 가능 수에서 제외했습니다.", "",
             "## 2. 전체 집계", "", "| 항목 | 수 |", "|---|---:|"]
    labels = (("manifest_documents_total", "manifest 전체"), ("actual_documents_excluding_fixtures", "fixture 제외 실제/후보 문서"), ("fixture_documents", "fixture 문서"), ("production_usable_documents", "production 사용 가능"), ("raw_files", "원문 파일"), ("extracted_text_files", "추출 텍스트 파일"), ("actual_chunks_excluding_fixtures", "fixture 제외 chunk"), ("indexed_documents", "index 문서"), ("indexed_chunks", "index chunk"))
    for key, label in labels:
        lines.append("| %s | %s |" % (label, counts[key]))
    lines.extend(["", "## 3. 상태 및 권리", "", "### 상태", ""])
    for key, value in report["status_counts"].items():
        lines.append("- `%s`: %s" % (key, value))
    lines.extend(["", "### 권리 상태", ""])
    for key, value in report["license_counts"].items():
        lines.append("- `%s`: %s" % (key, value))
    lines.extend(["", "## 4. 누락 필드", "", "| 의미 필드 | 누락 |", "|---|---:|"])
    for key, value in report["missing_field_counts"].items():
        lines.append("| `%s` | %s |" % (key, value))
    lines.extend(["", "필드 대응은 JSON의 `schema_mapping`과 `schema_field_assessment`에 기록했습니다. 현재 스키마는 `publisher`, `copyright_status`, `accessed_date`, `local_path`, `keywords`, `places`, `people`를 사용하며, 개발 레인은 별도 필드 집합을 사용합니다. `usage_scope`, `production_approved`, `clean_text_path`는 일반 스키마에 없습니다. 이번 감사에서는 스키마를 변경하지 않았습니다.", "",
                  "## 5. 중복과 불일치", ""])
    for key, values in report["duplicates"].items():
        lines.append("- %s: %s건" % (key, len(values)))
    for key, values in report["mismatches"].items():
        if key != "index_details":
            lines.append("- %s: %s건" % (key, len(values)))
    lines.extend(["", "## 6. 품질 경고", "", "- 경고 문서: %s건" % len(report["quality_warnings"]), "- 상세 항목은 JSON 보고서에서 확인합니다. 휴리스틱 경고는 수동 검토 대상으로만 사용합니다.", "",
                  "## 7. 주제별 커버리지", "", "| 주제 | 문서 | chunk | 기관 | 승인 | provisional | 권리 미확정 | 최대 출처 비중 | 부족 | 추가 권고 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"])
    for item in report["topic_coverage"]:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %.0f%% | %s | %s |" % (item["topic"], item["document_count"], item["chunk_count"], item["source_institution_count"], item["production_approved_count"], item["provisional_count"], item["license_unknown_count"], item["dominant_source_share"] * 100, "예" if item["shortage"] else "아니오", item["recommended_additional_count"]))
    lines.extend(["", "## 8. 다음 수집 배치", "", "- 1차 배치는 부족도가 높은 핵심 주제 중심 10~20건을 권고합니다.", "- 기관별 최대 3~5건으로 제한하고, 주제별 최소 2개 독립 기관을 목표로 합니다.", "- 로그인·캡차·유료벽·이용조건 불명확 자료는 제외합니다.", "- 원문과 추출 텍스트를 분리하고 기존 정책의 draft 상태를 유지합니다.", "- 사람 검토 전 `production_approved`를 변경하지 않습니다.", "", "## 9. 수동 검토 필요", "", "- 역사적 사실성과 목포 직접 관련성", "- OCR/인코딩 휴리스틱 경고", "- 권리 상태가 unknown, unconfirmed 또는 pending_review인 전 자료", "- manifest와 파일·chunk·index 불일치", ""])
    return "\n".join(lines)


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--json-output", type=Path, default=Path("reports/history_data_audit.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/HISTORY_DATA_AUDIT.md"))
    args = parser.parse_args(argv)
    report = audit_repository(args.root, args.data_dir, args.fixture_dir)
    json_output = args.json_output if args.json_output.is_absolute() else args.root / args.json_output
    markdown_output = args.markdown_output if args.markdown_output.is_absolute() else args.root / args.markdown_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print("wrote %s and %s" % (json_output, markdown_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
