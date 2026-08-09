"""Build a local-only, evidence-preserving corpus for the hackathon chatbot."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from history_chatbot.indexing.snapshot import stable_json_hash


DIRECT_TERMS = (
    "목포", "목포부", "목포항", "목포역", "목포해관", "목포세관",
    "무안감리서", "삼학도", "유달산", "호남선", "동양척식", "호남은행",
)
HISTORY_TERMS = (
    "역사", "근대", "개항", "대한제국", "일제강점기", "독립운동",
    "학생운동", "노동운동", "문화유산", "등록문화", "설립", "건립",
    "당시", "연혁", "변천", "사건", "기록", "철도", "항만", "사료",
)
NOISE_TERMS = (
    "access denied", "captcha", "로그인이 필요", "페이지를 찾을 수 없",
    "서비스 이용에 불편", "검색 결과가 없습니다",
)
BOILERPLATE = (
    "본문 바로가기", "주메뉴 바로가기", "메뉴 바로가기", "하단메뉴 바로가기",
    "다국어 입력", "통합검색", "개인정보처리방침", "이 누리집은 대한민국 공식",
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _title(record: dict[str, Any], text: str) -> str:
    match = re.search(r"^제목:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return _normalize(match.group(1))
    return _normalize(str(record.get("title") or record.get("source_title") or "제목 없음"))


def core_text(text: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = _normalize(re.sub(r"https?://\S+", " ", raw))
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    value = _normalize(" ".join(lines))
    for marker in BOILERPLATE:
        value = value.replace(marker, " ")
    return _normalize(value)


def _source_core(record: dict[str, Any], value: str) -> str:
    source_id = str(record.get("source_id", ""))
    if source_id.startswith("grandculture"):
        match = re.search(r"(?:\[정의\]|정의\])\s*", value)
        if match:
            value = value[match.start():]
    elif source_id.startswith("encykorea"):
        marker = value.find("정의 닫기")
        if marker >= 0:
            value = value[marker:]
    for ending in ("콘텐츠 만족도 조사", "이용약관 개인정보처리방침"):
        marker = value.find(ending)
        if marker > 500:
            value = value[:marker]
    return _normalize(value)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class Verification:
    classification: str
    reasons: tuple[str, ...]
    title: str
    core: str
    core_sha256: str


def verify_candidate(record: dict[str, Any], *, root: Path) -> Verification:
    reasons: list[str] = []
    raw_path = root / str(record.get("raw_path", ""))
    extracted_path = root / str(record.get("extracted_path", ""))
    if not raw_path.is_file() or not extracted_path.is_file():
        return Verification("INVALID", ("missing_artifact",), "", "", "")
    raw = raw_path.read_bytes()
    extracted_bytes = extracted_path.read_bytes()
    text = extracted_bytes.decode("utf-8", errors="replace")
    title = _title(record, text)
    core = _source_core(record, core_text(text))
    if not raw or not core:
        reasons.append("empty_artifact")
    if record.get("raw_sha256") and _sha256(raw) != record["raw_sha256"]:
        reasons.append("raw_hash_mismatch")
    if record.get("extracted_sha256") and _sha256(extracted_bytes) != record["extracted_sha256"]:
        reasons.append("extracted_hash_mismatch")
    if record.get("duplicate_status") == "confirmed" or (
        record.get("provenance") or {}
    ).get("new_unique_increment") != 1:
        return Verification("DUPLICATE", ("declared_duplicate",), title, core, _sha256(core.encode()))
    if record.get("extraction_status") != "success" or len(core) < 550:
        reasons.append("insufficient_body")
    if "�" in core and core.count("�") / max(len(core), 1) > 0.002:
        reasons.append("encoding_damage")
    lowered = core.casefold()
    if any(term in lowered for term in NOISE_TERMS):
        reasons.append("error_or_access_page")
    urls = (record.get("source_url"), record.get("canonical_url"))
    if any(urlsplit(str(url or "")).scheme != "https" for url in urls):
        reasons.append("invalid_source_url")
    combined = f"{title} {core}"
    direct_hits = sum(combined.count(term) for term in DIRECT_TERMS)
    history_hits = sum(term in combined for term in HISTORY_TERMS)
    title_direct = any(term in title for term in DIRECT_TERMS)
    if not (title_direct or direct_hits >= 2):
        reasons.append("weak_mokpo_relevance")
    if history_hits < 2:
        reasons.append("weak_historical_substance")
    # The newspaper pages collected in round one are mostly UI/metadata shells.
    if record.get("source_id") == "national_library_newspaper":
        article_tail = core[-3500:]
        if len(core) < 11_000 or not re.search(r"(?:기사본문|원문텍스트|본문내용)", article_tail):
            reasons.append("newspaper_metadata_shell")
    hard = {
        "missing_artifact", "empty_artifact", "raw_hash_mismatch",
        "extracted_hash_mismatch", "insufficient_body", "encoding_damage",
        "error_or_access_page", "invalid_source_url", "weak_mokpo_relevance",
        "weak_historical_substance", "newspaper_metadata_shell",
    }
    classification = "INVALID" if hard.intersection(reasons) else "VALID"
    return Verification(classification, tuple(reasons), title, core, _sha256(core.encode()))


def _chunks(text: str, *, size: int = 950, overlap: int = 120) -> Iterable[str]:
    position = 0
    while position < len(text):
        end = min(len(text), position + size)
        if end < len(text):
            boundary = max(text.rfind(". ", position, end), text.rfind("다. ", position, end))
            if boundary > position + size // 2:
                end = boundary + 2
        yield text[position:end].strip()
        if end >= len(text):
            break
        position = max(position + 1, end - overlap)


def build_verified_corpus(
    *,
    root: Path,
    candidate_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in candidate_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    verified: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_core: set[str] = set()
    counts: Counter[str] = Counter()
    for record in records:
        result = verify_candidate(record, root=root)
        counts[result.classification] += 1
        if result.classification != "VALID" or result.core_sha256 in seen_core:
            if result.classification == "VALID":
                counts["DUPLICATE"] += 1
                counts["VALID"] -= 1
            continue
        seen_core.add(result.core_sha256)
        document_id = str(record.get("document_id") or record["candidate_id"])
        source_url = str(record.get("source_url") or record.get("canonical_url"))
        rights_status = str(record.get("rights_status") or "unknown")
        item = {
            "document_id": document_id,
            "candidate_id": str(record["candidate_id"]),
            "title": result.title,
            "current_name": record.get("current_name") or "unknown",
            "historical_name": record.get("historical_name") or "unknown",
            "year": record.get("year") or "unknown",
            "period": record.get("historical_period") or "unknown",
            "place_id": record.get("place_id") or "unknown",
            "source_id": record.get("source_id") or "unknown",
            "institution": record.get("institution") or "unknown",
            "publisher": record.get("publisher") or "unknown",
            "publisher_family": record.get("publisher_family") or "unknown",
            "source_title": record.get("source_title") or result.title,
            "source_url": source_url,
            "canonical_url": record.get("canonical_url") or source_url,
            "rights_status": rights_status,
            "human_review_required": True,
            "review_status": "verified_hackathon_local_audit",
            "verification_status": "VALID",
            "raw_sha256": record.get("raw_sha256") or "",
            "extracted_sha256": record.get("extracted_sha256") or "",
            "core_sha256": result.core_sha256,
            "usage_status": "verified_hackathon",
            "allowed_for_rag": True,
            "allowed_for_training": False,
            "production_approved": False,
            "data_classification": "real_historical_source",
            "topic_categories": record.get("topic_categories") or [],
        }
        verified.append(item)
        for index, chunk_text in enumerate(_chunks(result.core)):
            chunk_id = f"{document_id}::chunk-{index:04d}"
            chunks.append({
                **item,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "content_sha256": _sha256(chunk_text.encode("utf-8")),
                "source_name": item["institution"],
                "copyright_status": rights_status,
                "source_reliability": "A" if record.get("source_tier") == "tier_1" else "B",
            })
    if len(verified) < 100:
        raise ValueError(f"verified hackathon corpus requires at least 100 documents: {len(verified)}")
    manifest_dir = output_root / "manifests"
    index_ready = output_root / "index_ready"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    index_ready.mkdir(parents=True, exist_ok=True)
    documents_path = manifest_dir / "verified_documents.jsonl"
    chunks_path = index_ready / "chunks.jsonl"
    documents_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in verified), encoding="utf-8")
    chunks_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in chunks), encoding="utf-8")
    index_manifest = {
        "data_lane": "verified_hackathon",
        "documents": {item["document_id"]: item["core_sha256"] for item in verified},
        "document_count": len(verified),
        "chunk_count": len(chunks),
        "snapshot_sha256": stable_json_hash(chunks),
        "source_distribution": dict(Counter(str(x["source_id"]) for x in verified)),
        "rights_note": "Rights metadata is preserved; this is not production approval.",
    }
    (index_ready / "index_manifest.json").write_text(json.dumps(index_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**index_manifest, "classification_counts": dict(counts)}
