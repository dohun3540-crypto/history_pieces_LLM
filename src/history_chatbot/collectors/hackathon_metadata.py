"""Conservative metadata/excerpt-only collection for approved archive pages."""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


ALLOWED_HOST = "theme.archives.go.kr"
MIN_EXCERPT_LENGTH = 20
MAX_EXCERPT_LENGTH = 400

FIXED_RIGHTS = {
    "review_status": "draft",
    "usage_status": "provisional_hackathon",
    "copyright_status": "unknown",
    "rights_status": "unconfirmed",
    "allowed_for_rag": False,
    "allowed_for_training": False,
    "redistribution_allowed": False,
    "public_release_allowed": False,
    "production_approved": False,
    "usage_scope": "hackathon_internal_review",
    "raw_source_status": "remote_only",
    "robots_status": "not_published_or_404",
}

CANDIDATES = {
    "archives-cja0002271-0027148187": {
        "document_id": "archives-cja0002271-0027148187",
        "title": "진남포 목포 각국 조계장정(역문)(1897년 10월16일조인)",
        "institution": "국가기록원",
        "source_url": "https://theme.archives.go.kr/next/government/viewGovernmentArchivesEvent.do?docid=0027148187&id=0001564827",
        "canonical_url": "https://theme.archives.go.kr/next/government/viewGovernmentArchivesEvent.do?docid=0027148187&id=0001564827",
        "topic_tags": ["외국인 거류지·조계지"],
        "related_document_id": "archives-cja0002271-overview",
        "raw_source_status": "remote_only",
        "robots_status": "not_published_or_404",
        "expected_host": ALLOWED_HOST,
        "document_type": "archive_item",
    },
    "archives-cja0002271-overview": {
        "document_id": "archives-cja0002271-overview",
        "title": "각국 거류지관계취극서(부근지도) 기록철 설명",
        "institution": "국가기록원",
        "source_url": "https://theme.archives.go.kr/next/government/viewGovernmentArchives.do?id=0001564827",
        "canonical_url": "https://theme.archives.go.kr/next/government/viewGovernmentArchives.do?id=0001564827",
        "topic_tags": ["외국인 거류지·조계지", "목포 개항"],
        "related_document_id": "archives-cja0002271-0027148187",
        "raw_source_status": "remote_only",
        "robots_status": "not_published_or_404",
        "expected_host": ALLOWED_HOST,
        "document_type": "archive_series_overview",
    },
}


class CollectionError(RuntimeError):
    pass


@dataclass
class FetchResponse:
    final_url: str
    content_type: str
    body: bytes
    status: int = 200


class _VisibleTextParser(HTMLParser):
    BLOCKED = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript"}

    def __init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)
        self.depth = 0
        self.title_depth = 0
        self.title_parts = []  # type: List[str]
        self.parts = []  # type: List[str]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        lowered = tag.lower()
        if lowered in self.BLOCKED:
            self.depth += 1
        if lowered == "title":
            self.title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self.BLOCKED and self.depth:
            self.depth -= 1
        if lowered == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if not self.depth:
            self.parts.append(data)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_title(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", normalize_space(value).lower())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_url(url: str, expected_host: str = ALLOWED_HOST) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise CollectionError("HTTPS URL만 허용됩니다: " + url)
    if (parsed.hostname or "").lower() != expected_host:
        raise CollectionError("허용되지 않은 host입니다: " + (parsed.hostname or ""))
    if parsed.username or parsed.password:
        raise CollectionError("URL 인증정보는 허용되지 않습니다")


def default_fetch(url: str, timeout: float, max_response_bytes: int) -> FetchResponse:
    validate_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "MokpoHistoryHackathonMetadata/1.0"})
    opener = urllib.request.build_opener(_SameHostRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        final_url = response.geturl()
        validate_url(final_url)
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in ("text/html", "application/xhtml+xml"):
            raise CollectionError("HTML 응답만 허용됩니다: " + media_type)
        body = response.read(max_response_bytes + 1)
        result = FetchResponse(
            final_url=final_url,
            content_type=content_type,
            body=body,
            status=getattr(response, "status", 200),
        )
    return result


class _SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        validate_url(newurl)
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )


def validate_response(response: FetchResponse, max_response_bytes: int) -> None:
    validate_url(response.final_url)
    if response.status < 200 or response.status >= 300:
        raise CollectionError("HTTP 응답 상태가 성공이 아닙니다")
    media_type = response.content_type.split(";", 1)[0].strip().lower()
    if media_type not in ("text/html", "application/xhtml+xml"):
        raise CollectionError("HTML 응답만 허용됩니다: " + media_type)
    if not response.body:
        raise CollectionError("빈 HTML 응답입니다")
    if len(response.body) > max_response_bytes:
        raise CollectionError("응답 크기 상한을 초과했습니다")


def _metadata_value(text: str, labels: Sequence[str]) -> str:
    for label in labels:
        match = re.search(re.escape(label) + r"\s*[:：]?\s*([^|\n]{1,120})", text)
        if match:
            value = normalize_space(match.group(1))
            value = re.split(r"\s+(?:관리번호|문서번호|생산연도|생산기관|공개구분|문서유형|기록철)\s*[:：]", value)[0]
            return value.strip()
    return ""


def extract_page(body: bytes, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        source = body.decode("utf-8")
    except UnicodeDecodeError:
        try:
            source = body.decode("cp949")
        except UnicodeDecodeError as exc:
            raise CollectionError("지원되는 한글 인코딩으로 HTML을 해석할 수 없습니다") from exc
    parser = _VisibleTextParser()
    parser.feed(source)
    page_title = normalize_space(" ".join(parser.title_parts))
    visible_raw = html.unescape("\n".join(parser.parts))
    visible = normalize_space(visible_raw)
    if not page_title:
        raise CollectionError("페이지 제목이 없습니다")
    if not visible:
        raise CollectionError("추출 가능한 본문이 없습니다")

    sentences = [normalize_space(item) for item in re.split(r"(?<=[.!?。])\s+|[\r\n]+", visible_raw)]
    relevant = [item for item in sentences if "목포" in item]
    excerpt = normalize_space(" ".join(relevant))
    if len(excerpt) < MIN_EXCERPT_LENGTH:
        raise CollectionError("목포 직접 관련 excerpt가 없거나 너무 짧습니다")
    excerpt = excerpt[:MAX_EXCERPT_LENGTH].rstrip()
    if "목포" not in excerpt:
        raise CollectionError("목포 직접 관련 excerpt가 없습니다")

    return {
        "page_title": page_title,
        "institution": _metadata_value(visible, ["기관명", "소장기관"]) or candidate["institution"],
        "management_number": _metadata_value(visible, ["관리번호"]),
        "document_number": _metadata_value(visible, ["문서번호"]),
        "production_year": _metadata_value(visible, ["생산연도"]),
        "producing_institution": _metadata_value(visible, ["생산기관"]),
        "disclosure_status": _metadata_value(visible, ["공개구분"]),
        "document_type": _metadata_value(visible, ["문서유형", "기록물 유형"]) or candidate["document_type"],
        "parent_id": _metadata_value(visible, ["기록철", "부모 ID"]) or candidate["related_document_id"],
        "canonical_url": candidate["canonical_url"],
        "excerpt": excerpt,
        "excerpt_sha256": sha256_text(excerpt),
    }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollectionError("manifest JSONL 오류 (행 %d): %s" % (number, exc))
        if not isinstance(item, dict):
            raise CollectionError("manifest 행은 객체여야 합니다")
        records.append(item)
    return records


def load_repository_records(manifest: Path, repository_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    paths = [manifest]
    if repository_root is not None:
        for path in sorted((repository_root / "data").glob("**/*.jsonl")):
            lowered = str(path).lower()
            if path != manifest and ("manifest" in lowered or path.name == "sources.jsonl"):
                paths.append(path)
    records = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        for original in iter_jsonl(path):
            record = dict(original)
            if repository_root is not None:
                text_path = _field(record, ["extracted_text_path", "clean_text_path"])
                if text_path:
                    candidate_path = Path(text_path)
                    if not candidate_path.is_absolute():
                        candidate_path = repository_root / candidate_path
                    if candidate_path.is_file():
                        record["_computed_extracted_hash"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            records.append(record)
    return records


def _field(record: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return normalize_space(value)
    return ""


def duplicate_check(candidate: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                    excerpt_hash: str = "", extracted_hash: str = "") -> List[str]:
    warnings = []
    candidate_title = normalize_title(candidate["title"])
    for record in records:
        for name in ("document_id", "source_id"):
            record_id = _field(record, [name])
            if candidate["document_id"] == record_id:
                warnings.append(name + " exact: " + record_id)
        record_urls = {_field(record, [name]) for name in ("source_url", "canonical_url", "canonical_source_url")}
        record_urls.discard("")
        if candidate["source_url"] in record_urls:
            warnings.append("source/canonical URL exact: " + candidate["source_url"])
        if candidate["canonical_url"] in record_urls:
            warnings.append("canonical/source URL exact: " + candidate["canonical_url"])
        title = normalize_title(_field(record, ["title", "source_title"]).__str__())
        if title and title == candidate_title:
            warnings.append("normalized title exact: " + _field(record, ["title", "source_title"]))
        elif title and SequenceMatcher(None, candidate_title, title).ratio() >= 0.88:
            warnings.append("similar title: " + _field(record, ["title", "source_title"]))
        record_hashes = {_field(record, [name]) for name in (
            "excerpt_sha256", "extracted_sha256", "_computed_extracted_hash", "content_hash", "body_hash"
        )}
        record_hashes.discard("")
        if excerpt_hash and excerpt_hash in record_hashes:
            warnings.append("excerpt/body hash exact: " + excerpt_hash)
        if extracted_hash and extracted_hash in record_hashes:
            warnings.append("extracted/body hash exact: " + extracted_hash)
    return sorted(set(warnings))


def build_record(candidate: Mapping[str, Any], extracted: Mapping[str, Any], extracted_path: Path, collected_at: str) -> Dict[str, Any]:
    rendered = render_extracted(candidate, extracted["excerpt"])
    record = {
        "source_id": candidate["document_id"],
        "document_id": candidate["document_id"],
        "source_title": candidate["title"],
        "title": candidate["title"],
        "institution": candidate["institution"],
        "source_url": candidate["source_url"],
        "canonical_url": candidate["canonical_url"],
        "topic": candidate["topic_tags"],
        "topic_tags": candidate["topic_tags"],
        "related_document_id": candidate["related_document_id"],
        "document_type": candidate["document_type"],
        "collected_at": collected_at,
        "network_requested": True,
        "active": True,
        "extracted_text_path": extracted_path.as_posix(),
        "excerpt_sha256": extracted["excerpt_sha256"],
        "extracted_sha256": sha256_text(rendered),
        "collection_metadata": {
            "page_title": extracted["page_title"],
            "management_number": extracted["management_number"],
            "document_number": extracted["document_number"],
            "production_year": extracted["production_year"],
            "producing_institution": extracted["producing_institution"],
            "disclosure_status": extracted["disclosure_status"],
            "parent_id": extracted["parent_id"],
            "excerpt_only": True,
        },
    }
    record.update(FIXED_RIGHTS)
    if any(record[key] != value for key, value in FIXED_RIGHTS.items()):
        raise CollectionError("권리 격리 metadata가 완화되었습니다")
    return record


def render_extracted(candidate: Mapping[str, Any], excerpt: str) -> str:
    return "제목: %s\n기관: %s\n상세 URL: %s\n\n%s\n" % (
        candidate["title"], candidate["institution"], candidate["source_url"], excerpt
    )


def append_transaction(manifest: Path, outputs: Sequence[Tuple[Path, str]], records: Sequence[Mapping[str, Any]], replace_file: Callable[[str, str], None] = os.replace) -> None:
    for target, unused in outputs:
        if target.exists():
            raise CollectionError("추출 파일이 이미 존재합니다: " + str(target))
    original = manifest.read_bytes() if manifest.exists() else b""
    if original and not original.endswith(b"\n"):
        original += b"\n"
    additions = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records).encode("utf-8")
    stage_root = Path(tempfile.mkdtemp(prefix=".hackathon-metadata-", dir=str(manifest.parent)))
    committed = []  # type: List[Path]
    try:
        stage_manifest = stage_root / "sources.jsonl"
        stage_manifest.write_bytes(original + additions)
        staged_outputs = []
        for index, (target, content) in enumerate(outputs):
            staged = stage_root / ("output-%d.txt" % index)
            # Preserve LF bytes so the recorded hash is identical on Windows.
            staged.write_bytes(content.encode("utf-8"))
            staged_outputs.append((staged, target))
        for staged, target in staged_outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            replace_file(str(staged), str(target))
            committed.append(target)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        replace_file(str(stage_manifest), str(manifest))
    except Exception:
        for target in committed:
            try:
                target.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(str(stage_root), ignore_errors=True)


def collect(
    candidate_ids: Sequence[str], manifest: Path, extracted_dir: Path, max_items: int,
    delay_seconds: float, timeout_seconds: float, max_response_bytes: int,
    fetcher: Callable[[str, float, int], FetchResponse] = default_fetch,
    repository_root: Optional[Path] = None, collected_at: Optional[str] = None,
    replace_file: Callable[[str, str], None] = os.replace,
) -> List[Dict[str, Any]]:
    if max_items < 1 or len(candidate_ids) > max_items:
        raise CollectionError("선택 후보가 max-items를 초과했습니다")
    if delay_seconds < 1.0:
        raise CollectionError("요청 간격은 최소 1초여야 합니다")
    if timeout_seconds <= 0 or max_response_bytes <= 0:
        raise CollectionError("timeout과 응답 크기 상한은 양수여야 합니다")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CollectionError("동일 후보를 두 번 요청할 수 없습니다")
    candidates = []
    for candidate_id in candidate_ids:
        if candidate_id not in CANDIDATES:
            raise CollectionError("등록되지 않은 후보입니다: " + candidate_id)
        candidate = CANDIDATES[candidate_id]
        validate_url(candidate["source_url"], candidate["expected_host"])
        candidates.append(candidate)
    existing = load_repository_records(manifest, repository_root)
    for candidate in candidates:
        warnings = duplicate_check(candidate, existing)
        if warnings:
            raise CollectionError("중복 또는 유사 후보: " + "; ".join(warnings))

    extracted_items = []
    for index, candidate in enumerate(candidates):
        if index:
            time.sleep(delay_seconds)
        response = fetcher(candidate["source_url"], timeout_seconds, max_response_bytes)
        validate_response(response, max_response_bytes)
        page = extract_page(response.body, candidate)
        rendered_hash = sha256_text(render_extracted(candidate, page["excerpt"]))
        warnings = duplicate_check(candidate, existing, page["excerpt_sha256"], rendered_hash)
        if warnings:
            raise CollectionError("중복 또는 유사 후보: " + "; ".join(warnings))
        extracted_items.append((candidate, page))
    for left in range(len(extracted_items)):
        for right in range(left + 1, len(extracted_items)):
            left_page = extracted_items[left][1]
            right_page = extracted_items[right][1]
            if left_page["excerpt_sha256"] == right_page["excerpt_sha256"]:
                raise CollectionError("후보 간 excerpt가 동일합니다")
            ratio = SequenceMatcher(None, left_page["excerpt"], right_page["excerpt"]).ratio()
            if ratio >= 0.90:
                raise CollectionError("후보 간 excerpt 유사도가 너무 높습니다")

    stamp = collected_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outputs = []
    records = []
    for candidate, page in extracted_items:
        target = extracted_dir / (candidate["document_id"] + ".txt")
        outputs.append((target, render_extracted(candidate, page["excerpt"])))
        records.append(build_record(candidate, page, target, stamp))
    append_transaction(manifest, outputs, records, replace_file=replace_file)
    return records


def dry_run(candidate_ids: Sequence[str], manifest: Path, extracted_dir: Path, max_items: int,
            delay_seconds: float, timeout_seconds: float, max_response_bytes: int,
            repository_root: Optional[Path] = None) -> Dict[str, Any]:
    if max_items < 1 or len(candidate_ids) > max_items:
        raise CollectionError("선택 후보가 max-items를 초과했습니다")
    if delay_seconds < 1.0 or timeout_seconds <= 0 or max_response_bytes <= 0:
        raise CollectionError("안전 제한값이 유효하지 않습니다")
    existing = load_repository_records(manifest, repository_root)
    selected = []
    for candidate_id in candidate_ids:
        if candidate_id not in CANDIDATES:
            raise CollectionError("등록되지 않은 후보입니다: " + candidate_id)
        candidate = CANDIDATES[candidate_id]
        validate_url(candidate["source_url"], candidate["expected_host"])
        selected.append({
            "document_id": candidate_id,
            "request_url": candidate["source_url"],
            "expected_host": candidate["expected_host"],
            "extracted_path": str(extracted_dir / (candidate_id + ".txt")),
            "duplicate_precheck": duplicate_check(candidate, existing) or ["none"],
        })
    return {
        "mode": "dry-run (network disabled; no files created)",
        "selected_candidates": selected,
        "fixed_rights": FIXED_RIGHTS,
        "maximum_requests_on_execute": len(selected),
        "delay_seconds": delay_seconds,
        "timeout_seconds": timeout_seconds,
        "max_response_bytes": max_response_bytes,
        "execute_would_change": [str(manifest)] + [item["extracted_path"] for item in selected],
        "unchanged": ["raw", "chunk", "index"],
    }
