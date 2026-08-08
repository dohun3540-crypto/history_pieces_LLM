"""48건 임시 자료의 선별·수집·청크화·제거를 원자적으로 관리한다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.provisional.cleaning import CleanSection, clean_sections
from history_chatbot.provisional.remap import CURRENT_DETAIL_TEMPLATE
from history_chatbot.retrieval.sparse import BM25Searcher


USAGE_STATUS = "provisional_hackathon"
RIGHTS_STATUS = "unconfirmed"
USAGE_SCOPE = "noncommercial_hackathon_demo"
REVIEW_AFTER = "2026-08-31"
OFFICIAL_HOSTS = {
    "search.i815.or.kr",
    "www.heritage.go.kr",
    "heritage.go.kr",
    "www.mokpo.go.kr",
}


class ProvisionalSelectionError(RuntimeError):
    """대상 수 또는 제외 조건이 기대와 다를 때 발생한다."""


class _TextExtractor(HTMLParser):
    BLOCKED = {"script", "style", "nav", "footer", "header", "aside", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.main_depth = 0
        self.main_found = False
        self.parts: list[str] = []
        self.main_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in self.BLOCKED:
            self.depth += 1
        elif lowered == "main":
            self.main_found = True
            self.main_depth += 1
            self.main_parts.append("\n")
        elif not self.depth and lowered in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n")
            if self.main_depth:
                self.main_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self.BLOCKED and self.depth:
            self.depth -= 1
        elif lowered == "main" and self.main_depth:
            self.main_parts.append("\n")
            self.main_depth -= 1
        elif not self.depth and lowered in {"p", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")
            if self.main_depth:
                self.main_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.parts.append(data)
            if self.main_depth:
                self.main_parts.append(data)

    def text(self) -> str:
        source = self.main_parts if self.main_found else self.parts
        lines = []
        for line in "".join(source).replace("\r\n", "\n").replace("\r", "\n").splitlines():
            value = re.sub(r"[ \t]+", " ", line).strip()
            if value and value not in lines[-3:]:
                lines.append(value)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DryRunReport:
    selected: int
    excluded: int
    institutions: dict[str, int]
    excluded_source_ids: tuple[str, ...]
    estimated_chunks_min: int
    estimated_chunks_max: int


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    total_selected: int
    reused_existing: int
    pending_network: int
    previous_failed: int
    missing_raw: int
    hash_mismatch: int
    forced: int
    institution_requests: dict[str, int]
    request_urls: tuple[str, ...]
    estimated_max_gets: int


class ProvisionalDataService:
    def __init__(
        self,
        *,
        audit_path: Path = Path("data/source_audit/mokpo_public_candidates.jsonl"),
        root: Path = Path("data/provisional_hackathon"),
        index_root: Path = Path(".runtime/indexes/hackathon"),
        session_path: Path = Path(".runtime/hackathon/sessions.json"),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.audit_path = audit_path
        self.root = root
        self.raw_dir = root / "raw"
        self.processed_dir = root / "processed"
        self.manifest_dir = root / "manifests"
        self.manifest_path = self.manifest_dir / "sources.jsonl"
        self.removal_log = self.manifest_dir / "removals.jsonl"
        self.chunks_path = self.processed_dir / "chunks.jsonl"
        self.index_root = index_root
        self.session_path = session_path
        self.now = now or (lambda: datetime.now(UTC))

    def _audit_rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def select(self) -> tuple[list[dict], list[dict]]:
        selected: list[dict] = []
        excluded: list[dict] = []
        for item in self._audit_rows():
            host = (urlparse(str(item.get("source_url", ""))).hostname or "").lower()
            disallowed = (
                item.get("kogl_type") == "KOGL-4"
                or item.get("rights_grade") == "unsuitable_for_rag"
            )
            eligible = (
                not disallowed
                and host in OFFICIAL_HOSTS
                and item.get("body_exists") is True
                and bool(item.get("source_id"))
                and bool(item.get("source_url"))
                and not item.get("duplicate_of")
                and item.get("kogl_type") == "unknown"
                and item.get("rights_grade")
                in {"permission_request_required", "not_assessed"}
            )
            (selected if eligible else excluded).append(item)
        if len(selected) != 48:
            raise ProvisionalSelectionError(
                f"임시 해커톤 대상은 정확히 48건이어야 하나 {len(selected)}건입니다."
            )
        if len(excluded) != 3 or any(
            item.get("kogl_type") != "KOGL-4"
            or item.get("rights_grade") != "unsuitable_for_rag"
            for item in excluded
        ):
            raise ProvisionalSelectionError("제외 대상 3건이 모두 KOGL-4/D등급인지 확인하세요.")
        return selected, excluded

    def dry_run(self) -> DryRunReport:
        selected, excluded = self.select()
        return DryRunReport(
            selected=len(selected),
            excluded=len(excluded),
            institutions=dict(Counter(str(item["institution"]) for item in selected)),
            excluded_source_ids=tuple(str(item["source_id"]) for item in excluded),
            estimated_chunks_min=len(selected) * 2,
            estimated_chunks_max=len(selected) * 5,
        )

    def prepare_manifest(self) -> list[dict]:
        selected, _ = self.select()
        collected_at = self.now().isoformat()
        records = [
            {
                "source_id": item["source_id"],
                "source_title": item["title"],
                "institution": item["institution"],
                "source_url": item["source_url"],
                "canonical_url": item["canonical_url"],
                "official_record_id": item["official_record_id"],
                "usage_status": USAGE_STATUS,
                "rights_status": RIGHTS_STATUS,
                "usage_scope": USAGE_SCOPE,
                "allowed_for_rag": False,
                "allowed_for_training": False,
                "public_release_allowed": False,
                "removable": True,
                "expires_or_review_after": REVIEW_AFTER,
                "images_included": False,
                "original_text_publication_allowed": False,
                "provenance": "data/source_audit/mokpo_public_candidates.jsonl",
                "collected_at": collected_at,
                "content_hash": "pending_collection",
                "topic": item["topic"],
                "period": item["period"],
                "active": True,
                "collection_status": "metadata_prepared",
            }
            for item in selected
        ]
        self._atomic_jsonl(self.manifest_path, records)
        return records

    def collect(
        self,
        fetcher: Callable[[str], bytes] | None = None,
        *,
        delay_seconds: float = 1.0,
        max_bytes: int = 2_000_000,
        force: bool = False,
        source_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        records = self.load_manifest() or self.prepare_manifest()
        plan, decisions = self.collection_plan(
            force=force, source_id=source_id, records=records
        )
        if dry_run:
            return {
                "total_selected": plan.total_selected,
                "reused_existing": plan.reused_existing,
                "pending_network": plan.pending_network,
                "previous_failed": plan.previous_failed,
                "missing_raw": plan.missing_raw,
                "hash_mismatch": plan.hash_mismatch,
                "forced": plan.forced,
                "institution_requests": plan.institution_requests,
                "request_urls": list(plan.request_urls),
                "estimated_max_gets": plan.estimated_max_gets,
            }
        fetch = fetcher or self._fetch
        existing_chunks = self._load_chunks()
        targeted_ids = {str(item["source_id"]) for item, _ in decisions}
        chunks = [
            item for item in existing_chunks if str(item.get("source_id")) not in targeted_ids
        ]
        collected = 0
        failed = 0
        reused = 0
        network_requests = 0
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for record, decision in decisions:
            raw_path = self.raw_dir / f"{record['source_id']}.html"
            if decision == "reused":
                payload = raw_path.read_bytes()
                text = self._extract_text(payload)
                chunks.extend(self._chunks(record, text))
                record["collection_status"] = "reused"
                record["network_requested"] = False
                record["reused"] = True
                record["last_verified_at"] = self.now().isoformat()
                reused += 1
                continue
            try:
                network_requests += 1
                payload = fetch(str(record["source_url"]))
                if len(payload) > max_bytes:
                    raise ValueError("응답이 해커톤 제한 크기를 초과했습니다.")
                text = self._extract_text(payload)
                if len(text) < 120:
                    raise ValueError("역사 본문으로 사용할 충분한 텍스트가 없습니다.")
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                raw_hash = hashlib.sha256(payload).hexdigest()
                next_chunks = self._chunks(record, text)
                self._atomic_bytes(raw_path, payload)
                record["content_hash"] = content_hash
                record["raw_sha256"] = raw_hash
                record["collection_status"] = "collected"
                record["collected_at"] = self.now().isoformat()
                record["network_requested"] = True
                record["reused"] = False
                record.pop("collection_error", None)
                chunks.extend(next_chunks)
                collected += 1
            except Exception as error:  # 오류는 manifest에만 안전하게 기록
                record["collection_status"] = "failed"
                record["collection_error"] = type(error).__name__
                record["failure_reason"] = str(error)
                record["attempted_at"] = self.now().isoformat()
                record["network_requested"] = True
                record["reused"] = False
                failed += 1
            if fetcher is None and delay_seconds:
                time.sleep(delay_seconds)
        self._atomic_jsonl(self.manifest_path, records)
        resolved_chunks = self._deduplicate(chunks)
        self._atomic_jsonl(self.chunks_path, resolved_chunks)
        return {
            "collected": collected,
            "reused": reused,
            "failed": failed,
            "network_requests": network_requests,
            "chunks": len(resolved_chunks),
        }

    def collection_plan(
        self,
        *,
        force: bool = False,
        source_id: str | None = None,
        records: list[dict] | None = None,
    ) -> tuple[CollectionPlan, list[tuple[dict, str]]]:
        selected, _ = self.select()
        selected_by_id = {str(item["source_id"]): item for item in selected}
        manifest = records if records is not None else self.load_manifest()
        if not manifest:
            manifest = [
                {
                    "source_id": item["source_id"],
                    "institution": item["institution"],
                    "source_url": item["source_url"],
                    "canonical_url": item["canonical_url"],
                    "collection_status": "metadata_prepared",
                    "active": True,
                }
                for item in selected
            ]
        if source_id is not None and source_id not in selected_by_id:
            raise ProvisionalSelectionError(f"대상 source_id를 찾을 수 없습니다: {source_id}")

        decisions: list[tuple[dict, str]] = []
        reason_counts: Counter[str] = Counter()
        institution_requests: Counter[str] = Counter()
        urls: list[str] = []
        for record in manifest:
            current_id = str(record.get("source_id", ""))
            if current_id not in selected_by_id or not record.get("active", True):
                continue
            if source_id is not None and current_id != source_id:
                continue
            audit_item = selected_by_id[current_id]
            reason = self._collection_decision(record, audit_item, force=force)
            decisions.append((record, reason))
            if reason == "reused":
                continue
            reason_counts[reason] += 1
            institution_requests[str(record.get("institution", ""))] += 1
            urls.append(str(record.get("source_url", "")))

        reused = sum(reason == "reused" for _, reason in decisions)
        pending = len(decisions) - reused
        plan = CollectionPlan(
            total_selected=len(decisions),
            reused_existing=reused,
            pending_network=pending,
            previous_failed=reason_counts["previous_failed"],
            missing_raw=reason_counts["missing_raw"],
            hash_mismatch=reason_counts["hash_mismatch"],
            forced=reason_counts["forced"],
            institution_requests=dict(institution_requests),
            request_urls=tuple(urls),
            estimated_max_gets=pending,
        )
        return plan, decisions

    def _collection_decision(
        self, record: dict, audit_item: dict, *, force: bool
    ) -> str:
        if force:
            return "forced"
        if record.get("collection_status") not in {"collected", "reused", "skipped_existing"}:
            return (
                "previous_failed"
                if record.get("collection_status") == "failed"
                else "missing_raw"
            )
        expected_source_url = str(audit_item.get("source_url"))
        expected_canonical_url = str(audit_item.get("canonical_url"))
        record_id = str(record.get("official_record_id", ""))
        if record_id.startswith("i815-person-"):
            current_url = CURRENT_DETAIL_TEMPLATE.format(
                record_id=record_id.removeprefix("i815-person-")
            )
            if (
                str(record.get("source_url")) == current_url
                and str(record.get("canonical_url")) == current_url
            ):
                expected_source_url = current_url
                expected_canonical_url = current_url
        if (
            not record.get("source_id")
            or not record.get("canonical_url")
            or not record.get("content_hash")
            or str(record.get("canonical_url")) != expected_canonical_url
            or str(record.get("source_url")) != expected_source_url
        ):
            return "hash_mismatch"
        if str(record.get("expires_or_review_after", "")) < self.now().date().isoformat():
            return "hash_mismatch"
        raw_path = self.raw_dir / f"{record['source_id']}.html"
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            return "missing_raw"
        try:
            payload = raw_path.read_bytes()
            raw_hash = hashlib.sha256(payload).hexdigest()
            if record.get("raw_sha256"):
                return "reused" if raw_hash == record["raw_sha256"] else "hash_mismatch"
            text_hash = hashlib.sha256(
                self._extract_text(payload).encode("utf-8")
            ).hexdigest()
            return "reused" if text_hash == record["content_hash"] else "hash_mismatch"
        except (OSError, UnicodeError, ValueError):
            return "hash_mismatch"

    @staticmethod
    def _extract_text(payload: bytes) -> str:
        parser = _TextExtractor()
        parser.feed(payload.decode("utf-8", errors="replace"))
        return parser.text()

    def _load_chunks(self) -> list[dict]:
        if not self.chunks_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def reprocess_local(self) -> dict[str, int]:
        """Rebuild processed chunks exclusively from already stored raw HTML."""

        selected, _ = self.select()
        selected_ids = {str(item["source_id"]) for item in selected}
        records = [
            record
            for record in self.load_manifest()
            if record.get("active", True)
            and str(record.get("source_id", "")) in selected_ids
        ]
        chunks: list[dict] = []
        for record in records:
            raw_path = self.raw_dir / f"{record['source_id']}.html"
            if not raw_path.is_file():
                raise FileNotFoundError(f"로컬 raw 원문이 없습니다: {record['source_id']}")
            payload = raw_path.read_bytes()
            expected_raw_hash = str(record.get("raw_sha256", ""))
            if expected_raw_hash and hashlib.sha256(payload).hexdigest() != expected_raw_hash:
                raise ValueError(
                    f"raw 원문 hash가 manifest와 다릅니다: {record['source_id']}"
                )
            chunks.extend(self._chunks(record, self._extract_text(payload)))
        resolved = self._deduplicate(chunks)
        self._atomic_jsonl(self.chunks_path, resolved)
        return {
            "documents": len(records),
            "chunks": len(resolved),
            "network_requests": 0,
        }

    @staticmethod
    def _fetch(url: str) -> bytes:
        host = (urlparse(url).hostname or "").lower()
        if host not in OFFICIAL_HOSTS:
            raise ValueError("공식 허용 도메인이 아닙니다.")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MokpoHistoryHackathonPilot/0.1 (+noncommercial research)"},
        )
        retries = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    final_host = (
                        urlparse(str(response.geturl())).hostname or ""
                    ).lower()
                    if final_host not in OFFICIAL_HOSTS:
                        raise ValueError("redirected outside the approved official domains")
                    content_type = response.headers.get_content_type().lower()
                    if content_type not in {"text/html", "application/xhtml+xml"}:
                        raise ValueError(f"unsupported content type: {content_type}")
                    return response.read(2_000_001)
            except urllib.error.HTTPError as error:
                retry_limit = 1 if error.code == 429 else 2 if 500 <= error.code <= 599 else 0
                if retries >= retry_limit:
                    raise
                retries += 1
                retry_after = error.headers.get("Retry-After") if error.headers else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 1.0)
            except (urllib.error.URLError, TimeoutError):
                if retries >= 2:
                    raise
                retries += 1
                time.sleep(1.0)

    def _chunks(
        self, record: dict, text: str, maximum: int = 900, minimum: int = 120
    ) -> list[dict]:
        grouped: list[tuple[CleanSection, str]] = []
        for section in clean_sections(record, text):
            grouped.extend(
                (section, value)
                for value in self._pack_section(
                    section, maximum=maximum, minimum=minimum
                )
            )
        result = []
        for sequence, (section, value) in enumerate(grouped):
            searchable_value = f"{section.title}\n{value}".strip()
            digest = hashlib.sha256(searchable_value.encode("utf-8")).hexdigest()
            chunk = {
                "document_id": record["source_id"],
                "chunk_id": f"{record['source_id']}::provisional-{sequence:04d}",
                "source_id": record["source_id"],
                "source_title": record["source_title"],
                "title": f"{record['source_title']} — {section.title}",
                "institution": record["institution"],
                "publisher": record["institution"],
                "source_url": record["source_url"],
                "usage_status": USAGE_STATUS,
                "rights_status": RIGHTS_STATUS,
                "usage_scope": USAGE_SCOPE,
                "allowed_for_rag": False,
                "allowed_for_training": False,
                "public_release_allowed": False,
                "text": searchable_value,
                "section_title": section.title,
                "sequence": sequence,
                "content_hash": digest,
                "content_sha256": digest,
                "expires_or_review_after": REVIEW_AFTER,
                "active": True,
            }
            chunk.update(section.metadata)
            result.append(chunk)
        return result

    @staticmethod
    def _pack_section(
        section: CleanSection, *, maximum: int, minimum: int
    ) -> list[str]:
        paragraphs: list[str] = []
        for paragraph in section.paragraphs:
            if len(paragraph) <= maximum:
                paragraphs.append(paragraph)
                continue
            sentences = [
                item.strip()
                for item in re.split(r"(?<=[.!?。])\s+", paragraph)
                if item.strip()
            ]
            paragraphs.extend(sentences or [paragraph])

        grouped: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > maximum:
                grouped.append(current)
                current = ""
            current = f"{current}\n{paragraph}".strip()
        if current:
            if grouped and len(current) < minimum:
                grouped[-1] = f"{grouped[-1]}\n{current}"
            else:
                grouped.append(current)
        return grouped

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        seen: set[str] = set()
        result = []
        for chunk in chunks:
            key = str(chunk["content_hash"])
            if key in seen:
                continue
            seen.add(key)
            result.append(chunk)
        return result

    def load_manifest(self) -> list[dict]:
        if not self.manifest_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def remove(
        self,
        *,
        source_id: str | None = None,
        institution: str | None = None,
        purge_all: bool = False,
        reason: str = "manual_removal",
    ) -> tuple[str, ...]:
        records = self.load_manifest()
        before = [dict(item) for item in records]
        chunks_before = (
            self.chunks_path.read_bytes() if self.chunks_path.is_file() else None
        )
        index_file = self.index_root / "hashing-v1--builtin.json"
        index_before = index_file.read_bytes() if index_file.is_file() else None
        removal_log_before = (
            self.removal_log.read_bytes() if self.removal_log.is_file() else None
        )
        session_before = (
            self.session_path.read_bytes() if self.session_path.is_file() else None
        )
        removed: list[str] = []
        for item in records:
            matched = purge_all or (
                source_id is not None and item["source_id"] == source_id
            ) or (
                institution is not None and item["institution"] == institution
            )
            if matched and item.get("active", True):
                item["active"] = False
                item["disabled_at"] = self.now().isoformat()
                item["disabled_reason"] = reason
                removed.append(str(item["source_id"]))
        if not removed:
            return ()
        try:
            self._atomic_jsonl(self.manifest_path, records)
            self._filter_chunks(set(removed))
            self.rebuild_index()
            self._append_removal_log(removed, reason)
            if self.session_path.is_file():
                self.session_path.unlink()
        except Exception:
            self._atomic_jsonl(self.manifest_path, before)
            if chunks_before is not None:
                self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
                self.chunks_path.write_bytes(chunks_before)
            self._restore_optional_file(index_file, index_before)
            self._restore_optional_file(self.removal_log, removal_log_before)
            self._restore_optional_file(self.session_path, session_before)
            raise
        return tuple(removed)

    def expire(self, on_date: str | None = None) -> tuple[str, ...]:
        cutoff = on_date or self.now().date().isoformat()
        records = self.load_manifest()
        expired = [
            str(item["source_id"])
            for item in records
            if item.get("active", True)
            and str(item.get("expires_or_review_after", "")) < cutoff
        ]
        removed: list[str] = []
        for source_id in expired:
            removed.extend(self.remove(source_id=source_id, reason="expired"))
        return tuple(removed)

    def _filter_chunks(self, removed: set[str]) -> None:
        if not self.chunks_path.is_file():
            return
        chunks = [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._atomic_jsonl(
            self.chunks_path,
            [item for item in chunks if item.get("source_id") not in removed],
        )

    def _append_removal_log(self, source_ids: list[str], reason: str) -> None:
        self.removal_log.parent.mkdir(parents=True, exist_ok=True)
        with self.removal_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "removed_at": self.now().isoformat(),
                        "source_ids": source_ids,
                        "reason": reason,
                        "metadata_retained": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    @staticmethod
    def _atomic_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    @classmethod
    def _restore_optional_file(cls, path: Path, payload: bytes | None) -> None:
        if payload is None:
            if path.is_file():
                path.unlink()
            return
        cls._atomic_bytes(path, payload)

    def source_snapshot(self) -> str:
        active = [item for item in self.load_manifest() if item.get("active", True)]
        return stable_json_hash(active)

    def rebuild_index(self):
        if not self.chunks_path.is_file():
            return None
        from history_chatbot.retrieval.service import (
            HybridRetrievalService,
            RetrievalConfig,
        )

        temporary_root = self.index_root.parent / (
            f".{self.index_root.name}-build-{uuid.uuid4().hex}"
        )
        active_file = self.index_root / "hashing-v1--builtin.json"
        staged_file = temporary_root / active_file.name
        try:
            temporary_root.mkdir(parents=True, exist_ok=False)
            if active_file.is_file():
                shutil.copy2(active_file, staged_file)
            retrieval = HybridRetrievalService(
                RetrievalConfig(
                    runtime_mode="hackathon",
                    provisional_chunks_path=self.chunks_path,
                    local_storage_path=temporary_root,
                )
            )
            report = retrieval.build_index()
            self._validate_staged_index(retrieval)
            self._before_index_swap(staged_file)
            self.index_root.mkdir(parents=True, exist_ok=True)
            if active_file.is_file():
                metadata = json.loads(active_file.read_text(encoding="utf-8")).get(
                    "metadata", {}
                )
                snapshot = str(metadata.get("source_snapshot", ""))
                if snapshot:
                    backup = (
                        self.index_root
                        / "snapshots"
                        / f"{active_file.stem}--{snapshot}.json"
                    )
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if not backup.exists():
                        shutil.copy2(active_file, backup)
            os.replace(staged_file, active_file)
            return replace(report, index_path=active_file)
        finally:
            if temporary_root.is_dir():
                shutil.rmtree(temporary_root)

    def _validate_staged_index(self, retrieval) -> None:
        errors = retrieval.validate_index()
        if errors:
            raise RuntimeError("; ".join(errors))
        chunks = retrieval.store.chunks()
        entries = retrieval.store.entries()
        metadata = retrieval.store.metadata()
        if len(entries) != len(chunks):
            raise RuntimeError("staged index의 chunk/vector 수가 일치하지 않습니다.")
        if int(metadata.get("chunk_count", -1)) != len(chunks):
            raise RuntimeError("staged index metadata의 chunk_count가 일치하지 않습니다.")
        if metadata.get("source_snapshot") != stable_json_hash(
            [chunk.payload for chunk in chunks]
        ):
            raise RuntimeError("staged index source snapshot이 입력 청크와 일치하지 않습니다.")
        if entries:
            dimensions = {len(vector) for _, vector in entries}
            if len(dimensions) != 1 or 0 in dimensions:
                raise RuntimeError("staged dense vector가 불완전합니다.")
            BM25Searcher(chunks).search(chunks[0].text, 1)

    def _before_index_swap(self, staged_file: Path) -> None:
        if not staged_file.is_file() or staged_file.stat().st_size == 0:
            raise RuntimeError("교체할 staged index가 없습니다.")

    def purge_runtime_index(self) -> None:
        if self.index_root.is_dir():
            shutil.rmtree(self.index_root)
