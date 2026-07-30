"""48건 임시 자료의 선별·수집·청크화·제거를 원자적으로 관리한다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from history_chatbot.indexing.snapshot import stable_json_hash


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
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if tag.lower() in self.BLOCKED:
            self.depth += 1
        elif not self.depth and tag.lower() in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCKED and self.depth:
            self.depth -= 1
        elif not self.depth and tag.lower() in {"p", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = []
        for line in "".join(self.parts).replace("\r\n", "\n").replace("\r", "\n").splitlines():
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
    ) -> dict[str, int]:
        records = self.load_manifest() or self.prepare_manifest()
        fetch = fetcher or self._fetch
        chunks: list[dict] = []
        collected = 0
        failed = 0
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            if not record.get("active", True):
                continue
            try:
                payload = fetch(str(record["source_url"]))
                if len(payload) > max_bytes:
                    raise ValueError("응답이 해커톤 제한 크기를 초과했습니다.")
                raw_path = self.raw_dir / f"{record['source_id']}.html"
                raw_path.write_bytes(payload)
                parser = _TextExtractor()
                parser.feed(payload.decode("utf-8", errors="replace"))
                text = parser.text()
                if len(text) < 120:
                    raise ValueError("역사 본문으로 사용할 충분한 텍스트가 없습니다.")
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record["content_hash"] = content_hash
                record["collection_status"] = "collected"
                record["collected_at"] = self.now().isoformat()
                chunks.extend(self._chunks(record, text))
                collected += 1
            except Exception as error:  # 오류는 manifest에만 안전하게 기록
                record["collection_status"] = "failed"
                record["collection_error"] = type(error).__name__
                failed += 1
            if fetcher is None and delay_seconds:
                time.sleep(delay_seconds)
        self._atomic_jsonl(self.manifest_path, records)
        self._atomic_jsonl(self.chunks_path, self._deduplicate(chunks))
        return {"collected": collected, "failed": failed, "chunks": len(chunks)}

    @staticmethod
    def _fetch(url: str) -> bytes:
        host = (urlparse(url).hostname or "").lower()
        if host not in OFFICIAL_HOSTS:
            raise ValueError("공식 허용 도메인이 아닙니다.")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MokpoHistoryHackathonPilot/0.1 (+noncommercial research)"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.read(2_000_001)

    def _chunks(self, record: dict, text: str, maximum: int = 900, minimum: int = 120) -> list[dict]:
        paragraphs = [item.strip() for item in re.split(r"\n{1,}", text) if item.strip()]
        grouped: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > maximum:
                if len(current) >= minimum:
                    grouped.append(current)
                current = ""
            current = f"{current}\n{paragraph}".strip()
        if current:
            if grouped and len(current) < minimum:
                grouped[-1] = f"{grouped[-1]}\n{current}"
            else:
                grouped.append(current)
        result = []
        for sequence, value in enumerate(grouped):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            result.append(
                {
                    "document_id": record["source_id"],
                    "chunk_id": f"{record['source_id']}::provisional-{sequence:04d}",
                    "source_id": record["source_id"],
                    "source_title": record["source_title"],
                    "title": record["source_title"],
                    "institution": record["institution"],
                    "publisher": record["institution"],
                    "source_url": record["source_url"],
                    "usage_status": USAGE_STATUS,
                    "rights_status": RIGHTS_STATUS,
                    "usage_scope": USAGE_SCOPE,
                    "allowed_for_rag": False,
                    "allowed_for_training": False,
                    "public_release_allowed": False,
                    "text": value,
                    "section_title": "",
                    "sequence": sequence,
                    "content_hash": digest,
                    "content_sha256": digest,
                    "expires_or_review_after": REVIEW_AFTER,
                    "active": True,
                }
            )
        return result

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
            self._append_removal_log(removed, reason)
            self.rebuild_index()
            if self.session_path.is_file():
                self.session_path.unlink()
        except Exception:
            self._atomic_jsonl(self.manifest_path, before)
            if chunks_before is not None:
                self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
                self.chunks_path.write_bytes(chunks_before)
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

        retrieval = HybridRetrievalService(
            RetrievalConfig(
                runtime_mode="hackathon",
                provisional_chunks_path=self.chunks_path,
                local_storage_path=self.index_root,
            )
        )
        return retrieval.build_index()

    def purge_runtime_index(self) -> None:
        if self.index_root.is_dir():
            shutil.rmtree(self.index_root)
