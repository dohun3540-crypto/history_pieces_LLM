"""검수 완료 청크를 중복 제거하고 인덱스 준비 산출물로 만든다."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from history_chatbot.indexing.loader import (
    IndexChunk,
    LoadReport,
    ReviewedChunkLoader,
)
from history_chatbot.indexing.manifest import (
    IndexDocumentState,
    IndexManifest,
    Tombstone,
)
from history_chatbot.indexing.snapshot import sha256_file, stable_json_hash


@dataclass(frozen=True, slots=True)
class PrepareResult:
    eligible_documents: int
    rejected_documents: int
    chunk_count: int
    duplicate_chunks: int
    changed_document_ids: tuple[str, ...]
    unchanged_document_ids: tuple[str, ...]
    tombstone_document_ids: tuple[str, ...]
    chunks_path: str
    manifest_path: str


class IndexBuilder:
    def __init__(
        self,
        loader: ReviewedChunkLoader,
        source_manifest_path: Path,
        output_dir: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.loader = loader
        self.source_manifest_path = source_manifest_path
        self.output_dir = output_dir
        self.now = now or (lambda: datetime.now(timezone.utc).astimezone())

    @property
    def chunks_path(self) -> Path:
        return self.output_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "index_manifest.json"

    def prepare(self) -> PrepareResult:
        report = self.loader.load()
        previous = IndexManifest.load(self.manifest_path)
        chunks, duplicate_count = self._deduplicate(report)
        current_documents = {
            loaded.document.document_id: IndexDocumentState(
                document_id=loaded.document.document_id,
                snapshot_sha256=loaded.snapshot_sha256,
                raw_sha256=loaded.raw_sha256,
                processed_sha256=loaded.processed_sha256,
                chunk_count=sum(
                    chunk.document_id == loaded.document.document_id for chunk in chunks
                ),
            )
            for loaded in report.eligible
        }
        changed = sorted(
            document_id
            for document_id, state in current_documents.items()
            if previous.documents.get(document_id) != state
        )
        unchanged = sorted(set(current_documents) - set(changed))
        removed = sorted(set(previous.documents) - set(current_documents))
        occurred_at = self.now().isoformat()
        prior_tombstones = {
            item.document_id: item
            for item in previous.tombstones
            if item.document_id not in current_documents
        }
        for document_id in removed:
            prior_tombstones[document_id] = Tombstone(
                document_id,
                occurred_at,
                "manifest에서 제거되었거나 더 이상 RAG 사용 조건을 충족하지 않음",
            )

        records = [chunk.to_dict() for chunk in chunks]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest = IndexManifest(
            generated_at=occurred_at,
            snapshot_sha256=stable_json_hash(records),
            source_manifest_sha256=sha256_file(self.source_manifest_path),
            documents=current_documents,
            tombstones=[
                prior_tombstones[key] for key in sorted(prior_tombstones)
            ],
            changed_document_ids=changed,
            unchanged_document_ids=unchanged,
            stats={
                "eligible_documents": len(report.eligible),
                "rejected_documents": len(report.rejected),
                "chunks": len(records),
                "duplicate_chunks_removed": duplicate_count,
            },
        )
        manifest.save(self.manifest_path)
        return PrepareResult(
            len(report.eligible),
            len(report.rejected),
            len(records),
            duplicate_count,
            tuple(changed),
            tuple(unchanged),
            tuple(removed),
            str(self.chunks_path),
            str(self.manifest_path),
        )

    @staticmethod
    def _deduplicate(report: LoadReport) -> tuple[list[IndexChunk], int]:
        chunks: list[IndexChunk] = []
        seen_chunk_ids: dict[str, str] = {}
        seen_content: set[tuple[str, str]] = set()
        duplicates = 0
        for loaded in sorted(report.eligible, key=lambda item: item.document.document_id):
            for chunk in sorted(
                loaded.chunks, key=lambda item: (item.chunk_index, item.chunk_id)
            ):
                existing_hash = seen_chunk_ids.get(chunk.chunk_id)
                if existing_hash is not None:
                    if existing_hash != chunk.content_sha256:
                        raise ValueError(f"동일 chunk_id의 본문이 다릅니다: {chunk.chunk_id}")
                    duplicates += 1
                    continue
                content_key = (
                    chunk.document_id,
                    re.sub(r"\s+", " ", chunk.text).strip(),
                )
                if content_key in seen_content:
                    duplicates += 1
                    continue
                seen_chunk_ids[chunk.chunk_id] = chunk.content_sha256
                seen_content.add(content_key)
                chunks.append(chunk)
        return chunks, duplicates

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.chunks_path.is_file() or not self.manifest_path.is_file():
            return ["index_ready 산출물이 없습니다. prepare를 먼저 실행하세요."]
        try:
            manifest = IndexManifest.load(self.manifest_path)
            report = self.loader.load()
            eligible_ids = {
                loaded.document.document_id for loaded in report.eligible
            }
            records = [
                json.loads(line)
                for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return [str(error)]

        seen_ids: set[str] = set()
        seen_content: set[tuple[str, str]] = set()
        for index, record in enumerate(records, start=1):
            for name in (
                "document_id",
                "chunk_id",
                "title",
                "publisher",
                "source_url",
                "text",
            ):
                if not str(record.get(name, "")).strip():
                    errors.append(f"청크 {index} 필수 필드 누락: {name}")
            document_id = str(record.get("document_id", ""))
            chunk_id = str(record.get("chunk_id", ""))
            if document_id not in eligible_ids:
                errors.append(f"허용되지 않은 문서가 index_ready에 포함됨: {document_id}")
            if chunk_id in seen_ids:
                errors.append(f"중복 chunk_id: {chunk_id}")
            seen_ids.add(chunk_id)
            content_key = (
                document_id,
                re.sub(r"\s+", " ", str(record.get("text", ""))).strip(),
            )
            if content_key in seen_content:
                errors.append(f"중복 청크 본문: {chunk_id}")
            seen_content.add(content_key)

        if stable_json_hash(records) != manifest.snapshot_sha256:
            errors.append("chunks.jsonl 스냅샷 해시가 manifest와 일치하지 않습니다.")
        if set(manifest.documents) != eligible_ids:
            errors.append("활성 문서 목록이 현재 eligibility 결과와 일치하지 않습니다.")
        tombstoned = {item.document_id for item in manifest.tombstones}
        if tombstoned & {str(record.get("document_id")) for record in records}:
            errors.append("tombstone 문서가 chunks.jsonl에 포함되어 있습니다.")
        return errors
