"""인덱스 준비 상태와 제거 대상을 기록하는 manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexDocumentState:
    document_id: str
    snapshot_sha256: str
    raw_sha256: str
    processed_sha256: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class Tombstone:
    document_id: str
    removed_at: str
    reason: str


@dataclass(slots=True)
class IndexManifest:
    version: int = 1
    generated_at: str = ""
    snapshot_sha256: str = ""
    source_manifest_sha256: str = ""
    documents: dict[str, IndexDocumentState] = field(default_factory=dict)
    tombstones: list[Tombstone] = field(default_factory=list)
    changed_document_ids: list[str] = field(default_factory=list)
    unchanged_document_ids: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["documents"] = {
            key: IndexDocumentState(**value)
            for key, value in payload.get("documents", {}).items()
        }
        payload["tombstones"] = [
            Tombstone(**value) for value in payload.get("tombstones", [])
        ]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "snapshot_sha256": self.snapshot_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "documents": {
                key: asdict(value) for key, value in sorted(self.documents.items())
            },
            "tombstones": [asdict(value) for value in self.tombstones],
            "changed_document_ids": self.changed_document_ids,
            "unchanged_document_ids": self.unchanged_document_ids,
            "stats": self.stats,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
