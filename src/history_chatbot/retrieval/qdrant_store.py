"""로컬 JSON 벡터 저장소와 선택적 Qdrant 확장 지점."""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import UTC, datetime
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk, VectorStore


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("질의 벡터와 인덱스 벡터 차원이 다릅니다.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class LocalJsonVectorStore(VectorStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: list[tuple[RetrievalChunk, list[float]]] = []
        self._metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._metadata = dict(payload.get("metadata", {}))
        self._entries = [
            (RetrievalChunk.from_record(item["chunk"]), list(item["vector"]))
            for item in payload.get("entries", [])
        ]

    def replace(
        self,
        entries: Iterable[tuple[RetrievalChunk, Sequence[float]]],
        *,
        model_id: str,
        revision: str,
        source_snapshot: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        next_index_version = int(self._metadata.get("index_version", 0)) + 1
        if self.path.is_file() and self._metadata.get("source_snapshot"):
            snapshots = self.path.parent / "snapshots"
            snapshots.mkdir(parents=True, exist_ok=True)
            backup = snapshots / f"{self.path.stem}--{self._metadata['source_snapshot']}.json"
            if not backup.exists():
                shutil.copy2(self.path, backup)
        self._entries = [(chunk, list(vector)) for chunk, vector in entries]
        self._metadata = {
            "format_version": 1,
            "model_id": model_id,
            "revision": revision,
            "source_snapshot": source_snapshot,
            "chunk_count": len(self._entries),
            "index_version": next_index_version,
            "created_at": datetime.now(UTC).isoformat(),
            **(extra_metadata or {}),
        }
        payload = {
            "metadata": self._metadata,
            "entries": [
                {"chunk": chunk.payload, "vector": vector}
                for chunk, vector in self._entries
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def rollback(self, source_snapshot: str) -> None:
        backup = (
            self.path.parent
            / "snapshots"
            / f"{self.path.stem}--{source_snapshot}.json"
        )
        if not backup.is_file():
            raise ValueError(f"인덱스 스냅샷을 찾을 수 없습니다: {source_snapshot}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".rollback")
        shutil.copy2(backup, temporary)
        os.replace(temporary, self.path)
        self._entries = []
        self._metadata = {}
        self._load()

    def search(self, vector: Sequence[float], limit: int) -> list[RankedChunk]:
        scored = [
            RankedChunk(chunk, _cosine(vector, stored), ("dense",), dense_score=_cosine(vector, stored))
            for chunk, stored in self._entries
        ]
        return sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]

    def chunks(self) -> list[RetrievalChunk]:
        return [chunk for chunk, _ in self._entries]

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def entries(self) -> list[tuple[RetrievalChunk, list[float]]]:
        return [(chunk, list(vector)) for chunk, vector in self._entries]


class QdrantVectorStore(VectorStore):
    """Qdrant Client Local/Server용 명시적 선택 지점.

    선택 의존성을 설치하기 전에는 생성 단계에서 즉시 오류를 낸다.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        try:
            import qdrant_client  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Qdrant 백엔드는 선택 의존성 qdrant-client 설치 후 사용할 수 있습니다."
            ) from error
        raise NotImplementedError("Qdrant 영속 어댑터는 모델 승인 후 활성화합니다.")

    def replace(self, entries: Iterable[tuple[RetrievalChunk, Sequence[float]]], **kwargs: str) -> None:
        raise NotImplementedError

    def search(self, vector: Sequence[float], limit: int) -> list[RankedChunk]:
        raise NotImplementedError

    def chunks(self) -> list[RetrievalChunk]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError
