"""index_ready 전용 하이브리드 검색 인덱스 빌드와 질의 서비스."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.retrieval.base import DenseEncoder, RankedChunk, RetrievalChunk
from history_chatbot.retrieval.dense import DenseSearcher, HashingDenseEncoder
from history_chatbot.retrieval.fusion import reciprocal_rank_fusion
from history_chatbot.retrieval.qdrant_store import LocalJsonVectorStore
from history_chatbot.retrieval.query_normalizer import normalize_query
from history_chatbot.retrieval.reranker import NoOpReranker
from history_chatbot.retrieval.sparse import BM25Searcher
from history_chatbot.retrieval.thresholds import apply_thresholds


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    backend: str = "local_json"
    embedding_model: str = "hashing-v1"
    embedding_revision: str = "builtin"
    sparse_model: str = "bm25"
    reranker_model: str = "none"
    dense_top_k: int = 12
    sparse_top_k: int = 12
    final_top_k: int = 5
    minimum_score: float = 0.20
    minimum_dense_score: float = 0.72
    max_chunks_per_document: int = 2
    local_storage_path: Path = Path("data/retrieval_index")
    index_ready_path: Path = Path("data/index_ready")

    def validate(self) -> None:
        if self.backend != "local_json":
            raise ValueError("현재 다운로드 없는 기본 백엔드는 local_json만 지원합니다.")
        if self.embedding_model != "hashing-v1":
            raise ValueError(
                "실제 임베딩 모델은 아직 설치되지 않았습니다. 모델 승인 후 연결하세요."
            )
        for name in (
            "dense_top_k",
            "sparse_top_k",
            "final_top_k",
            "max_chunks_per_document",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}은 양수여야 합니다.")
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score는 0~1이어야 합니다.")
        if not -1 <= self.minimum_dense_score <= 1:
            raise ValueError("minimum_dense_score는 -1~1이어야 합니다.")

    @classmethod
    def load(cls, path: Path = Path("configs/retrieval.yaml")) -> "RetrievalConfig":
        if not path.is_file():
            return cls()
        values: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.endswith(":") or line.startswith("#"):
                continue
            key, raw_value = (part.strip() for part in line.split(":", 1))
            value = raw_value.strip('"')
            if key in {"dense_top_k", "sparse_top_k", "final_top_k", "max_chunks_per_document"}:
                values[key] = int(value)
            elif key in {"minimum_score", "minimum_dense_score"}:
                values[key] = float(value)
            elif key in {"local_storage_path", "index_ready_path"}:
                values[key] = Path(value)
            else:
                values[key] = value
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class BuildReport:
    chunks: int
    embedded_chunks: int
    reused_chunks: int
    removed_chunks: int
    source_snapshot: str
    model_id: str
    revision: str
    index_path: Path


class IndexReadyReader:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def chunks_path(self) -> Path:
        return self.directory / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "index_manifest.json"

    def load(self) -> tuple[list[RetrievalChunk], str]:
        if not self.chunks_path.is_file() or not self.manifest_path.is_file():
            return [], ""
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in self.chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = str(manifest.get("snapshot_sha256", ""))
        if stable_json_hash(records) != expected:
            raise ValueError("index_ready 청크 해시가 manifest와 일치하지 않습니다.")
        active_documents = set(manifest.get("documents", {}))
        tombstones = {
            str(item.get("document_id", "")) for item in manifest.get("tombstones", [])
        }
        chunks: list[RetrievalChunk] = []
        seen: set[str] = set()
        seen_content: set[str] = set()
        for record in records:
            document_id = str(record.get("document_id", ""))
            if document_id not in active_documents or document_id in tombstones:
                raise ValueError(f"비활성 문서가 index_ready에 포함됨: {document_id}")
            if str(record.get("review_status", "reviewed")) != "reviewed":
                raise ValueError(f"검수 전 문서가 index_ready에 포함됨: {document_id}")
            if not str(record.get("reviewed_by", "")).strip() or not str(
                record.get("reviewed_at", "")
            ).strip():
                raise ValueError(f"검수 이력이 없는 문서가 index_ready에 포함됨: {document_id}")
            chunk = RetrievalChunk.from_record(record)
            content_key = stable_json_hash(" ".join(chunk.text.split()))
            if chunk.chunk_id in seen or content_key in seen_content:
                continue
            seen.add(chunk.chunk_id)
            seen_content.add(content_key)
            chunks.append(chunk)
        return chunks, expected


class HybridRetrievalService:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        encoder: DenseEncoder | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.encoder = encoder or HashingDenseEncoder()
        self.store = LocalJsonVectorStore(
            config.local_storage_path / self.index_filename
        )
        self.reranker = NoOpReranker()

    @property
    def index_filename(self) -> str:
        safe_model = self.config.embedding_model.replace("/", "--")
        safe_revision = self.config.embedding_revision.replace("/", "-")
        return f"{safe_model}--{safe_revision}.json"

    def build_index(self, *, force: bool = False) -> BuildReport:
        chunks, snapshot = IndexReadyReader(self.config.index_ready_path).load()
        previous = self.store.entries()
        can_reuse = (
            not force
            and self.store.metadata().get("model_id") == self.encoder.model_id
            and self.store.metadata().get("revision") == self.encoder.revision
        )
        previous_vectors = {
            (
                chunk.chunk_id,
                str(chunk.payload.get("content_sha256", "")),
            ): vector
            for chunk, vector in previous
        } if can_reuse else {}
        vectors: list[list[float] | None] = []
        pending_texts: list[str] = []
        pending_positions: list[int] = []
        reused = 0
        for chunk in chunks:
            key = (chunk.chunk_id, str(chunk.payload.get("content_sha256", "")))
            vector = previous_vectors.get(key)
            vectors.append(vector)
            if vector is None:
                pending_positions.append(len(vectors) - 1)
                pending_texts.append(chunk.text)
            else:
                reused += 1
        encoded = self.encoder.encode(pending_texts, is_query=False)
        for position, vector in zip(pending_positions, encoded):
            vectors[position] = vector
        resolved_vectors = [vector for vector in vectors if vector is not None]
        self.store.replace(
            zip(chunks, resolved_vectors),
            model_id=self.encoder.model_id,
            revision=self.encoder.revision,
            source_snapshot=snapshot,
        )
        return BuildReport(
            len(chunks),
            len(encoded),
            reused,
            max(0, len(previous) - reused),
            snapshot,
            self.encoder.model_id,
            self.encoder.revision,
            self.store.path,
        )

    def validate_index(self) -> list[str]:
        metadata = self.store.metadata()
        if not metadata:
            return ["검색 인덱스가 없습니다. build-index를 먼저 실행하세요."]
        errors: list[str] = []
        if metadata.get("model_id") != self.encoder.model_id:
            errors.append("임베딩 모델 ID가 검색 인덱스와 일치하지 않습니다.")
        if metadata.get("revision") != self.encoder.revision:
            errors.append("임베딩 모델 revision이 검색 인덱스와 일치하지 않습니다.")
        _, current_snapshot = IndexReadyReader(self.config.index_ready_path).load()
        if metadata.get("source_snapshot") != current_snapshot:
            errors.append("index_ready 스냅샷이 변경되어 재색인이 필요합니다.")
        return errors

    def search(self, query_text: str) -> list[RankedChunk]:
        if self.validate_index():
            return []
        query = normalize_query(query_text)
        dense = DenseSearcher(self.encoder, self.store).search(
            query.normalized, self.config.dense_top_k
        )
        sparse = BM25Searcher(self.store.chunks()).search(
            query.normalized, self.config.sparse_top_k
        )
        fused = reciprocal_rank_fusion(dense, sparse)
        reranked = self.reranker.rerank(query.normalized, fused)
        return apply_thresholds(
            query,
            reranked,
            minimum_score=self.config.minimum_score,
            minimum_dense_score=self.config.minimum_dense_score,
            max_chunks_per_document=self.config.max_chunks_per_document,
            final_top_k=self.config.final_top_k,
        )

    def status(self) -> dict[str, Any]:
        metadata = self.store.metadata()
        return {
            "ready": bool(metadata) and not self.validate_index(),
            "chunks": len(self.store.chunks()),
            "model_id": metadata.get("model_id", self.encoder.model_id),
            "revision": metadata.get("revision", self.encoder.revision),
            "errors": self.validate_index(),
        }
