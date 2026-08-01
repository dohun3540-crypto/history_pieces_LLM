"""index_ready 전용 하이브리드 검색 인덱스 빌드와 질의 서비스."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.retrieval.base import DenseEncoder, RankedChunk, RetrievalChunk
from history_chatbot.retrieval.dense import DenseSearcher, HashingDenseEncoder, SentenceTransformerEncoder
from history_chatbot.retrieval.fusion import reciprocal_rank_fusion
from history_chatbot.retrieval.qdrant_store import LocalJsonVectorStore
from history_chatbot.retrieval.query_normalizer import normalize_query
from history_chatbot.retrieval.reranker import NoOpReranker
from history_chatbot.retrieval.sparse import BM25Searcher
from history_chatbot.retrieval.thresholds import apply_thresholds
from history_chatbot.runtime import (
    FIXTURE_NOTICE,
    ProductionNotReadyError,
    ProvisionalDataDetectedError,
    ProvisionalIndexDetectedError,
    RuntimeMode,
)


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
    runtime_mode: str = "production"
    fixture_chunks_path: Path | None = None
    provisional_chunks_path: Path | None = None

    def validate(self) -> None:
        mode = RuntimeMode.parse(self.runtime_mode)
        if mode == RuntimeMode.PRODUCTION and self.fixture_chunks_path is not None:
            raise ValueError("production 모드에는 fixture_chunks_path를 설정할 수 없습니다.")
        if mode == RuntimeMode.PRODUCTION and self.provisional_chunks_path is not None:
            raise ProvisionalDataDetectedError(
                "production 모드에는 provisional_hackathon 자료를 설정할 수 없습니다."
            )
        if mode != RuntimeMode.HACKATHON and self.provisional_chunks_path is not None:
            raise ValueError("provisional_chunks_path는 hackathon 모드에서만 사용할 수 있습니다.")
        if mode == RuntimeMode.HACKATHON and self.fixture_chunks_path is not None:
            raise ValueError("hackathon 모드에는 개발 fixture를 사용할 수 없습니다.")
        if self.backend != "local_json":
            raise ValueError("현재 다운로드 없는 기본 백엔드는 local_json만 지원합니다.")
        if self.embedding_model != "hashing-v1" and not self.embedding_model.startswith("intfloat/"):
            raise ValueError("허용되지 않은 임베딩 모델입니다.")
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
            elif key in {"fixture_chunks_path", "provisional_chunks_path"}:
                values[key] = Path(value) if value else None
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
    index_version: int
    index_path: Path


class IndexReadyReader:
    def __init__(self, directory: Path, *, runtime_mode: RuntimeMode = RuntimeMode.PRODUCTION) -> None:
        self.directory = directory
        self.runtime_mode = runtime_mode

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
            if record.get("data_classification") == "fictional_fixture":
                raise ValueError("개발용 fixture는 data/index_ready에 포함할 수 없습니다.")
            document_id = str(record.get("document_id", ""))
            if document_id not in active_documents or document_id in tombstones:
                raise ValueError(f"비활성 문서가 index_ready에 포함됨: {document_id}")
            if str(record.get("review_status", "reviewed")) != "reviewed":
                raise ValueError(f"검수 전 문서가 index_ready에 포함됨: {document_id}")
            if record.get("allowed_for_rag") is not True:
                raise ValueError(f"RAG 사용이 허용되지 않은 문서가 index_ready에 포함됨: {document_id}")
            if str(record.get("copyright_status", "")) in {"unknown", "restricted"}:
                raise ValueError(f"저작권 검증을 통과하지 않은 문서가 index_ready에 포함됨: {document_id}")
            if str(record.get("source_reliability", "")) not in {"A", "B"}:
                raise ValueError(f"신뢰도 A/B가 아닌 문서가 index_ready에 포함됨: {document_id}")
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


class FixtureReader:
    """운영 데이터 경로와 분리된 명시적 개발 fixture 로더."""

    def __init__(self, path: Path, runtime_mode: RuntimeMode) -> None:
        self.path = path
        self.runtime_mode = runtime_mode

    def load(self) -> tuple[list[RetrievalChunk], str]:
        if not self.runtime_mode.allows_fixtures:
            raise ValueError("production 모드에서는 fixture를 사용할 수 없습니다.")
        if not self.path.is_file():
            return [], ""
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunks: list[RetrievalChunk] = []
        for record in records:
            if record.get("data_classification") != "fictional_fixture":
                raise ValueError("fixture 파일에 실제 자료 또는 분류되지 않은 자료가 섞여 있습니다.")
            if FIXTURE_NOTICE not in str(record.get("text", "")):
                raise ValueError("fixture 문장에 필수 가상 자료 표시가 없습니다.")
            chunks.append(RetrievalChunk.from_record(record))
        return chunks, stable_json_hash(records)


class ProvisionalReader:
    """권리 미확정 자료를 hackathon 모드에만 노출한다."""

    def __init__(self, path: Path, runtime_mode: RuntimeMode) -> None:
        self.path = path
        self.runtime_mode = runtime_mode

    def load(self) -> tuple[list[RetrievalChunk], str]:
        if self.runtime_mode == RuntimeMode.PRODUCTION:
            raise ProvisionalDataDetectedError(
                "production 로더에서 provisional_hackathon 자료가 탐지되었습니다."
            )
        if self.runtime_mode != RuntimeMode.HACKATHON:
            raise ValueError("임시 해커톤 자료는 hackathon 모드에서만 검색할 수 있습니다.")
        if not self.path.is_file():
            return [], ""
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunks: list[RetrievalChunk] = []
        for record in records:
            if record.get("usage_status") != "provisional_hackathon":
                raise ValueError("해커톤 청크에 허용되지 않은 usage_status가 있습니다.")
            if record.get("rights_status") != "unconfirmed":
                raise ValueError("해커톤 청크의 rights_status가 unconfirmed가 아닙니다.")
            if record.get("allowed_for_rag") is not False:
                raise ValueError("임시 자료는 allowed_for_rag=false를 유지해야 합니다.")
            if record.get("allowed_for_training") is not False:
                raise ValueError("임시 자료는 allowed_for_training=false를 유지해야 합니다.")
            chunks.append(RetrievalChunk.from_record(record))
        return chunks, stable_json_hash(records)


class HybridRetrievalService:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        encoder: DenseEncoder | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.encoder = encoder or (
            HashingDenseEncoder()
            if config.embedding_model == "hashing-v1"
            else SentenceTransformerEncoder(config.embedding_model, revision=config.embedding_revision)
        )
        self.store = LocalJsonVectorStore(
            config.local_storage_path / self.index_filename
        )
        if (
            self.runtime_mode == RuntimeMode.PRODUCTION
            and self.store.metadata().get("mode") == "hackathon"
        ):
            raise ProvisionalIndexDetectedError(
                "production 경로에서 hackathon 전용 인덱스가 탐지되었습니다."
            )
        self.reranker = NoOpReranker()

    @property
    def runtime_mode(self) -> RuntimeMode:
        return RuntimeMode.parse(self.config.runtime_mode)

    def _reader(self):
        if self.config.fixture_chunks_path is not None:
            return FixtureReader(self.config.fixture_chunks_path, self.runtime_mode)
        if self.config.provisional_chunks_path is not None:
            return ProvisionalReader(
                self.config.provisional_chunks_path, self.runtime_mode
            )
        return IndexReadyReader(
            self.config.index_ready_path, runtime_mode=self.runtime_mode
        )

    @property
    def index_filename(self) -> str:
        safe_model = self.config.embedding_model.replace("/", "--")
        safe_revision = self.config.embedding_revision.replace("/", "-")
        return f"{safe_model}--{safe_revision}.json"

    def build_index(self, *, force: bool = False) -> BuildReport:
        chunks, snapshot = self._reader().load()
        if self.runtime_mode == RuntimeMode.PRODUCTION and not chunks:
            raise ProductionNotReadyError(
                "현재 운영 인덱싱 가능한 reviewed + allowed_for_rag 실제 자료가 없습니다."
            )
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
            extra_metadata=self._index_metadata(chunks),
        )
        return BuildReport(
            len(chunks),
            len(encoded),
            reused,
            max(0, len(previous) - reused),
            snapshot,
            self.encoder.model_id,
            self.encoder.revision,
            int(self.store.metadata().get("index_version", 1)),
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
        if metadata.get("dimension") != self.encoder.dimension:
            errors.append("임베딩 차원이 검색 인덱스와 일치하지 않습니다.")
        if metadata.get("normalization") != bool(getattr(self.encoder, "normalize_embeddings", True)):
            errors.append("임베딩 정규화 정책이 검색 인덱스와 일치하지 않습니다.")
        try:
            _, current_snapshot = self._reader().load()
        except (ValueError, ProductionNotReadyError) as error:
            errors.append(str(error))
            return errors
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

    def rollback(self, source_snapshot: str) -> None:
        self.store.rollback(source_snapshot)

    def _index_metadata(self, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        embedding = {
            "dimension": self.encoder.dimension,
            "normalization": bool(getattr(self.encoder, "normalize_embeddings", True)),
            "query_prefix": str(getattr(self.encoder, "query_prefix", "")),
            "passage_prefix": str(getattr(self.encoder, "passage_prefix", "")),
        }
        if self.runtime_mode != RuntimeMode.HACKATHON:
            return {"mode": self.runtime_mode.value, **embedding}
        source_ids = sorted(
            {
                str(chunk.payload.get("source_id", chunk.document_id))
                for chunk in chunks
            }
        )
        return {
            "mode": "hackathon",
            "provisional_document_count": len(source_ids),
            "provisional_chunk_count": len(chunks),
            "rights_scope": "unconfirmed_noncommercial_demo",
            "removable_source_ids": source_ids,
            **embedding,
        }
