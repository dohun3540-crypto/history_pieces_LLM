"""실제 48문서/133청크에서 hashing, BM25, multilingual-e5를 비교한다."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import huggingface_hub
import sentence_transformers
import torch
import transformers

from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.retrieval.base import RankedChunk
from history_chatbot.retrieval.dense import DenseSearcher, HashingDenseEncoder, SentenceTransformerEncoder
from history_chatbot.retrieval.evaluation import evaluate_backend, load_cases
from history_chatbot.retrieval.fusion import reciprocal_rank_fusion
from history_chatbot.retrieval.qdrant_store import LocalJsonVectorStore
from history_chatbot.retrieval.query_normalizer import normalize_query
from history_chatbot.retrieval.service import HybridRetrievalService, ProvisionalReader, RetrievalConfig
from history_chatbot.retrieval.sparse import BM25Searcher
from history_chatbot.retrieval.thresholds import apply_thresholds
from history_chatbot.runtime import RuntimeMode


ROOT = Path(__file__).resolve().parents[1]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CHUNKS = ROOT / "data/provisional_hackathon/processed/chunks.jsonl"
HASH_INDEX = ROOT / ".runtime/indexes/hackathon/hashing-v1--builtin.json"
E5_ROOT = ROOT / ".runtime/indexes/hackathon/e5"
CASES = ROOT / "tests/fixtures/retrieval/multilingual_e5_evaluation.json"
REPORT = ROOT / "reports/embedding_benchmark.json"


class TimedEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.model_id = encoder.model_id
        self.revision = encoder.revision
        self.dimension = encoder.dimension
        self.query_prefix = encoder.query_prefix
        self.passage_prefix = encoder.passage_prefix
        self.normalize_embeddings = encoder.normalize_embeddings
        self.passage_seconds = 0.0

    def encode(self, texts, *, is_query):
        started = time.perf_counter()
        result = self.encoder.encode(texts, is_query=is_query)
        if not is_query:
            self.passage_seconds += time.perf_counter() - started
        return result


def _filtered(query: str, results: list[RankedChunk], *, dense_threshold: float, final_top_k: int = 5, max_per_document: int = 2) -> list[RankedChunk]:
    return apply_thresholds(
        normalize_query(query), results, minimum_score=.20,
        minimum_dense_score=dense_threshold, max_chunks_per_document=max_per_document,
        final_top_k=final_top_k,
    )


def _integrity(store: LocalJsonVectorStore, expected_snapshot: str) -> dict[str, object]:
    entries = store.entries()
    ids = [chunk.chunk_id for chunk, _ in entries]
    source_ids = [str(chunk.payload.get("source_id", "")) for chunk, _ in entries]
    vectors = [vector for _, vector in entries]
    return {
        "vector_count": len(vectors),
        "dimension": len(vectors[0]) if vectors else 0,
        "nan_or_inf": sum(not all(math.isfinite(value) for value in vector) for vector in vectors),
        "empty_vectors": sum(not vector or math.sqrt(sum(value * value for value in vector)) == 0 for vector in vectors),
        "duplicate_chunk_ids": len(ids) - len(set(ids)),
        "missing_source_ids": sum(not value for value in source_ids),
        "document_count": len({chunk.document_id for chunk, _ in entries}),
        "data_lanes": sorted({str(chunk.payload.get("usage_status", "")) for chunk, _ in entries}),
        "source_snapshot_matches": store.metadata().get("source_snapshot") == expected_snapshot,
    }


def main() -> None:
    cache = Path(os.environ.get("HF_HOME", ROOT / ".runtime/model_cache/huggingface"))
    started = time.perf_counter()
    base_encoder = SentenceTransformerEncoder(MODEL, revision=REVISION, cache_folder=str(cache / "hub"))
    load_seconds = time.perf_counter() - started
    encoder = TimedEncoder(base_encoder)
    config = RetrievalConfig(
        embedding_model=MODEL, embedding_revision=REVISION,
        local_storage_path=E5_ROOT,
        provisional_chunks_path=CHUNKS,
        runtime_mode="hackathon",
        minimum_score=.20, minimum_dense_score=.72,
        dense_top_k=12, sparse_top_k=12, final_top_k=5, max_chunks_per_document=2,
    )
    service = HybridRetrievalService(config, encoder=encoder)
    build_started = time.perf_counter()
    build = service.build_index(force=True)
    build_seconds = time.perf_counter() - build_started
    save_seconds = max(0.0, build_seconds - encoder.passage_seconds)
    chunks, snapshot = ProvisionalReader(CHUNKS, RuntimeMode.HACKATHON).load()
    e5_store = service.store
    hash_store = LocalJsonVectorStore(HASH_INDEX)
    if len(hash_store.chunks()) != 133:
        raise RuntimeError("기존 hashing-v1 인덱스가 133개 청크가 아닙니다.")
    bm25 = BM25Searcher(chunks)
    hash_encoder = HashingDenseEncoder()
    cases = load_cases(CASES)

    def sparse(query: str):
        return _filtered(query, bm25.search(query, 12), dense_threshold=.72)

    def hash_dense(query: str):
        raw = DenseSearcher(hash_encoder, hash_store).search(query, 12)
        return _filtered(query, raw, dense_threshold=.72)

    def e5_dense(query: str):
        raw = DenseSearcher(encoder, e5_store).search(query, 12)
        return _filtered(query, raw, dense_threshold=.72)

    def hybrid(query: str, dense_search, *, rank_constant=10, threshold=.72, dense_top_k=12, sparse_top_k=12, final_top_k=5, max_per_document=2):
        dense = dense_search(query)[:dense_top_k]
        sparse_results = bm25.search(query, sparse_top_k)
        fused = reciprocal_rank_fusion(dense, sparse_results, rank_constant=rank_constant)
        return _filtered(query, fused, dense_threshold=threshold, final_top_k=final_top_k, max_per_document=max_per_document)

    hash_raw = lambda q: DenseSearcher(hash_encoder, hash_store).search(q, 20)
    e5_raw = lambda q: DenseSearcher(encoder, e5_store).search(q, 20)
    evaluations = [
        evaluate_backend("bm25_only", cases, sparse),
        evaluate_backend("hashing_dense_only", cases, hash_dense),
        evaluate_backend("e5_dense_only", cases, e5_dense),
        evaluate_backend("bm25_hashing_hybrid", cases, lambda q: hybrid(q, hash_raw)),
        evaluate_backend("bm25_e5_hybrid", cases, lambda q: hybrid(q, e5_raw)),
    ]
    tuning = []
    for rank_constant, threshold, dense_top_k, sparse_top_k, final_top_k, max_per_document in (
        (10,.72,12,12,5,2), (30,.72,12,12,5,2), (60,.72,12,12,5,2),
        (10,.76,12,12,5,2), (10,.80,12,12,5,2),
        (10,.82,12,12,5,2), (10,.84,12,12,5,2), (10,.86,12,12,5,2),
        (10,.72,8,12,5,2), (10,.72,16,12,5,2),
        (10,.72,12,8,5,2), (10,.72,12,16,5,2),
        (10,.72,12,12,3,2), (10,.72,12,12,5,1),
    ):
        name = f"rrf{rank_constant}_th{threshold}_d{dense_top_k}_s{sparse_top_k}_f{final_top_k}_m{max_per_document}"
        result = evaluate_backend(name, cases, lambda q, rc=rank_constant, th=threshold, dk=dense_top_k, sk=sparse_top_k, fk=final_top_k, mp=max_per_document: hybrid(q, e5_raw, rank_constant=rc, threshold=th, dense_top_k=dk, sparse_top_k=sk, final_top_k=fk, max_per_document=mp))
        tuning.append({"name": name, "metrics": result["metrics"]})

    cache_files = [path for path in cache.rglob("*") if path.is_file()]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
            "device": "cpu", "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count(),
        },
        "model": {
            "name": MODEL, "revision": REVISION, "dimension": encoder.dimension,
            "normalized": True, "query_prefix": encoder.query_prefix,
            "passage_prefix": encoder.passage_prefix, "cache_path": str(cache),
            "cache_bytes": sum(path.stat().st_size for path in cache_files),
            "cache_file_count": len(cache_files),
        },
        "indexing": {
            "model_load_seconds": round(load_seconds, 4),
            "total_build_seconds": round(build_seconds, 4),
            "embedding_seconds": round(encoder.passage_seconds, 4),
            "average_embedding_ms_per_chunk": round(encoder.passage_seconds * 1000 / len(chunks), 4),
            "index_save_seconds_estimate": round(save_seconds, 4),
            "index_path": str(build.index_path), "index_bytes": build.index_path.stat().st_size,
            "source_snapshot": snapshot, "corpus_sha256": hashlib.sha256(CHUNKS.read_bytes()).hexdigest(),
            "integrity": _integrity(e5_store, snapshot), "metadata": e5_store.metadata(),
            "hashing_index_path": str(HASH_INDEX), "hashing_index_bytes": HASH_INDEX.stat().st_size,
        },
        "evaluation_fixture": str(CASES.relative_to(ROOT)),
        "evaluation_case_count": len(cases),
        "evaluations": evaluations,
        "tuning_candidates": tuning,
        "limitations": [
            "30개 소규모 수동 평가셋이므로 통계적으로 일반화할 수 없다.",
            "중국어 질의는 평가용 번역이며 기록새 중국어 대사 데이터가 아니다.",
            "provisional_hackathon corpus의 일부 웹 페이지 청크에는 탐색 UI 잡음이 많다.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "indexing": report["indexing"],
        "metrics": {item["backend"]: item["metrics"] for item in evaluations},
        "tuning": tuning,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
