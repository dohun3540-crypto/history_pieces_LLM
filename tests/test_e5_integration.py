import json
import math
import os
from pathlib import Path

import pytest

from history_chatbot.retrieval.dense import DenseSearcher, SentenceTransformerEncoder
from history_chatbot.retrieval.qdrant_store import LocalJsonVectorStore
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig


pytestmark = pytest.mark.integration
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
INDEX = Path(f".runtime/indexes/hackathon/e5/intfloat--multilingual-e5-small--{REVISION}.json")


@pytest.fixture(scope="module")
def runtime():
    if os.environ.get("RUN_E5_INTEGRATION") != "1":
        pytest.skip("RUN_E5_INTEGRATION=1에서만 실제 모델 테스트 실행")
    cache = Path(os.environ.get("HF_HOME", ".runtime/model_cache/huggingface"))
    encoder = SentenceTransformerEncoder(
        "intfloat/multilingual-e5-small", revision=REVISION,
        cache_folder=str(cache / "hub"),
    )
    return encoder, LocalJsonVectorStore(INDEX)


def test_real_model_dimension_revision_prefix_and_normalization(runtime) -> None:
    encoder, _ = runtime
    assert encoder.dimension == 384
    assert encoder.revision == REVISION
    assert encoder.query_prefix == "query: " and encoder.passage_prefix == "passage: "
    vector = encoder.encode(["목포 개항"], is_query=True)[0]
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-5)


def test_real_index_has_239_finite_vectors_and_48_documents(runtime) -> None:
    _, store = runtime
    entries = store.entries()
    assert len(entries) == 239
    assert len({chunk.document_id for chunk, _ in entries}) == 48
    assert all(len(vector) == 384 and all(math.isfinite(x) for x in vector) for _, vector in entries)
    assert all(chunk.payload["usage_status"] == "provisional_hackathon" for chunk, _ in entries)
    assert store.metadata()["data_lane"] == "provisional_hackathon"


def test_real_chinese_query_retrieves_korean_document_in_top_ten(runtime) -> None:
    encoder, store = runtime
    results = DenseSearcher(encoder, store).search(
        "朴爱顺的出生地和独立运动类别是什么？", 10
    )
    assert any(
        item.chunk.document_id == "mokpo-7e74b49138e96a23" for item in results
    )


def test_e5_and_hashing_indexes_are_separate(runtime) -> None:
    _, store = runtime
    hashing = Path(".runtime/indexes/hackathon/hashing-v1--builtin.json")
    assert INDEX.is_file() and hashing.is_file() and INDEX.resolve() != hashing.resolve()
    assert store.metadata()["model_name"] == "intfloat/multilingual-e5-small"


def test_wrong_real_index_dimension_is_rejected_without_hashing_fallback(runtime) -> None:
    encoder, _ = runtime
    service = HybridRetrievalService(
        RetrievalConfig(
            embedding_model="intfloat/multilingual-e5-small", embedding_revision=REVISION,
            local_storage_path=Path(".runtime/indexes/hackathon/e5"),
            provisional_chunks_path=Path("data/provisional_hackathon/processed/chunks.jsonl"),
            runtime_mode="hackathon",
        ),
        encoder=encoder,
    )
    service.store._metadata["dimension"] = 999
    assert any("차원" in error for error in service.validate_index())
    assert service.encoder.model_id == "intfloat/multilingual-e5-small"


def test_evaluation_fixture_is_not_in_corpus() -> None:
    evaluation = json.loads(Path("tests/fixtures/retrieval/multilingual_e5_evaluation.json").read_text(encoding="utf-8"))
    corpus = Path("data/provisional_hackathon/processed/chunks.jsonl").read_text(encoding="utf-8")
    assert all(item["case_id"] not in corpus for item in evaluation["cases"])
