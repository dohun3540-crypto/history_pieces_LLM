import json
from pathlib import Path

import pytest

from history_chatbot.collectors.status import tour_api_status
from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.models.factory import build_llm_backend
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig
from history_chatbot.runtime import ProductionNotReadyError, RuntimeMode


FIXTURE_CHUNKS = Path(__file__).parent / "fixtures" / "rag" / "fictional_chunks.jsonl"


def config(tmp_path, **overrides):
    values = {
        "runtime_mode": "test",
        "embedding_model": "hashing-v1",
        "embedding_revision": "builtin",
        "fixture_chunks_path": FIXTURE_CHUNKS,
        "local_storage_path": tmp_path / "retrieval",
        "index_ready_path": tmp_path / "index_ready",
        "minimum_score": 0.0,
        "minimum_dense_score": -1.0,
    }
    values.update(overrides)
    return RetrievalConfig(**values)


def write_ready(path: Path, records: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "chunks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    (path / "index_manifest.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": stable_json_hash(records),
                "documents": {item["document_id"]: {} for item in records},
                "tombstones": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_fixture_is_blocked_in_production(tmp_path) -> None:
    with pytest.raises(ValueError, match="production"):
        HybridRetrievalService(config(tmp_path, runtime_mode="production"))


def test_fixture_cannot_be_mixed_into_real_index_ready(tmp_path) -> None:
    record = json.loads(FIXTURE_CHUNKS.read_text(encoding="utf-8").splitlines()[0])
    ready = tmp_path / "index_ready"
    write_ready(ready, [record])
    service = HybridRetrievalService(
        config(tmp_path, fixture_chunks_path=None, index_ready_path=ready)
    )
    with pytest.raises(ValueError, match="fixture"):
        service.build_index()


def test_empty_production_index_reports_not_ready(tmp_path) -> None:
    service = HybridRetrievalService(
        config(
            tmp_path,
            runtime_mode="production",
            fixture_chunks_path=None,
        )
    )
    with pytest.raises(ProductionNotReadyError, match="reviewed"):
        service.build_index()


def test_fixture_based_search(tmp_path) -> None:
    service = HybridRetrievalService(config(tmp_path))
    report = service.build_index()
    results = service.search("붉은 등대 전시관")
    assert report.chunks == 2
    assert results
    assert results[0].chunk.payload["data_classification"] == "fictional_fixture"


def test_incremental_add_modify_delete_and_rollback(tmp_path) -> None:
    ready = tmp_path / "index_ready"
    base = {
        "document_id": "actual-1",
        "chunk_id": "actual-1::chunk-0000",
        "text": "검수된 실제 자료를 대신하는 단위 테스트 입력",
        "title": "단위 테스트 입력",
        "publisher": "테스트 기관",
        "source_url": "https://example.invalid/actual-1",
        "review_status": "reviewed",
        "allowed_for_rag": True,
        "copyright_status": "open_license",
        "source_reliability": "A",
        "reviewed_by": "tester",
        "reviewed_at": "2026-07-30T00:00:00+09:00",
        "content_sha256": "v1",
    }
    write_ready(ready, [base])
    service = HybridRetrievalService(
        config(tmp_path, fixture_chunks_path=None, index_ready_path=ready)
    )
    first = service.build_index()
    first_snapshot = first.source_snapshot

    added = {
        **base,
        "document_id": "actual-2",
        "chunk_id": "actual-2::chunk-0000",
        "source_url": "https://example.invalid/actual-2",
    }
    modified = {**base, "text": "수정 후 재검수된 단위 테스트 입력", "content_sha256": "v2"}
    write_ready(ready, [modified, added])
    second = service.build_index()
    assert second.embedded_chunks == 2
    assert second.index_version == first.index_version + 1

    write_ready(ready, [added])
    third = service.build_index()
    assert third.removed_chunks == 1
    assert {item.document_id for item in service.store.chunks()} == {"actual-2"}

    service.rollback(first_snapshot)
    assert {item.document_id for item in service.store.chunks()} == {"actual-1"}


def test_tour_api_without_key_is_pending_and_network_forbidden() -> None:
    status = tour_api_status({})
    assert status.status == "pending_credentials"
    assert not status.network_allowed


def test_mock_backend_is_only_available_outside_production() -> None:
    assert isinstance(
        build_llm_backend(
            "mock", runtime_mode=RuntimeMode.DEVELOPMENT, fallback_message="fallback"
        ),
        MockLLM,
    )
    with pytest.raises(ValueError, match="production"):
        build_llm_backend(
            "mock", runtime_mode=RuntimeMode.PRODUCTION, fallback_message="fallback"
        )
