import json
from pathlib import Path

import pytest

from history_chatbot.indexing.snapshot import stable_json_hash
from history_chatbot.retrieval.base import DenseEncoder
from history_chatbot.retrieval.dense import HashingDenseEncoder
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig


class FixtureEncoder(DenseEncoder):
    model_id = "fixture-semantic"
    revision = "test-1"
    dimension = 3

    def __init__(self):
        self.encoded_passages = 0

    def encode(self, texts, *, is_query):
        if not is_query:
            self.encoded_passages += len(texts)
        vectors = []
        for text in texts:
            if "항구를 열" in text or "개항" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "세관" in text or "해관" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def chunk(document_id: str, index: int, text: str, **overrides):
    record = {
        "document_id": document_id,
        "chunk_id": f"{document_id}::chunk-{index:04d}",
        "chunk_index": index,
        "text": text,
        "title": f"{document_id} 테스트용 가상 자료",
        "publisher": "테스트 기관",
        "source_url": f"https://example.invalid/{document_id}",
        "review_status": "reviewed",
        "allowed_for_rag": True,
        "copyright_status": "open_license",
        "source_reliability": "A",
        "reviewed_by": "검수자",
        "reviewed_at": "2026-07-30T00:00:00+09:00",
        "content_sha256": stable_json_hash(" ".join(text.split())),
        "keywords": [],
    }
    record.update(overrides)
    return record


def write_index_ready(path: Path, records: list[dict], tombstones=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "chunks.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    documents = {
        record["document_id"]: {"chunk_count": 1} for record in records
    }
    manifest = {
        "version": 1,
        "snapshot_sha256": stable_json_hash(records),
        "documents": documents,
        "tombstones": tombstones or [],
    }
    (path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def make_service(tmp_path: Path, records: list[dict]) -> HybridRetrievalService:
    ready = tmp_path / "index_ready"
    write_index_ready(ready, records)
    config = RetrievalConfig(
        embedding_model="hashing-v1",
        embedding_revision="builtin",
        minimum_score=0.20,
        minimum_dense_score=0.70,
        local_storage_path=tmp_path / "retrieval",
        index_ready_path=ready,
        runtime_mode="test",
    )
    return HybridRetrievalService(config, encoder=FixtureEncoder())


def make_hashing_service(tmp_path: Path, records: list[dict]) -> HybridRetrievalService:
    ready = tmp_path / "index_ready"
    write_index_ready(ready, records)
    config = RetrievalConfig(
        embedding_model="hashing-v1",
        embedding_revision="builtin",
        minimum_score=0.20,
        minimum_dense_score=0.72,
        local_storage_path=tmp_path / "retrieval",
        index_ready_path=ready,
        runtime_mode="test",
    )
    return HybridRetrievalService(config, encoder=HashingDenseEncoder())


def test_dense_and_sparse_results_are_fused(tmp_path) -> None:
    service = make_service(
        tmp_path,
        [
            chunk("open-port", 0, "목포 개항에 관한 테스트용 가상 설명"),
            chunk("customs", 0, "목포 해관에 관한 테스트용 가상 설명"),
        ],
    )
    service.build_index()

    result = service.search("목포 개항")[0]

    assert result.chunk.document_id == "open-port"
    assert set(result.methods) == {"dense", "sparse"}
    assert result.dense_score > 0
    assert result.sparse_score > 0


def test_semantic_rephrasing_can_be_found_by_dense_search(tmp_path) -> None:
    service = make_service(
        tmp_path, [chunk("open-port", 0, "목포 개항에 관한 테스트용 가상 설명")]
    )
    service.build_index()

    results = service.search("목포에서 항구를 열었던 과정")

    assert [item.chunk.document_id for item in results] == ["open-port"]
    assert "dense" in results[0].methods


def test_unrelated_astronaut_question_does_not_match_common_mokpo_word(tmp_path) -> None:
    service = make_service(
        tmp_path, [chunk("open-port", 0, "목포 개항에 관한 테스트용 가상 설명")]
    )
    service.build_index()

    assert service.search("목포 출신 최초의 우주비행사는 누구인가요?") == []


def test_hashing_backend_rejects_partial_question_boilerplate_overlap(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("method", 0, "독립운동을 전개하는 방법을 논의하였다")],
    )
    service.build_index()

    assert service.search("양자컴퓨터의 큐비트 오류 정정 방법을 설명해 주세요.") == []


def test_hashing_backend_keeps_multi_chunk_topic_coverage(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("rail", 0, "목포역과 호남선 철도 발전에 관한 기록"),
            chunk("port", 0, "목포 항만 발전에 관한 기록"),
        ],
    )
    service.build_index()

    results = service.search(
        "목포역은 근대 목포의 철도와 항만 발전에 어떤 역할을 했나요?"
    )

    assert {item.chunk.document_id for item in results} == {"rail", "port"}


def test_korean_particle_and_spacing_variant_is_retrieved(tmp_path) -> None:
    service = make_service(
        tmp_path, [chunk("open-port", 0, "목포의 개항에 관한 테스트용 가상 설명")]
    )
    service.build_index()

    results = service.search("목포는언제 개항했나요?")

    assert results[0].chunk.document_id == "open-port"


@pytest.mark.parametrize("review_status", ["draft", "rejected"])
def test_draft_or_rejected_chunk_is_never_indexed(tmp_path, review_status) -> None:
    service = make_service(
        tmp_path,
        [chunk("unsafe", 0, "검수 전 자료", review_status=review_status)],
    )

    with pytest.raises(ValueError, match="검수 전"):
        service.build_index()


def test_duplicate_chunk_text_is_removed(tmp_path) -> None:
    service = make_service(
        tmp_path,
        [
            chunk("first", 0, "중복 테스트용 본문"),
            chunk("second", 0, "중복   테스트용 본문"),
        ],
    )

    report = service.build_index()

    assert report.chunks == 1


def test_per_document_limit_prevents_chunk_monopoly(tmp_path) -> None:
    records = [
        chunk("open-port", index, f"목포 개항 테스트 설명 {index}")
        for index in range(4)
    ]
    records.append(chunk("second", 0, "다른 개항 테스트 설명"))
    service = make_service(tmp_path, records)
    service.build_index()

    results = service.search("목포 개항")

    assert sum(item.chunk.document_id == "open-port" for item in results) <= 2


def test_model_version_mismatch_blocks_search(tmp_path) -> None:
    service = make_service(
        tmp_path, [chunk("open-port", 0, "목포 개항 테스트 설명")]
    )
    service.build_index()
    changed = RetrievalConfig(
        embedding_model="hashing-v1",
        embedding_revision="builtin",
        local_storage_path=service.config.local_storage_path,
        index_ready_path=service.config.index_ready_path,
        runtime_mode=service.config.runtime_mode,
    )
    mismatched_encoder = FixtureEncoder()
    mismatched_encoder.revision = "test-2"
    mismatched = HybridRetrievalService(changed, encoder=mismatched_encoder)

    assert any("revision" in error for error in mismatched.validate_index())
    assert mismatched.search("목포 개항") == []


def test_changed_snapshot_and_deleted_document_require_reindex(tmp_path) -> None:
    first = chunk("open-port", 0, "목포 개항 테스트 설명")
    service = make_service(tmp_path, [first])
    service.build_index()
    write_index_ready(
        service.config.index_ready_path,
        [chunk("customs", 0, "목포 해관 테스트 설명")],
        tombstones=[
            {
                "document_id": "open-port",
                "removed_at": "2026-07-30T00:00:00+09:00",
                "reason": "removed",
            }
        ],
    )

    assert any("재색인" in error for error in service.validate_index())
    assert service.search("목포 개항") == []
    service.build_index()
    assert service.search("목포 개항") == []
    assert service.search("목포 해관")[0].chunk.document_id == "customs"


def test_empty_index_ready_builds_empty_offline_index(tmp_path) -> None:
    service = make_service(tmp_path, [])

    report = service.build_index()

    assert report.chunks == 0
    assert service.search("목포 개항") == []


def test_incremental_build_reuses_unchanged_vectors(tmp_path) -> None:
    service = make_service(
        tmp_path, [chunk("open-port", 0, "목포 개항 테스트 설명")]
    )

    first = service.build_index()
    second = service.build_index()

    assert first.embedded_chunks == 1
    assert second.embedded_chunks == 0
    assert second.reused_chunks == 1
