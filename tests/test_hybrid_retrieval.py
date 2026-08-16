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


def test_hashing_backend_requires_the_longest_subject_anchor(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("roman", 0, "고대 로마 건축 인물에 관한 일반 기록")],
    )
    service.build_index()

    assert service.search("고대 로마 콜로세움의 건립 인물을 알려 줘") == []


def test_hashing_backend_rejects_unknown_subject_without_title_match(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("admiral", 0, "조선 왕조의 인물과 임금에 관한 기록", title="이순신"),
            chunk("river", 0, "마지막 구간과 지역의 역사 기록", title="영산강"),
        ],
    )
    service.build_index()

    assert service.search("존재하지 않는 해솔왕조의 마지막 임금은 누구야") == []


def test_hashing_backend_filters_each_unrelated_result(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("station", 0, "목포역은 호남선의 철도역이다", title="목포역"),
            chunk("unrelated", 0, "다른 지역의 역사적 사건과 시기를 설명한다"),
        ],
    )
    service.build_index()

    results = service.search("목포역의 역사적 사건과 시기를 알려 줘")

    assert [item.chunk.document_id for item in results] == ["station"]


def test_hashing_backend_prefers_factual_chunk_over_scraped_footer(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("station", 0, "정의 닫기 목포역은 1913년 영업을 시작했다", title="목포역"),
            chunk("station", 1, "수정 의견 작성 비밀번호 파일선택 다운로드가 완료되었습니다", title="목포역"),
        ],
    )
    service.build_index()

    results = service.search("목포역을 알려 줘")

    assert results[0].chunk.chunk_id == "station::chunk-0000"


def test_hashing_backend_rejects_subject_only_in_navigation_breadcrumb(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk(
                "memorial", 0,
                "코스 자세히 보기 동그라미 유달산 > 목포진 > 기념관 "
                "관련 여행코스 위치 및 주변정보 기념관의 전시 내용",
                title="다른 인물 기념관",
            ),
        ],
    )
    service.build_index()

    assert service.search("목포진의 역할과 시기를 알려 줘") == []


def test_hashing_backend_keeps_subject_in_factual_opening_without_title_match(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("local", 0, "정의 닫기 가람도는 항구와 연결된 섬이다", title="섬 기록")],
    )
    service.build_index()

    results = service.search("가람도의 역사적 특징을 알려 줘")

    assert [item.chunk.document_id for item in results] == ["local"]


def test_hashing_backend_keeps_entity_when_question_wording_differs(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("station", 0, "목포역은 1913년 5월 15일 영업을 시작했다", title="목포역")],
    )
    service.build_index()

    results = service.search("목포역 언제 만들어졌어?")

    assert [item.chunk.document_id for item in results] == ["station"]


def test_compound_spacing_variant_preserves_subject(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("consulate", 0, "일본영사관은 1900년에 건립되었다", title="구 목포 일본영사관")],
    )
    service.build_index()

    results = service.search("목포에 있던 일본 영사관 건물, 언제 다 지은 거야?")

    assert [item.chunk.document_id for item in results] == ["consulate"]


def test_japanese_consulate_does_not_rank_oriental_development_company_date_first(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("consulate", 0, "일본영사관은 1900년에 완공되었다.", title="근대역사관1관"),
            chunk("company", 0, "인근 동양척식주식회사 목포지점은 1921년에 건립되었다.", title="근대역사관2관"),
        ],
    )
    service.build_index()

    results = service.search("구 목포 일본영사관은 언제 지어졌어?")

    assert results[0].chunk.document_id == "consulate"


def test_oriental_development_company_direct_question_prefers_museum_two_alias(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [
            chunk("direct", 0, "동양척식주식회사 목포지점 건물의 연혁을 설명한다.", title="근대역사관2관"),
            chunk("mixed", 0, "일본영사관과 동양척식주식회사 목포지점이 함께 남아 있다.", title="근대 역사 공간"),
        ],
    )
    service.build_index()

    results = service.search("동양척식주식회사 목포지점은 뭐야?")

    assert results[0].chunk.document_id == "direct"


def test_hashing_backend_can_find_named_person_inside_factual_record(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("archive", 0, "행사 참석자는 이범석, 안호상 등이었다.", title="행사 기록")],
    )
    service.build_index()

    results = service.search("이범석은 누구야?")

    assert [item.chunk.document_id for item in results] == ["archive"]


def test_hashing_backend_rejects_incidental_subject_late_in_other_article(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk(
            "consulate", 0,
            "이 건물은 외국 영사관으로 사용되었다. 여러 시설의 변천을 설명한 뒤 "
            "과거에 폐지된 가람진을 잠시 빌렸다는 기록이 나온다.",
            title="근대 영사관",
        )],
    )
    service.build_index()

    assert service.search("가람진의 역할과 시기를 알려 줘") == []


def test_hashing_backend_rejects_followup_without_validated_subject(tmp_path) -> None:
    service = make_hashing_service(
        tmp_path,
        [chunk("place", 0, "실제 장소는 전라남도의 섬이다", title="다른 장소")],
    )
    service.build_index()

    assert service.search("그럼 실제 장소는 어디야") == []


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
