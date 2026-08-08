from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from history_chatbot.chat.citation_builder import build_citations
from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.service import ChatApplicationService
from history_chatbot.chat.session import SessionStore
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.provisional.service import ProvisionalDataService
from history_chatbot.retrieval.base import RankedChunk, RetrievalChunk
from history_chatbot.retrieval.dense import HashingDenseEncoder
from history_chatbot.retrieval.qdrant_store import LocalJsonVectorStore
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig
from history_chatbot.runtime import (
    ProvisionalDataDetectedError,
    ProvisionalIndexDetectedError,
    RuntimeMode,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "source_audit" / "mokpo_public_candidates.jsonl"


def service(tmp_path: Path, **overrides) -> ProvisionalDataService:
    values = {
        "audit_path": AUDIT,
        "root": tmp_path / "provisional",
        "index_root": tmp_path / "indexes" / "hackathon",
        "session_path": tmp_path / "sessions.json",
    }
    values.update(overrides)
    return ProvisionalDataService(**values)


def fake_html(url: str) -> bytes:
    identifier = url.rsplit("=", 1)[-1]
    body = (
        f"<html><header>공통 메뉴</header><article><h1>테스트 상세 {identifier}</h1>"
        "<p>테스트용 가상 본문입니다. 실제 역사 사실 검증에 사용하지 않습니다. "
        "해커톤 임시 자료의 청크 처리와 출처 추적만 검증하기 위한 문장입니다.</p>"
        "<p>동일한 공식 식별자와 원본 URL을 유지하는지 확인합니다. "
        "이 문단 역시 네트워크 없는 테스트 전용입니다.</p></article>"
        "<footer>공통 푸터</footer></html>"
    )
    return body.encode("utf-8")


def prepare_and_collect(tmp_path: Path) -> ProvisionalDataService:
    provisional = service(tmp_path)
    provisional.prepare_manifest()
    result = provisional.collect(fake_html, delay_seconds=0)
    assert result["collected"] == 48
    return provisional


def prepare_seven_existing(tmp_path: Path) -> ProvisionalDataService:
    provisional = service(tmp_path)
    records = provisional.prepare_manifest()
    provisional.raw_dir.mkdir(parents=True)
    for record in records[:7]:
        payload = fake_html(record["source_url"])
        text = provisional._extract_text(payload)
        (provisional.raw_dir / f"{record['source_id']}.html").write_bytes(payload)
        record["collection_status"] = "collected"
        record["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    provisional._atomic_jsonl(provisional.manifest_path, records)
    return provisional


def test_exactly_48_are_selected_and_three_kogl4_are_excluded(tmp_path) -> None:
    report = service(tmp_path).dry_run()
    assert report.selected == 48
    assert report.excluded == 3
    assert set(report.excluded_source_ids) == {
        "mokpo-1c01bfd2fec3f61b",
        "mokpo-d261027d2770b77a",
        "mokpo-2d9f0974efa7795f",
    }


def test_manifest_keeps_conservative_rights_and_removability(tmp_path) -> None:
    records = service(tmp_path).prepare_manifest()
    assert len(records) == 48
    assert all(item["usage_status"] == "provisional_hackathon" for item in records)
    assert all(item["allowed_for_rag"] is False for item in records)
    assert all(item["allowed_for_training"] is False for item in records)
    assert all(item["public_release_allowed"] is False for item in records)
    assert all(item["removable"] is True for item in records)


def test_fake_collection_excludes_common_page_chrome_and_images(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    chunks = [
        json.loads(line)
        for line in provisional.chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    assert chunks
    assert all("공통 메뉴" not in item["text"] for item in chunks)
    assert all("공통 푸터" not in item["text"] for item in chunks)
    assert all(item["allowed_for_rag"] is False for item in chunks)
    assert all(item["allowed_for_training"] is False for item in chunks)


def test_local_reprocess_uses_only_stored_raw_and_preserves_manifest(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    manifest_before = provisional.manifest_path.read_bytes()
    raw_before = {
        path.name: path.read_bytes() for path in provisional.raw_dir.glob("*.html")
    }

    report = provisional.reprocess_local()

    assert report["documents"] == 48
    assert report["chunks"] > 0
    assert report["network_requests"] == 0
    assert provisional.manifest_path.read_bytes() == manifest_before
    assert {
        path.name: path.read_bytes() for path in provisional.raw_dir.glob("*.html")
    } == raw_before


def test_extraction_prefers_semantic_main_over_div_based_page_chrome(tmp_path) -> None:
    provisional = service(tmp_path)
    payload = (
        "<html><body>"
        "<div class='global-links'>통합검색 인기검색어 반복 메뉴</div>"
        "<main id='content'><h1>구 동양척식주식회사 목포지점</h1>"
        "<p>검증 대상 역사 설명 본문입니다.</p></main>"
        "<div class='search-layer'>검색 필터 breadcrumb 반복 링크</div>"
        "</body></html>"
    ).encode("utf-8")

    extracted = provisional._extract_text(payload)

    assert "구 동양척식주식회사 목포지점" in extracted
    assert "검증 대상 역사 설명 본문" in extracted
    assert "통합검색 인기검색어" not in extracted
    assert "검색 필터 breadcrumb" not in extracted


def test_existing_successes_are_reused_without_network(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    calls: list[str] = []
    result = provisional.collect(
        lambda url: calls.append(url) or fake_html(url), delay_seconds=0
    )
    assert calls == []
    assert result["reused"] == 48
    assert result["network_requests"] == 0
    records = provisional.load_manifest()
    assert all(item["collection_status"] == "reused" for item in records)
    assert all(item["network_requested"] is False for item in records)
    assert all(item["reused"] is True for item in records)


def test_remapped_i815_success_is_reused_without_network(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    records = provisional.load_manifest()
    record = next(
        item
        for item in records
        if str(item["official_record_id"]).startswith("i815-person-")
    )
    record_id = str(record["official_record_id"]).removeprefix("i815-person-")
    current_url = (
        f"https://search.i815.or.kr/dictionary/detail/print.do?id={record_id}"
    )
    record["source_url"] = current_url
    record["canonical_url"] = current_url
    provisional._atomic_jsonl(provisional.manifest_path, records)

    report = provisional.collect(source_id=record["source_id"], dry_run=True)

    assert report["total_selected"] == 1
    assert report["reused_existing"] == 1
    assert report["pending_network"] == 0


def test_collection_plan_matches_current_seven_and_forty_one(tmp_path) -> None:
    provisional = prepare_seven_existing(tmp_path)
    before = provisional.manifest_path.read_bytes()
    report = provisional.collect(dry_run=True)
    assert report["total_selected"] == 48
    assert report["reused_existing"] == 7
    assert report["pending_network"] == 41
    assert report["estimated_max_gets"] == 41
    assert report["missing_raw"] == 41
    assert len(report["request_urls"]) == 41
    assert provisional.manifest_path.read_bytes() == before
    assert not provisional.chunks_path.exists()
    assert not provisional.index_root.exists()


def test_hash_mismatch_force_and_source_id_are_scoped(tmp_path) -> None:
    provisional = prepare_seven_existing(tmp_path)
    records = provisional.load_manifest()
    damaged = records[0]
    (provisional.raw_dir / f"{damaged['source_id']}.html").write_bytes(b"damaged")
    mismatch = provisional.collect(dry_run=True)
    assert mismatch["reused_existing"] == 6
    assert mismatch["hash_mismatch"] == 1
    assert mismatch["pending_network"] == 42

    forced = provisional.collect(force=True, dry_run=True)
    assert forced["forced"] == 48
    assert forced["estimated_max_gets"] == 48

    scoped = provisional.collect(source_id=records[1]["source_id"], dry_run=True)
    assert scoped["total_selected"] == 1
    assert scoped["reused_existing"] == 1
    assert scoped["estimated_max_gets"] == 0


def test_chunk_failure_does_not_replace_existing_raw(tmp_path, monkeypatch) -> None:
    provisional = prepare_seven_existing(tmp_path)
    source_id = provisional.load_manifest()[0]["source_id"]
    raw_path = provisional.raw_dir / f"{source_id}.html"
    before = raw_path.read_bytes()
    monkeypatch.setattr(
        provisional,
        "_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("chunk failure")),
    )
    result = provisional.collect(
        lambda _: fake_html("changed=1"),
        force=True,
        source_id=source_id,
        delay_seconds=0,
    )
    assert result["failed"] == 1
    assert raw_path.read_bytes() == before


def test_production_blocks_provisional_and_hackathon_allows_it(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    with pytest.raises(ProvisionalDataDetectedError):
        HybridRetrievalService(
            RetrievalConfig(
                runtime_mode="production",
                provisional_chunks_path=provisional.chunks_path,
                local_storage_path=tmp_path / "indexes" / "production",
            )
        )
    retrieval = HybridRetrievalService(
        RetrievalConfig(
            runtime_mode="hackathon",
            provisional_chunks_path=provisional.chunks_path,
            local_storage_path=tmp_path / "indexes" / "hackathon",
        )
    )
    report = retrieval.build_index()
    assert report.chunks > 0
    assert retrieval.store.metadata()["mode"] == "hackathon"
    assert retrieval.store.metadata()["rights_scope"] == "unconfirmed_noncommercial_demo"


def test_source_institution_and_all_removal_rebuild_index(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    provisional.rebuild_index()
    records = provisional.load_manifest()
    source_id = records[0]["source_id"]
    assert provisional.remove(source_id=source_id) == (source_id,)
    assert source_id not in {
        item.payload["source_id"]
        for item in HybridRetrievalService(
            RetrievalConfig(
                runtime_mode="hackathon",
                provisional_chunks_path=provisional.chunks_path,
                local_storage_path=provisional.index_root,
            )
        ).store.chunks()
    }
    institution = records[1]["institution"]
    removed = provisional.remove(institution=institution)
    assert removed
    remaining = [item for item in provisional.load_manifest() if item.get("active")]
    assert all(item["institution"] != institution for item in remaining)
    provisional.remove(purge_all=True)
    assert not any(item.get("active") for item in provisional.load_manifest())


def test_expiry_disables_every_active_source(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    removed = provisional.expire("2026-09-01")
    assert len(removed) == 48


def test_failed_removal_restores_manifest_and_chunks(tmp_path, monkeypatch) -> None:
    provisional = prepare_and_collect(tmp_path)
    provisional.rebuild_index()
    manifest_before = provisional.manifest_path.read_bytes()
    chunks_before = provisional.chunks_path.read_bytes()
    index_path = provisional.index_root / "hashing-v1--builtin.json"
    index_before = index_path.read_bytes()
    metadata_before = json.loads(index_before)["metadata"]
    monkeypatch.setattr(
        provisional,
        "rebuild_index",
        lambda: (_ for _ in ()).throw(RuntimeError("test failure")),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        provisional.remove(source_id=provisional.load_manifest()[0]["source_id"])
    assert provisional.manifest_path.read_bytes() == manifest_before
    assert provisional.chunks_path.read_bytes() == chunks_before
    assert index_path.read_bytes() == index_before
    metadata_after = json.loads(index_path.read_bytes())["metadata"]
    assert metadata_after["index_version"] == metadata_before["index_version"]
    assert metadata_after["source_snapshot"] == metadata_before["source_snapshot"]


def test_post_rebuild_removal_failure_restores_entire_generation(
    tmp_path, monkeypatch
) -> None:
    provisional = prepare_and_collect(tmp_path)
    provisional.rebuild_index()
    index_path = provisional.index_root / "hashing-v1--builtin.json"
    manifest_before = provisional.manifest_path.read_bytes()
    chunks_before = provisional.chunks_path.read_bytes()
    index_before = index_path.read_bytes()
    monkeypatch.setattr(
        provisional,
        "_append_removal_log",
        lambda *args: (_ for _ in ()).throw(RuntimeError("log failure")),
    )
    with pytest.raises(RuntimeError, match="log failure"):
        provisional.remove(source_id=provisional.load_manifest()[0]["source_id"])
    assert provisional.manifest_path.read_bytes() == manifest_before
    assert provisional.chunks_path.read_bytes() == chunks_before
    assert index_path.read_bytes() == index_before
    assert not provisional.removal_log.exists()


@pytest.mark.parametrize("failure_point", ["dense", "metadata", "bm25", "swap"])
def test_failed_rebuild_keeps_previous_searchable_snapshot(
    tmp_path, monkeypatch, failure_point
) -> None:
    provisional = prepare_and_collect(tmp_path)
    provisional.rebuild_index()
    index_path = provisional.index_root / "hashing-v1--builtin.json"
    before = index_path.read_bytes()
    metadata = json.loads(before)["metadata"]
    old_service = HybridRetrievalService(
        RetrievalConfig(
            runtime_mode="hackathon",
            provisional_chunks_path=provisional.chunks_path,
            local_storage_path=provisional.index_root,
            minimum_score=0,
            minimum_dense_score=-1,
        )
    )
    query = old_service.store.chunks()[0].text
    old_ids = [item.chunk.chunk_id for item in old_service.search(query)]

    if failure_point == "dense":
        monkeypatch.setattr(
            HashingDenseEncoder,
            "encode",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dense failure")),
        )
    elif failure_point == "metadata":
        monkeypatch.setattr(
            LocalJsonVectorStore,
            "replace",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("metadata failure")),
        )
    elif failure_point == "bm25":
        monkeypatch.setattr(
            "history_chatbot.provisional.service.BM25Searcher",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bm25 failure")),
        )
    else:
        monkeypatch.setattr(
            provisional,
            "_before_index_swap",
            lambda *args: (_ for _ in ()).throw(RuntimeError("swap failure")),
        )

    with pytest.raises(RuntimeError, match="failure"):
        provisional.rebuild_index()
    assert index_path.read_bytes() == before
    monkeypatch.undo()
    current = HybridRetrievalService(
        RetrievalConfig(
            runtime_mode="hackathon",
            provisional_chunks_path=provisional.chunks_path,
            local_storage_path=provisional.index_root,
            minimum_score=0,
            minimum_dense_score=-1,
        )
    )
    assert current.store.metadata()["index_version"] == metadata["index_version"]
    assert current.store.metadata()["source_snapshot"] == metadata["source_snapshot"]
    assert [item.chunk.chunk_id for item in current.search(query)] == old_ids
    assert not list(provisional.index_root.parent.glob(".hackathon-build-*"))
    assert not (tmp_path / "indexes" / "production").exists()


def test_provisional_citation_and_response_notice_are_present(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    retrieval = HybridRetrievalService(
        RetrievalConfig(
            runtime_mode="hackathon",
            provisional_chunks_path=provisional.chunks_path,
            local_storage_path=provisional.index_root,
            minimum_dense_score=-1,
            minimum_score=0,
        )
    )
    retrieval.build_index()
    chunk = retrieval.store.chunks()[0]
    citation = build_citations([RankedChunk(chunk, 0.9, ("dense",))])[0]
    assert citation.provisional_notice == "해커톤 시연용 공식 참고자료"
    assert len(citation.excerpt) <= 160
    orchestrator = ConversationalRagOrchestrator(
        retrieval,
        MockLLM("확인 가능한 자료가 부족합니다."),
        SessionStore(RuntimeMode.HACKATHON),
        mode=RuntimeMode.HACKATHON,
    )
    response = orchestrator.ask("테스트 상세 자료를 알려줘")
    assert response.provisional_sources_used >= 1
    assert response.usage_scope == "noncommercial_hackathon_demo"
    assert "재사용 범위는 검토 중" in response.rights_notice
    assert response.source_ids
    assert len(orchestrator._apply_hackathon_policy("가" * 2000, [RankedChunk(chunk, 0.9, ("dense",))])) <= 1201
    assert ChatApplicationService(orchestrator).readiness()["status"] == "hackathon_data_partial"


def test_raw_processed_and_runtime_index_are_git_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/provisional_hackathon/raw/" in ignore
    assert "data/provisional_hackathon/processed/" in ignore
    assert ".runtime/indexes/hackathon/" in ignore


def test_hackathon_and_production_storage_are_physically_separate(tmp_path) -> None:
    hackathon = RetrievalConfig(
        runtime_mode="hackathon",
        provisional_chunks_path=tmp_path / "chunks.jsonl",
        local_storage_path=tmp_path / "indexes" / "hackathon",
    )
    production = RetrievalConfig(
        runtime_mode="production",
        local_storage_path=tmp_path / "indexes" / "production",
    )
    assert hackathon.local_storage_path != production.local_storage_path


def test_production_rejects_existing_hackathon_index(tmp_path) -> None:
    provisional = prepare_and_collect(tmp_path)
    provisional.rebuild_index()
    with pytest.raises(ProvisionalIndexDetectedError):
        HybridRetrievalService(
            RetrievalConfig(
                runtime_mode="production",
                local_storage_path=provisional.index_root,
                index_ready_path=tmp_path / "production-ready",
            )
        )
