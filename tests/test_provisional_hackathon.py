from __future__ import annotations

import json
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
    manifest_before = provisional.manifest_path.read_bytes()
    chunks_before = provisional.chunks_path.read_bytes()
    monkeypatch.setattr(
        provisional,
        "rebuild_index",
        lambda: (_ for _ in ()).throw(RuntimeError("test failure")),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        provisional.remove(source_id=provisional.load_manifest()[0]["source_id"])
    assert provisional.manifest_path.read_bytes() == manifest_before
    assert provisional.chunks_path.read_bytes() == chunks_before


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
