from __future__ import annotations

import json
from pathlib import Path

from history_chatbot.chat.service import create_development_real_service
from history_chatbot.indexing.loader import ReviewedChunkLoader
from history_chatbot.ingestion.development import DevelopmentManifestLoader
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.service import (
    HybridRetrievalService,
    IndexReadyReader,
    RetrievalConfig,
)
from history_chatbot.runtime import RuntimeMode


ROOT = Path("data/development_real")
EXPECTED_IDS = {"mokpo_hist_0004", "mokpo_hist_0005", "mokpo_hist_0006"}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_manifest_and_chunks_contain_only_three_approved_documents() -> None:
    report = DevelopmentManifestLoader(
        ROOT / "manifests/sources.jsonl", runtime_mode=RuntimeMode.DEVELOPMENT
    ).load()
    chunks = read_jsonl(ROOT / "index_ready/chunks.jsonl")

    assert {item.document_id for item in report.approved} == EXPECTED_IDS
    assert report.rejected == ()
    assert {str(item["document_id"]) for item in chunks} == EXPECTED_IDS
    assert len(chunks) == 6
    assert all(item["approval_tier"] == "development_approved" for item in chunks)
    assert all(item["production_approved"] is False for item in chunks)
    assert all(item["public_release_allowed"] is False for item in chunks)
    assert all(item["data_classification"] == "real_historical_source" for item in chunks)
    assert all(item["is_fixture"] is False for item in chunks)


def test_processed_files_preserve_two_evidence_chunks_per_document() -> None:
    for document_id in EXPECTED_IDS:
        chunks = read_jsonl(ROOT / f"processed/{document_id}.jsonl")
        assert len(chunks) == 2
        assert all(item["document_id"] == document_id for item in chunks)
        assert all(str(item["evidence_quote"]).strip() for item in chunks)
        assert all(str(item["citation_url"]).startswith("https://biz.mokpo.go.kr/") for item in chunks)


def test_production_reader_rejects_every_development_chunk() -> None:
    chunks = read_jsonl(ROOT / "index_ready/chunks.jsonl")
    assert all(IndexReadyReader._development_lane_errors(item) for item in chunks)
    assert ReviewedChunkLoader.__module__ == "history_chatbot.indexing.loader"


def test_local_hashing_index_is_ready_and_fixture_free() -> None:
    config = RetrievalConfig.load(Path("configs/retrieval.development-real.yaml"))
    retrieval = HybridRetrievalService(config)
    assert retrieval.status()["ready"] is True
    assert retrieval.status()["chunks"] == 6
    assert retrieval.store.metadata()["data_lane"] == "development_real"
    assert retrieval.store.metadata()["production_approved"] is False
    assert all(chunk.payload["is_fixture"] is False for chunk in retrieval.store.chunks())


def test_fixture_free_questions_retrieve_the_expected_primary_document() -> None:
    retrieval = HybridRetrievalService(
        RetrievalConfig.load(Path("configs/retrieval.development-real.yaml"))
    )
    expectations = {
        "구 목포 일본영사관은 원래 어떤 건물이었고 언제 건립되었나요?": "mokpo_hist_0004",
        "목포진은 언제 설치되고 폐지되었나요?": "mokpo_hist_0005",
        "동양척식주식회사 목포지점 건물은 언제 건립되었나요?": "mokpo_hist_0006",
    }
    for question, document_id in expectations.items():
        results = retrieval.search(question)
        assert results
        assert results[0].chunk.document_id == document_id


def test_service_citations_are_real_development_sources_with_warnings() -> None:
    service = create_development_real_service(llm=MockLLM("결정론적 개발 응답"))
    session_id = service.orchestrator.sessions.create().session_id
    response = service.chat(
        {
            "session_id": session_id,
            "user_query": "목포진은 언제 설치되고 폐지되었나요?",
            "conversation_mode": "free_chat",
        }
    )
    assert response["request_state"] == "success"
    assert response["citations"]
    for citation in response["citations"]:
        assert citation["is_fixture"] is False
        assert citation["source_status"] == "development_only"
        assert citation["approval_tier"] == "development_approved"
        assert citation["production_approved"] is False
        assert citation["badge_label"] == "개발 검증용 자료"
        assert "production 공개 승인" in citation["usage_notice"]


def test_missing_subject_returns_insufficient_without_citation() -> None:
    service = create_development_real_service(llm=MockLLM("호출되면 안 되는 응답"))
    session_id = service.orchestrator.sessions.create().session_id
    response = service.chat(
        {
            "session_id": session_id,
            "user_query": "경동성당은 언제 건립되었나요?",
            "conversation_mode": "free_chat",
        }
    )
    assert response["request_state"] == "insufficient_evidence"
    assert response["citations"] == ()
    assert response["retrieved_chunk_ids"] == ()


def test_frontend_renders_optional_development_citation_notices_as_text() -> None:
    script = Path("src/history_chatbot/web/static/app.js").read_text(encoding="utf-8")
    styles = Path("src/history_chatbot/web/static/styles.css").read_text(encoding="utf-8")

    assert 'typeof citation.badge_label === "string"' in script
    assert "badge.textContent = citation.badge_label.trim()" in script
    assert 'typeof citation.usage_notice === "string"' in script
    assert "notice.textContent = citation.usage_notice.trim()" in script
    assert "production_approved" not in script
    assert "innerHTML" not in script
    assert ".citation-badge" in styles
    assert ".citation .citation-notice" in styles
