from __future__ import annotations

import hashlib
import json
from pathlib import Path

from history_chatbot.chat.context_resolver import ConversationContextResolver
from history_chatbot.chat.service import create_hackathon_orchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.history_collection.verified_corpus import build_verified_corpus
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.runtime import RuntimeMode


def _candidate(root: Path, index: int) -> dict[str, object]:
    document_id = f"candidate-{index:03d}"
    text = f"제목: 목포역 역사 자료 {index}\n기관: 공공 역사 기관\n\n" + " ".join(
        "목포역은 근대 목포의 철도와 항만 교통을 연결한 역사적 장소입니다. "
        "일제강점기 당시 목포의 도시 형성과 상업 활동, 학생운동의 이동 경로를 "
        f"이해하는 데 필요한 기록 {part}입니다. 목포항과 호남선의 변천도 함께 설명합니다."
        for part in range(7)
    )
    raw = f"<main>{text}</main>".encode()
    extracted = text.encode()
    raw_path = root / f"raw/{document_id}.html"
    extracted_path = root / f"extracted/{document_id}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    extracted_path.write_bytes(extracted)
    url = f"https://history.example.org/item/{index}"
    return {
        "candidate_id": document_id,
        "document_id": document_id,
        "source_id": "official_test",
        "source_tier": "tier_1",
        "institution": "공공 역사 기관",
        "publisher": "공공 역사 기관",
        "publisher_family": "official_test",
        "source_title": f"목포역 역사 자료 {index}",
        "source_url": url,
        "canonical_url": url,
        "raw_path": str(raw_path.relative_to(root)),
        "extracted_path": str(extracted_path.relative_to(root)),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "extracted_sha256": hashlib.sha256(extracted).hexdigest(),
        "extraction_status": "success",
        "duplicate_status": "new_unique",
        "rights_status": "unknown",
        "provenance": {"new_unique_increment": 1},
    }


def test_verified_builder_selects_only_valid_and_preserves_rights(tmp_path: Path) -> None:
    rows = [_candidate(tmp_path, index) for index in range(100)]
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = build_verified_corpus(root=tmp_path, candidate_manifest=manifest, output_root=tmp_path / "verified")
    assert report["document_count"] == 100
    chunks = [json.loads(line) for line in (tmp_path / "verified/index_ready/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks
    assert all(item["verification_status"] == "VALID" for item in chunks)
    assert all(item["rights_status"] == "unknown" for item in chunks)
    assert all(item["human_review_required"] is True for item in chunks)
    assert all(item["production_approved"] is False for item in chunks)


def test_context_resolver_uses_place_and_recent_user_topic_only() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    store.add_turn(session.session_id, "목포 학생운동은 어떻게 진행됐어?", "근거 기반 응답")
    store.update_context(session.session_id, recent_event="학생운동", recent_entities=("목포 학생운동",))
    resolved = ConversationContextResolver().resolve(
        "그때 여기서는 무슨 일이 있었어?", session,
        current_place_id="mokpo-station", current_piece_id=None,
    )
    assert resolved.followup_resolved
    assert "목포역" in resolved.search_query
    assert "학생운동" in resolved.search_query
    assert "근거 기반 응답" not in resolved.search_query


def test_hackathon_factory_uses_verified_lane_and_multiturn(tmp_path: Path) -> None:
    rows = [_candidate(tmp_path, index) for index in range(100)]
    manifest = tmp_path / "candidates.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "verified"
    build_verified_corpus(root=tmp_path, candidate_manifest=manifest, output_root=output)
    chat = create_hackathon_orchestrator(
        runtime_dir=tmp_path / "runtime",
        chunks_path=output / "index_ready/chunks.jsonl",
        session_path=tmp_path / "sessions.json",
        llm=MockLLM("목포역 관련 기록을 근거로 설명합니다."),
    )
    first = chat.ask("목포 학생운동은 어떻게 진행됐어?", current_place_id="mokpo-station")
    second = chat.ask("그때 여기서는 무슨 일이 있었어?", session_id=first.session_id, current_place_id="mokpo-station")
    assert second.status == "ok"
    assert second.context_metadata["followup_resolved"] is True
    assert "목포역" in second.context_metadata["search_query"]
    assert chat.retrieval.store.metadata()["data_lane"] == "verified_hackathon"


def test_place_change_replaces_active_place() -> None:
    store = SessionStore(RuntimeMode.HACKATHON)
    session = store.create()
    resolver = ConversationContextResolver()
    first = resolver.resolve("여기는 왜 중요해?", session, current_place_id="mokpo-station", current_piece_id=None)
    store.update_context(session.session_id, active_place=first.active_place)
    second = resolver.resolve("여기는 왜 중요해?", session, current_place_id="mokpo-port", current_piece_id=None)
    assert second.active_place == "목포항"
    assert second.search_query.startswith("목포항")
    assert "목포역" not in second.search_query
