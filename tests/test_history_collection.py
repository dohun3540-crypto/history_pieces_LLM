import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from history_chatbot.history_collection.balance import calculate_balance
from history_chatbot.history_collection.checkpoint import GateThresholds, build_checkpoint
from history_chatbot.history_collection.corroboration import FactCandidate, assess_facts
from history_chatbot.history_collection.dedup import DuplicateIndex, canonicalize_url
from history_chatbot.history_collection.models import (
    AcceptanceStatus, CandidateDocument, DuplicateStatus, HardRejectionCode,
    Phase, ReviewStatus, RightsEvidence, SourceTier, TopicCategory,
    TRUST_GRADE_TO_TIER, VerifiedDocument,
)
from history_chatbot.history_collection.pipeline import AppendOnlyJsonl, DecisionAuditLog, DecisionEvent, atomic_write
from history_chatbot.history_collection.quality import classify_topics, evaluate_content
from history_chatbot.history_collection.reporting import write_report
from history_chatbot.history_collection.rights import evaluate_rights, normalize_kogl
from history_chatbot.history_collection.scoring import classify_score, score_candidate


ROOT = Path(__file__).parents[1]


def candidate(number=1, **overrides):
    body = ("목포는 1897년 개항한 뒤 목포항과 해관을 중심으로 무역이 성장하였다. "
            "기록 자료에 따르면 항만 시설과 도시 기반시설도 확장되었다. ") * 8
    values = dict(
        candidate_id=f"candidate-{number}", batch_id="phase-a-fixture", phase=Phase.A,
        source_id="fixture_official",
        source_url=f"https://archives.example/item/{number}", canonical_url=f"https://archives.example/item/{number}",
        source_title=f"목포 개항 기록 {number}", publisher="공식기관", institution="공식기관",
        publisher_family=f"publisher-{number % 3}", source_tier=SourceTier.TIER_1,
        document_type="historical_document", topic_categories=[], historical_period="개항기",
        location=["목포"], discovered_at="2026-08-09T00:00:00Z", fetched_at="2026-08-09T00:00:01Z",
        raw_path=f"data/history_candidates/raw/{number}.html", raw_sha256=f"sha-{number}",
        extracted_path=f"data/history_candidates/extracted/{number}.txt",
        extracted_sha256=f"extracted-sha-{number}", normalized_body_sha256=f"body-sha-{number}",
        response_final_url=f"https://archives.example/item/{number}",
        response_http_status=200, response_content_type="text/html",
        extraction_status="success", extraction_method="html", language="ko",
        robots_status="verified_allowed", access_status="public", rights_status="open_license",
        license="KOGL-1", kogl_type="KOGL-1",
        rights_evidence=RightsEvidence(
            publisher="공식기관", rights_holder="공식기관", policy_url="https://archives.example/policy",
            document_rights_url=f"https://archives.example/item/{number}#rights", license_text="KOGL-1",
            kogl_type="KOGL-1", commercial_use="allowed", modification="allowed",
            redistribution="allowed", checked_at="2026-08-09T00:00:00Z", human_review_required=False,
        ),
        provenance={"record_id": f"record-{number}", "discovered_from": "official-api"}, body_text=body,
    )
    values.update(overrides)
    return CandidateDocument(**values)


def test_schema_required_fields_match_python_models():
    candidate_schema = json.loads((ROOT / "data/schemas/history_candidate.schema.json").read_text(encoding="utf-8"))
    verified_schema = json.loads((ROOT / "data/schemas/history_verified_document.schema.json").read_text(encoding="utf-8"))
    candidate_fields = {item.name for item in fields(CandidateDocument)} - {"body_text"}
    verified_fields = {item.name for item in fields(VerifiedDocument)}
    assert set(candidate_schema["required"]) == candidate_fields
    assert set(candidate_schema["properties"]) == candidate_fields
    assert set(verified_schema["required"]) == verified_fields
    assert set(verified_schema["properties"]) == verified_fields


def test_candidate_document_id_defaults_to_candidate_id_and_rejects_mismatch():
    item = candidate()
    assert item.document_id == item.candidate_id
    assert item.to_dict()["document_id"] == item.candidate_id
    with pytest.raises(ValueError, match="document_id must match candidate_id"):
        candidate(document_id="different-document")


@pytest.mark.parametrize("score,expected", [(59, ReviewStatus.AUTO_REJECTED), (60, ReviewStatus.NEEDS_HUMAN_REVIEW),
                                             (74, ReviewStatus.NEEDS_HUMAN_REVIEW), (75, ReviewStatus.AUTO_CANDIDATE)])
def test_scoring_boundaries(score, expected):
    status, _ = classify_score(score, False, True, False)
    assert status == expected
    assert classify_score(100, True, True, False)[0] == ReviewStatus.AUTO_REJECTED


def test_source_tier_mapping_is_separate_and_tier4_cannot_be_accepted():
    assert TRUST_GRADE_TO_TIER["A"] == SourceTier.TIER_1
    item = candidate(source_tier=SourceTier.TIER_4, review_status=ReviewStatus.VERIFIED)
    with pytest.raises(ValueError, match="Tier 4"):
        VerifiedDocument.from_candidate(item, verified_document_id="v1", verified_at="now",
                                        verified_by="reviewer", processed_path="processed/v1.jsonl",
                                        verification_notes="reviewed")


def test_rights_and_robots_fail_closed_and_kogl_mapping():
    evidence = candidate().rights_evidence
    assert normalize_kogl("공공누리 제1유형") == "KOGL-1"
    assert evaluate_rights("verified_allowed", "public", "open_license", evidence).usable_for_rag
    blocked = evaluate_rights("unknown", "public", "open_license", evidence)
    assert HardRejectionCode.ROBOTS_BLOCKED in blocked.hard_rejections
    ambiguous = evaluate_rights("verified_allowed", "public", "unknown", replace(evidence, human_review_required=True))
    assert not ambiguous.usable_for_rag and ambiguous.needs_human_review


@pytest.mark.parametrize("access,code", [("login_required", HardRejectionCode.LOGIN_REQUIRED),
                                          ("captcha", HardRejectionCode.CAPTCHA), ("paywall", HardRejectionCode.PAYWALL)])
def test_access_barriers_are_hard_rejections(access, code):
    result = evaluate_rights("verified_allowed", access, "open_license", candidate().rights_evidence)
    assert code in result.hard_rejections


def test_content_quality_and_topic_multilabel():
    text = "목포는 1897년 개항하였다. 목포항과 호남선 철도, 호남은행과 일본영사관이 도시 변화에 영향을 주었다. " * 8
    quality = evaluate_content("목포 근대사", text)
    topics = classify_topics("목포 개항과 철도", text)
    assert quality.score >= 7 and not quality.noise
    assert {TopicCategory.OPENING_TRADE, TopicCategory.PORT_MARITIME,
            TopicCategory.RAIL_TRANSPORT, TopicCategory.ECONOMY_FINANCE,
            TopicCategory.ARCHITECTURE_HERITAGE} <= set(topics)
    assert evaluate_content("제목", "제목").title_only


def test_score_populates_100_point_fields_and_hard_rejection_wins():
    item = candidate()
    quality = evaluate_content(item.source_title, item.body_text)
    rights = evaluate_rights(item.robots_status, item.access_status, item.rights_status, item.rights_evidence)
    score_candidate(item, quality, rights, corroboration_score=15)
    assert item.total_quality_score == sum((item.authority_score, item.provenance_score,
                                            item.historical_relevance_score, item.mokpo_relevance_score,
                                            item.corroboration_score, item.content_quality_score,
                                            item.objectivity_score, item.rights_score))
    assert item.total_quality_score <= 100
    assert item.uniqueness_score == 10
    blocked = candidate(2, source_tier=SourceTier.TIER_4)
    score_candidate(blocked, evaluate_content(blocked.source_title, blocked.body_text), rights, corroboration_score=15)
    assert blocked.review_status == ReviewStatus.AUTO_REJECTED
    assert HardRejectionCode.TIER4_DISCOVERY_ONLY in blocked.rejection_reasons


def test_canonical_raw_and_normalized_body_dedup():
    index = DuplicateIndex()
    first = candidate()
    index.add(first)
    by_url = candidate(2, canonical_url=first.canonical_url + "?utm_source=x")
    assert index.add(by_url).method == "canonical_url"
    by_raw = candidate(3, raw_sha256=first.raw_sha256)
    assert index.add(by_raw).method == "raw_sha256"
    by_body = candidate(4, body_text="  " + first.body_text.replace(" ", "  "))
    assert index.add(by_body).method == "normalized_body_sha256"
    assert canonicalize_url("HTTPS://EXAMPLE.COM/a?utm_source=x&b=2") == "https://example.com/a?b=2"


def test_simhash_near_candidate_and_person_title_false_positive():
    index = DuplicateIndex(near_distance=16)
    first = candidate(body_text=("목포 항만 철도 도시 역사 기록 자료 " * 80))
    index.add(first)
    near = candidate(2, body_text=("목포 항만 철도 도시 역사 기록 자료 " * 79) + "추가 문장")
    decision = index.add(near)
    assert decision.status == DuplicateStatus.SUSPECTED
    people = DuplicateIndex()
    people.add(candidate(10, source_title="김옥실 - 독립운동인명사전", body_text="김옥실 목포 만세운동 " * 30))
    other = candidate(11, source_title="김옥남 - 독립운동인명사전", body_text="김옥남 정명여학교 학생운동 " * 30)
    assert people.add(other).status != DuplicateStatus.CONFIRMED


def test_confirmed_mirror_preserves_group():
    index = DuplicateIndex(near_distance=16)
    first = candidate(body_text="목포 개항 항만 도시 역사 원문 " * 60,
                      provenance={"record_id": "a", "canonical_work_id": "work-1"})
    index.add(first)
    mirror = candidate(2, publisher_family="mirror-publisher",
                       body_text="목포 개항 항만 도시 역사 원문 " * 59 + "복제 표시",
                       provenance={"record_id": "b", "canonical_work_id": "work-1"})
    decision = index.add(mirror)
    assert decision.status == DuplicateStatus.CONFIRMED and decision.method == "mirror"
    assert mirror.duplicate_of == first.candidate_id and mirror.duplicate_group


def test_corroboration_independence_and_conflict_preservation():
    facts = [
        FactCandidate("호남선", "개통일", "1914년 1월", "1914-01", "month", "목포", "근거1", "a", "family-a", .9),
        FactCandidate("호남선", "개통일", "1914년 1월", "1914-01", "month", "목포", "복제", "b", "family-a", .8),
    ]
    assert not assess_facts(facts)[0].corroborated
    corroborated = assess_facts(facts + [replace(facts[0], candidate_id="c", publisher_family="family-b")])[0]
    assert corroborated.corroborated
    conflict = assess_facts(facts + [replace(facts[0], value="1913년", normalized_value="1913", candidate_id="d", publisher_family="family-c")])[0]
    assert conflict.fact_conflict and len(conflict.conflicting_values) == 2
    assert conflict.unresolved_reason


def test_balance_and_checkpoint_pass_stop():
    records = []
    qualities = {}
    topics = [TopicCategory.OPENING_TRADE, TopicCategory.ECONOMY_FINANCE,
              TopicCategory.URBAN_INFRASTRUCTURE]
    for number in range(50):
        item = candidate(number, publisher_family=f"publisher-{number % 4}",
                         topic_categories=[topics[number % len(topics)]])
        item.mokpo_relevance_score = 10
        item.review_status = ReviewStatus.AUTO_CANDIDATE
        records.append(item)
        qualities[item.candidate_id] = evaluate_content(item.source_title, item.body_text)
    checkpoint = build_checkpoint(Phase.A, records, qualities, discovered=50, fetched=50)
    assert checkpoint.gate_status == "PASS"
    concentrated = [replace(item, publisher_family="one") for item in records]
    stopped = build_checkpoint(Phase.A, concentrated, qualities, discovered=50, fetched=50)
    assert stopped.gate_status == "STOP" and "publisher_concentration" in stopped.stop_reasons
    assert calculate_balance(concentrated).largest_publisher_share == 1.0


def test_append_only_audit_atomic_write_and_report_no_overwrite(tmp_path):
    audit = DecisionAuditLog(tmp_path / "decisions.jsonl")
    event = DecisionEvent("c1", "b1", "A", "now", "score", "draft", "auto_candidate", ())
    audit.append_event(event)
    audit.append_event(replace(event, action="review", new_status="verified"))
    assert len((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    target = tmp_path / "atomic.json"
    atomic_write(target, b"one")
    atomic_write(target, b"two")
    assert target.read_bytes() == b"two"
    records = [candidate(number, publisher_family=f"p-{number % 4}", topic_categories=[TopicCategory.OPENING_TRADE]) for number in range(50)]
    qualities = {item.candidate_id: evaluate_content(item.source_title, item.body_text) for item in records}
    for item in records:
        item.mokpo_relevance_score = 10
    checkpoint = build_checkpoint(Phase.A, records, qualities, discovered=50, fetched=50,
                                  thresholds=GateThresholds(max_topic_share=1.0))
    write_report(tmp_path / "reports", checkpoint, "batch-1")
    with pytest.raises(FileExistsError):
        write_report(tmp_path / "reports", checkpoint, "batch-1")
