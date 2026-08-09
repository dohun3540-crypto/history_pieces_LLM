import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from history_chatbot.collectors.public_history_batch import (
    ADAPTERS, ACCEPTED_DECISIONS, BatchCandidate, BatchPipeline, BatchResponse,
    RequestController, SOURCE_SPECS,
)
from history_chatbot.history_collection.models import (
    AcceptanceStatus, CandidateDocument, ReviewStatus,
)
from history_chatbot.history_collection.phase_a import phase_a_candidate_record_builder


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests/fixtures/history_collection"
SOURCES = {
    "national_archives_html": "https://www.archives.go.kr/next/newsearch/searchTotal.do?keyword=목포",
    "heritage_portal": "https://www.heritage.go.kr/heri/cul/culSelectViewList.do?searchKeyword=목포",
    "mokpo_official": "https://www.mokpo.go.kr/search/?query=목포",
}


def fixture(source_id, name):
    return (FIXTURES / source_id / (name + ".html")).read_bytes()


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_official_html_discovery_and_detail_fixtures(source_id):
    adapter = ADAPTERS[source_id]
    search = SOURCES[source_id]
    found = adapter.discover(BatchResponse(search, 200, "text/html; charset=utf-8",
                                           fixture(source_id, "discovery")), search)
    assert len(found) == 2
    assert all(item.discovery_metadata["discovery_request_url"] == search for item in found)
    detail = adapter.fetch_detail(found[0], BatchResponse(found[0].source_url, 200,
                                  "text/html; charset=utf-8", fixture(source_id, "detail")))
    assert "목포" in detail.text
    assert detail.metadata["page_title"]
    assert "document_canonical_url" in detail.metadata
    assert "메뉴" not in detail.text


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_official_html_forbidden_and_irrelevant_links_are_not_discovered(source_id):
    search = SOURCES[source_id]
    found = ADAPTERS[source_id].discover(BatchResponse(search, 200, "text/html",
                                         fixture(source_id, "negative")), search)
    assert found == []


def test_mokpo_official_accepts_only_verified_history_section_under_www():
    adapter = ADAPTERS["mokpo_official"]
    root_host = (
        "mokpo"
        + "."
        + "go"
        + "."
        + "kr"
    )
    www_host = (
        "www"
        + "."
        + root_host
    )
    base = (
        "https"
        + ":"
        + "//"
        + www_host
    )

    assert adapter._is_detail_url(
        base + "/www/introduce/history/origin"
    )
    assert adapter._is_detail_url(
        base + "/www/introduce/history/city_development"
    )
    assert not adapter._is_detail_url(
        base + "/www/introduce/general"
    )


class FixtureTransport:
    def __init__(self, responses, calls):
        self.responses, self.calls = responses, calls

    def get(self, url, timeout, max_bytes):
        self.calls.append(url)
        return self.responses[url]


def test_phase_a_builder_counts_and_raw_provenance_for_accepted_rejected_duplicate(tmp_path):
    source_id = "national_archives_html"
    spec = replace(SOURCE_SPECS[source_id], robots_status="verified_allowed", policy_status="allowed")
    adapter = type(ADAPTERS[source_id])(spec)
    urls = ["https://www.archives.go.kr/next/newsearch/showDetailPopup.do?id=" + str(i) for i in range(1, 4)]
    candidates = [BatchCandidate("doc-" + str(i), source_id, "목포 개항 기록 " + str(i), "국가기록원",
                                 url, url, portal_name="국가기록원", original_institution="국가기록원",
                                 discovery_metadata={"discovery_request_url": SOURCES[source_id],
                                                     "discovery_query": "목포"})
                  for i, url in enumerate(urls, 1)]
    body = ("목포는 1897년 개항하였고 해관과 항만 무역의 역사 기록이 남아 있다. " * 12)
    responses = {
        urls[0]: BatchResponse(urls[0] + "&final=1", 200, "text/html; charset=utf-8",
                               ("<html><main>" + body + "</main></html>").encode()),
        urls[1]: BatchResponse(urls[1], 200, "text/html", b"<html><main></main></html>"),
        urls[2]: BatchResponse(urls[2], 200, "text/html",
                               ("<html><main>" + body + "</main></html>").encode()),
    }
    calls = []
    controller = RequestController(5, 1.2, lambda hosts: FixtureTransport(responses, calls),
                                   sleep=lambda value: None, require_source_preflight=True)
    pipeline = BatchPipeline({source_id: adapter}, controller, phase_a_authorized=True)
    catalog = tmp_path / "data/history_candidates/manifests/batch.catalog.jsonl"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("".join(json.dumps(item.__dict__, ensure_ascii=False) + "\n" for item in candidates), encoding="utf-8")
    manifest = tmp_path / "data/history_candidates/manifests/candidates.jsonl"
    raw_dir = tmp_path / "data/history_candidates/raw"
    extracted_dir = tmp_path / "data/history_candidates/extracted"
    builder = phase_a_candidate_record_builder(
        batch_id="batch", source_plan={source_id: {"publisher_family": "national_archives", "source_tier": "tier_1"}},
        readiness={source_id: {"robots_status": "verified_allowed", "policy_status": "allowed",
                               "public_access_status": "public", "rights_metadata_status": "document_level_required",
                               "evidence": []}},
    )
    report = pipeline.execute("batch", [source_id], catalog, manifest, extracted_dir,
                              tmp_path / "report.json", tmp_path / "report.md", 2, 100000,
                              {"max_accepted": 3, "max_per_source": 3, "max_requests": 5, "delay_seconds": 1.2},
                              "2026-08-09T00:00:00+00:00", raw_dir=raw_dir, record_builder=builder)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert report["counts"]["stored"] == 1
    assert report["counts"]["manifested"] == 3
    assert report["counts"]["raw_artifacts"] == 3
    assert report["counts"]["extracted_artifacts"] == 2
    assert [row["duplicate_status"] for row in rows] == ["unique", "unique", "confirmed"]
    assert rows[1]["extraction_status"] == "failed"
    for row, response in zip(rows, responses.values()):
        assert hashlib.sha256(response.body).hexdigest() == row["raw_sha256"]
        assert row["response_final_url"] == response.final_url
        assert row["response_http_status"] == response.status
        assert row["response_content_type"] == response.content_type.split(";", 1)[0]
        CandidateDocument.from_dict(row)
    assert not (tmp_path / "data/provisional_hackathon").exists()


def test_legacy_record_builder_none_keeps_legacy_manifest_shape(tmp_path):
    source_id = "national_archives"
    item = BatchCandidate("legacy-1", source_id, "목포 개항 역사 기록", "국가기록원",
                          "https://www.archives.go.kr/detail/1", "https://www.archives.go.kr/detail/1")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(item.__dict__, ensure_ascii=False) + "\n", encoding="utf-8")
    response = BatchResponse(item.source_url, 200, "text/html",
                             ("<html><main>" + "목포 개항 항만 역사 기록 " * 40 + "</main></html>").encode())
    controller = RequestController(2, 1.2, lambda hosts: FixtureTransport({item.source_url: response}, []),
                                   sleep=lambda value: None)
    manifest = tmp_path / "manifest.jsonl"
    BatchPipeline({source_id: ADAPTERS[source_id]}, controller).execute(
        "legacy", [source_id], catalog, manifest, tmp_path / "extracted",
        tmp_path / "report.json", tmp_path / "report.md", 2, 100000,
        {"max_accepted": 1, "max_per_source": 1, "max_requests": 2, "delay_seconds": 1.2},
        "2026-08-09T00:00:00+00:00")
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert "document_id" in row and "candidate_id" not in row
    assert "collection_metadata" in row and "raw_path" not in row


def test_phase_a_builder_keeps_unknown_rights_fail_closed_with_kogl_evidence(tmp_path):
    source_id = "national_archives_html"
    url = "https://www.archives.go.kr/next/newsearch/showDetailPopup.do?id=rights"
    candidate = BatchCandidate(
        "rights-fixture", source_id, "목포 개항과 해관 기록", "국가기록원",
        url, url, portal_name="국가기록원", original_institution="국가기록원",
        discovery_metadata={
            "discovery_request_url": SOURCES[source_id],
            "discovery_response_final_url": SOURCES[source_id],
            "discovery_query": "목포",
        },
    )
    response = BatchResponse(url, 200, "text/html; charset=utf-8",
                             fixture(source_id, "detail"))
    detail = ADAPTERS[source_id].fetch_detail(candidate, response)
    assert detail.metadata["kogl_type"] == "KOGL-1"
    builder = phase_a_candidate_record_builder(
        batch_id="rights-batch",
        source_plan={source_id: {
            "publisher_family": "national_archives", "source_tier": "tier_1",
            "policy_url": "https://www.archives.go.kr/policy",
        }},
        readiness={source_id: {
            "robots_status": "verified_allowed", "policy_status": "allowed",
            "public_access_status": "public",
            "rights_metadata_status": "document_level_required", "evidence": [],
        }},
    )
    row = builder(
        candidate=candidate, detail=detail, response=response,
        raw_target=tmp_path / "data/history_candidates/raw/rights-fixture.html",
        extracted_target=tmp_path / "data/history_candidates/extracted/rights-fixture.txt",
        decision="accepted_hackathon", collected_at="2026-08-09T00:00:00+00:00",
        body_hash="body-sha", extracted_hash="extracted-sha", reasons=[], warnings=[],
    )
    document = CandidateDocument.from_dict(row)
    assert document.rights_status == "unknown"
    assert document.rights_evidence.human_review_required is True
    assert document.review_status != ReviewStatus.VERIFIED
    assert document.review_status != ReviewStatus.ACCEPTED
    assert document.acceptance_status != AcceptanceStatus.ACCEPTED
