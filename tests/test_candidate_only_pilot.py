import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from history_chatbot.collectors.public_history_batch import BatchError, BatchResponse
from history_chatbot.history_collection.models import (
    AcceptanceStatus,
    CandidateDocument,
    ReviewStatus,
)
from history_chatbot.history_collection.phase_a import (
    EXECUTION_ACKNOWLEDGEMENT,
    CandidateOnlyExecutor,
    build_candidate_ready_adapters,
    build_verified_adapters,
)


HERITAGE_BASE = (
    "https"
    + "://"
    + "www"
    + ".heritage.go.kr"
)

URL = (
    HERITAGE_BASE
    + "/heri/cul/culSelectDetail.do"
    + "?ccbaCpno=4413607180000"
    + "&pageNo=1_1_1_0"
)
BASELINE_ID = "mokpo-ff314916d24c0cb5"


class FixtureTransport:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    def get(self, url, timeout, max_bytes):
        self.calls.append(url)
        return self.responses[url]


def config():
    return {
        "phase_a_source_plan": [{
            "source_id": "heritage_portal",
            "publisher_family": "heritage_agency",
            "source_tier": "tier_1",
            "unique_target": 10,
            "minimum_delay_seconds": 1.5,
            "allowed_hosts": [
                "www.heritage.go.kr",
                "heritage.go.kr",
            ],
            "policy_url": "",
        }],
        "phase_a_request_budget": {
            "collection_maximum_requests": 120,
            "delay_seconds": 1.5,
            "timeout_seconds": 2,
            "max_response_bytes": 100000,
        },
    }


def candidate_readiness():
    return {
        "phase": "A",
        "status": "PARTIAL",
        "collection_network_requests": 0,
        "sources": [{
            "source_id": "heritage_portal",
            "robots_status": "verified_allowed",
            "policy_status": "needs_human_review",
            "rights_metadata_status": "document_level_required",
            "public_access_status": "public",
            "live_extraction_status": "success",
            "candidate_collection_ready": True,
            "verified_collection_ready": False,
            "crawl_delay_seconds": 0,
            "blockers": [],
        }],
    }


def write_seed(
    path: Path,
    *,
    duplicate_of: str = BASELINE_ID,
    count: int = 1,
):
    rows = []
    for number in range(count):
        suffix = "" if number == 0 else str(number)
        rows.append({
            "source_id": "heritage_portal",
            "document_id": f"candidate-heritage-pilot-{number + 1}",
            "title": "목포 근대역사문화공간" + suffix,
            "source_url": URL + suffix,
            "canonical_url": URL + suffix,
            "institution": "국가유산청",
            "publisher_family": "heritage_agency",
            "duplicate_of": duplicate_of if number == 0 else "",
        })
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def detail_response():
    body = (
        "<html><head>"
        "<title>국가등록문화유산 목포 근대역사문화공간</title>"
        f'<link rel="canonical" href="{URL}">'
        "</head><body><main>"
        + (
            "목포 근대역사문화공간은 개항 이후 형성된 도시와 항만, "
            "상업 건축의 변화를 보여준다. " * 15
        )
        + "</main></body></html>"
    ).encode("utf-8")
    return BatchResponse(
        URL,
        200,
        "text/html; charset=utf-8",
        body,
    )


def run_pilot(tmp_path: Path, baseline_rows):
    seed = tmp_path / "seed.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    write_seed(
        seed,
        duplicate_of=BASELINE_ID if baseline_rows else "",
    )
    baseline.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in baseline_rows
        ),
        encoding="utf-8",
    )

    calls = []
    executor = CandidateOnlyExecutor(
        config(),
        candidate_readiness(),
        source_ids=["heritage_portal"],
        maximum_total_requests=1,
        environment={},
        transport_factory=lambda hosts: FixtureTransport(
            {URL: detail_response()},
            calls,
        ),
        sleep=lambda seconds: None,
    )
    result = executor.collect_exact(
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        batch_id="candidate-pilot",
        exact_seed_catalog=seed,
        baseline_manifest=baseline,
        output_root=tmp_path / "data",
        max_documents=1,
        timeout=1,
        max_bytes=100000,
    )

    manifest = (
        tmp_path
        / "data/history_candidates/manifests/candidates.jsonl"
    )
    record = json.loads(
        manifest.read_text(encoding="utf-8").strip()
    )
    return result, CandidateDocument.from_dict(record), calls


def test_heritage_url_and_allowed_hosts_are_plain_strings():
    assert urlsplit(URL).hostname == "www.heritage.go.kr"
    assert "[" not in URL
    assert "](" not in URL
    assert config()["phase_a_source_plan"][0]["allowed_hosts"] == [
        "www.heritage.go.kr",
        "heritage.go.kr",
    ]


def test_candidate_only_accepts_human_review_policy_but_verified_stays_strict():
    adapters = build_candidate_ready_adapters(
        config()["phase_a_source_plan"],
        candidate_readiness(),
        {},
        ["heritage_portal"],
    )
    assert adapters["heritage_portal"].spec.candidate_only is True

    with pytest.raises(BatchError, match="PASS report"):
        build_verified_adapters(
            config()["phase_a_source_plan"],
            candidate_readiness(),
            {},
        )


def test_baseline_duplicate_is_stored_but_does_not_increment_unique(tmp_path):
    baseline = [{
        "document_id": BASELINE_ID,
        "source_id": "heritage_portal",
        "source_url": URL,
        "canonical_url": URL,
    }]

    result, record, calls = run_pilot(tmp_path, baseline)

    assert calls == [URL]
    assert result["stored"] == 1
    assert result["actual_candidates_created"] == 1
    assert result["new_unique_increment"] == 0
    assert record.duplicate_status.value == "confirmed"
    assert record.duplicate_of == BASELINE_ID
    assert record.provenance["new_unique_increment"] == 0
    assert result["new_unique_increment"] == record.provenance["new_unique_increment"]


def test_new_candidate_is_stored_and_increments_unique(tmp_path):
    result, record, calls = run_pilot(tmp_path, [])

    assert calls == [URL]
    assert result["stored"] == 1
    assert result["actual_candidates_created"] == 1
    assert result["new_unique_increment"] == 1
    assert record.duplicate_status.value == "unique"
    assert record.duplicate_of == ""
    assert record.provenance["new_unique_increment"] == 1
    assert result["new_unique_increment"] == record.provenance["new_unique_increment"]
    assert record.rights_status == "unknown"
    assert record.rights_evidence.human_review_required is True
    assert record.review_status not in {ReviewStatus.VERIFIED, ReviewStatus.ACCEPTED}
    assert record.acceptance_status not in {
        AcceptanceStatus.VERIFIED, AcceptanceStatus.ACCEPTED,
    }


def test_candidate_only_rights_remain_human_review_and_never_promote(tmp_path):
    _, record, _ = run_pilot(tmp_path, [])

    assert record.rights_status == "unknown"
    assert record.rights_evidence.human_review_required is True
    assert record.review_status not in {
        ReviewStatus.VERIFIED,
        ReviewStatus.ACCEPTED,
    }
    assert record.acceptance_status not in {
        AcceptanceStatus.VERIFIED,
        AcceptanceStatus.ACCEPTED,
    }


def test_exact_seed_and_request_ceiling_are_enforced_before_network(tmp_path):
    seed = tmp_path / "seed.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    write_seed(seed, count=2)
    baseline.write_text("", encoding="utf-8")

    calls = []
    executor = CandidateOnlyExecutor(
        config(),
        candidate_readiness(),
        source_ids=["heritage_portal"],
        maximum_total_requests=1,
        environment={},
        transport_factory=lambda hosts: FixtureTransport({}, calls),
        sleep=lambda seconds: None,
    )

    with pytest.raises(BatchError, match="seed count"):
        executor.collect_exact(
            acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
            batch_id="candidate-pilot",
            exact_seed_catalog=seed,
            baseline_manifest=baseline,
            output_root=tmp_path / "data",
            max_documents=1,
            timeout=1,
            max_bytes=100000,
        )

    assert calls == []
    assert not (tmp_path / "data/history_candidates").exists()
