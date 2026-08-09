import json
from dataclasses import replace
from pathlib import Path

import pytest

from history_chatbot.collectors.public_history_batch import (
    BatchError, BatchPipeline, BatchResponse, RequestController, SOURCE_SPECS,
)
from history_chatbot.history_collection.phase_a import (
    EXECUTION_ACKNOWLEDGEMENT, PhaseAExecutor, build_verified_adapters,
)
from history_chatbot.history_collection.pipeline import historical_unique_target, metadata_discovery_quota
from history_chatbot.history_collection.preflight import PhaseAPreflight, PreflightController
from scripts.collect_history_1000 import main


class FixtureTransport:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    def get(self, url, timeout, max_bytes):
        self.calls.append(url)
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return value


def response(url, body, content_type="text/html"):
    return BatchResponse(url, 200, content_type, body.encode("utf-8"))


def source_plan(**overrides):
    value = {
        "source_id": "national_archives_html", "publisher_family": "national_archives",
        "source_tier": "tier_1",
        "unique_target": 10, "discovery_metadata_quota": 0, "source_role": "primary_official",
        "readiness": "robots_policy_preflight_required", "discovery_request_budget": 2,
        "detail_request_budget": 10, "robots_url": "https://www.archives.go.kr/robots.txt",
        "robots_target_url": "https://www.archives.go.kr/next/newsearch/searchTotal.do?keyword=x",
        "policy_url": "https://www.archives.go.kr/policy", "allowed_hosts": ["www.archives.go.kr"],
    }
    value.update(overrides)
    return value


def ready_report(source_id="national_archives_html"):
    return {
        "phase": "A", "status": "PASS", "collection_network_requests": 0,
        "sources": [{
            "source_id": source_id, "robots_status": "verified_allowed",
            "policy_status": "allowed", "endpoint_status": "not_applicable",
            "api_key_status": "not_applicable", "rights_metadata_status": "document_level_required",
            "public_access_status": "public", "crawl_delay_seconds": 0,
            "collection_ready": True,
        }],
    }


def phase_config(plan=None):
    return {
        "phase_a_source_plan": plan or [source_plan()],
        "phase_a_request_budget": {
            "collection_maximum_requests": 20, "delay_seconds": 1.5,
            "timeout_seconds": 2, "max_response_bytes": 10000,
        },
    }


def test_preflight_has_separate_counter_and_creates_no_candidate_or_raw(tmp_path, monkeypatch):
    calls = []
    plan = source_plan()
    responses = {
        plan["robots_url"]: response(plan["robots_url"], "User-agent: *\nAllow: /", "text/plain"),
        plan["policy_url"]: response(plan["policy_url"], "공식 저작권 정책"),
    }
    controller = PreflightController(2, lambda hosts: FixtureTransport(responses, calls))
    monkeypatch.chdir(tmp_path)
    report = PhaseAPreflight(controller, {}).run([plan], [plan["source_id"]])
    assert report.preflight_network_requests == 2
    assert report.collection_network_requests == 0
    assert report.candidates_created == 0 and report.raw_history_documents_created == 0
    assert not list(tmp_path.iterdir())


def test_unknown_robots_and_policy_never_call_collection_transport():
    calls = []
    plan = source_plan()
    responses = {
        plan["robots_url"]: response(plan["robots_url"], "User-agent: *\nDisallow: /next/newsearch", "text/plain"),
        plan["policy_url"]: response(plan["policy_url"], "조건은 사람 검토 필요"),
    }
    report = PhaseAPreflight(PreflightController(2, lambda hosts: FixtureTransport(responses, calls)), {}).run(
        [plan], [plan["source_id"]]
    )
    source = report.sources[0]
    assert source.robots_status == "blocked"
    assert source.policy_status == "needs_human_review"
    assert not source.collection_ready and report.collection_network_requests == 0


def test_phase_a_pipeline_always_requires_preflight_controller():
    controller = RequestController(2, 1.2, lambda hosts: None, require_source_preflight=False)
    with pytest.raises(BatchError, match="require_source_preflight=True"):
        BatchPipeline({}, controller, phase_a_authorized=True)


def test_unknown_policy_stops_before_transport():
    calls = []

    class NeverCalled:
        def get(self, url, timeout, max_bytes):
            calls.append(url)
            raise AssertionError

    controller = RequestController(1, 1.2, lambda hosts: NeverCalled(),
                                   require_source_preflight=True)
    spec = replace(SOURCE_SPECS["national_archives_html"],
                   robots_status="verified_allowed", policy_status="unknown")
    with pytest.raises(BatchError, match="source_policy_not_allowed"):
        controller.get("https://www.archives.go.kr/robots.txt", spec, 1, 1000)
    assert calls == [] and controller.request_count == 0


def test_api_key_missing_stops_before_protected_endpoint_transport():
    plan = source_plan(source_id="national_archives_api", unique_target=6,
                       api_key_environment="NATIONAL_ARCHIVES_API_KEY")
    report = ready_report("national_archives_api")
    report["sources"][0]["endpoint_status"] = "verified"
    with pytest.raises(BatchError, match="API_KEY_MISSING"):
        build_verified_adapters([plan], report, {})


def test_metadata_only_quota_does_not_count_as_historical_unique():
    plan = [source_plan(unique_target=50),
            source_plan(source_id="data_portal", unique_target=0, discovery_metadata_quota=5,
                        source_role="metadata_discovery_only")]
    assert historical_unique_target(plan) == 50
    assert metadata_discovery_quota(plan) == 5


def test_phase_a_execution_requires_explicit_safety_flag_and_later_phases_block(capsys, tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(ready_report()), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--phase", "A", "--discover", "--allow-network",
              "--maximum-total-requests", "120", "--preflight-report", str(report)])
    with pytest.raises(SystemExit):
        main(["--phase", "B", "--preflight", "--allow-network", "--source", "x",
              "--maximum-total-requests", "1"])


def test_source_stage_request_budget_is_enforced():
    calls = []
    url = "https://www.archives.go.kr/x"
    responses = {url: response(url, "ok")}
    spec = SOURCE_SPECS["national_archives_html"]
    verified = type(spec)(**{**spec.__dict__, "robots_status": "verified_allowed", "policy_status": "allowed"})
    controller = RequestController(3, 1.2, lambda hosts: FixtureTransport(responses, calls),
                                   sleep=lambda value: None, require_source_preflight=True,
                                   source_stage_limits={spec.source_id: {"discovery": 1}})
    controller.get(url, verified, 1, 1000, "discovery")
    with pytest.raises(BatchError, match="source_stage_request_budget_exceeded"):
        controller.get(url, verified, 1, 1000, "discovery")
    assert len(calls) == 1


def test_source_specific_delay_is_applied_and_cannot_weaken_global_delay():
    sleeps = []
    ticks = iter([0.0, 0.0, 0.0, 0.0])
    url = "https://www.archives.go.kr/x"
    spec = replace(SOURCE_SPECS["national_archives_html"],
                   robots_status="verified_allowed", policy_status="allowed")
    controller = RequestController(
        2, 1.5, lambda hosts: FixtureTransport({url: response(url, "ok")}, []),
        sleep=sleeps.append, clock=lambda: next(ticks), require_source_preflight=True,
        source_delay_seconds={spec.source_id: 60},
    )
    controller.get(url, spec, 1, 1000)
    controller.get(url, spec, 1, 1000)
    assert sleeps == [60.0]
    with pytest.raises(BatchError, match="source-specific delay"):
        RequestController(1, 1.5, lambda hosts: None,
                          source_delay_seconds={spec.source_id: 1.0})


def test_preflight_crawl_delay_prefers_user_agent_rule():
    plan = source_plan(minimum_delay_seconds=10)
    robots = (
        "User-agent: *\n"
        "Allow: /\n"
        "Crawl-delay: 7\n"
        "\n"
        "User-agent: MokpoHistoryRAGCollector\n"
        "Allow: /\n"
        "Crawl-delay: 60\n"
    )
    responses = {
        plan["robots_url"]: response(plan["robots_url"], robots, "text/plain"),
        plan["policy_url"]: response(plan["policy_url"], "공식 정책"),
    }
    result = PhaseAPreflight(
        PreflightController(2, lambda hosts: FixtureTransport(responses, [])), {}
    ).run([plan], [plan["source_id"]]).sources[0]
    assert result.crawl_delay_seconds == 60
    assert "CONFIGURED_DELAY_BELOW_ROBOTS" in result.blockers


def test_preflight_crawl_delay_falls_back_to_wildcard():
    plan = source_plan(minimum_delay_seconds=7)
    robots = "User-agent: *\nAllow: /\nCrawl-delay: 7\n"
    responses = {
        plan["robots_url"]: response(plan["robots_url"], robots, "text/plain"),
        plan["policy_url"]: response(plan["policy_url"], "공식 정책"),
    }
    result = PhaseAPreflight(
        PreflightController(2, lambda hosts: FixtureTransport(responses, [])), {}
    ).run([plan], [plan["source_id"]]).sources[0]
    assert result.crawl_delay_seconds == 7
    assert "CONFIGURED_DELAY_BELOW_ROBOTS" not in result.blockers


def test_preflight_crawl_delay_defaults_to_zero_when_absent():
    plan = source_plan(minimum_delay_seconds=1.5)
    robots = "User-agent: *\nAllow: /\n"
    responses = {
        plan["robots_url"]: response(plan["robots_url"], robots, "text/plain"),
        plan["policy_url"]: response(plan["policy_url"], "공식 정책"),
    }
    result = PhaseAPreflight(
        PreflightController(2, lambda hosts: FixtureTransport(responses, [])), {}
    ).run([plan], [plan["source_id"]]).sources[0]
    assert result.crawl_delay_seconds == 0
    assert "CONFIGURED_DELAY_BELOW_ROBOTS" not in result.blockers


def test_verified_phase_a_execution_bridge_uses_candidate_lane(tmp_path):
    calls = []
    search = SOURCE_SPECS["national_archives_html"].discovery_templates[0].format(query="%EB%AA%A9%ED%8F%AC")
    html = '<html><a href="/detail/1">목포 개항 역사 기록</a></html>'
    responses = {search: response(search, html)}
    executor = PhaseAExecutor(phase_config(), ready_report(), environment={},
                              transport_factory=lambda hosts: FixtureTransport(responses, calls),
                              sleep=lambda value: None)
    with pytest.raises(BatchError, match="acknowledgement"):
        executor.discover(acknowledgement="", batch_id="fixture", keywords=["목포"],
                          output_root=tmp_path / "data", timeout=1, max_bytes=10000)
    assert calls == []
    result = executor.discover(acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
                               batch_id="fixture", keywords=["목포"],
                               output_root=tmp_path / "data", timeout=1, max_bytes=10000)
    assert result["preflight_forced"] is True
    assert result["collection_network_requests"] == 1
    assert "history_candidates" in result["candidate_lane"]
    assert not (tmp_path / "data/provisional_hackathon").exists()


def test_verified_phase_a_collect_bridge_discovers_and_fetches_only_candidate_lane(tmp_path):
    calls = []
    search = SOURCE_SPECS["national_archives_html"].discovery_templates[0].format(query="%EB%AA%A9%ED%8F%AC")
    detail = "https://www.archives.go.kr/detail/1"
    discovery_html = '<html><a href="/detail/1">목포 개항 역사 기록</a></html>'
    detail_html = "<html><title>목포 개항 역사 기록</title><main>" + (
        "목포는 1897년 개항하였고 해관과 항만을 중심으로 무역과 도시가 성장하였다. " * 12
    ) + "</main></html>"
    responses = {search: response(search, discovery_html), detail: response(detail, detail_html)}
    executor = PhaseAExecutor(phase_config(), ready_report(), environment={},
                              transport_factory=lambda hosts: FixtureTransport(responses, calls),
                              sleep=lambda value: None)
    result = executor.collect(acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
                              batch_id="fixture-collect", keywords=["목포"],
                              output_root=tmp_path / "data", timeout=1, max_bytes=10000)
    assert result["collection"]["counts"]["stored"] == 1
    assert result["collection_network_requests"] == 2
    assert (tmp_path / "data/history_candidates/manifests/candidates.jsonl").is_file()
    assert not (tmp_path / "data/provisional_hackathon").exists()
