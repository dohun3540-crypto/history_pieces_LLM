import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from history_chatbot.collectors.base import (
    BaseCollector,
    CollectedCandidate,
    CollectionError,
    CollectionReport,
    CollectorConfig,
    FetchResponse,
    load_collector_configs,
)
from history_chatbot.collectors.cli import main as collector_cli_main
from history_chatbot.collectors.pilot import (
    MAX_RESULTS_PER_SOURCE,
    MAX_TOTAL_RESULTS,
    build_pilot_plan,
    enforce_candidate_safety,
    run_pilot,
)
from history_chatbot.collectors.registry import CandidateRegistry
from history_chatbot.ingestion.license_policy import can_use_for_rag
from history_chatbot.ingestion.validator import can_index_for_service


class TestCollector(BaseCollector):
    __test__ = False


class FakeTransport:
    def __init__(self, responses: dict[str, FetchResponse]) -> None:
        self.responses = responses

    def request(self, url: str, *, timeout: float, user_agent: str) -> FetchResponse:
        try:
            return self.responses[url]
        except KeyError as error:
            raise CollectionError(f"테스트 네트워크 실패: {url}") from error


def collector_config(**overrides) -> CollectorConfig:
    values = {
        "source_id": "official",
        "name": "공식 테스트 출처",
        "collector_type": "heritage_portal",
        "base_url": "https://official.example/",
        "publisher": "공식 테스트 기관",
        "trust_grade": "A",
        "policy_url": "https://official.example/policy",
        "robots_url": "https://official.example/robots.txt",
        "allowed_domains": ("official.example",),
        "discovery_urls": ("https://official.example/search",),
        "request_delay_seconds": 0.1,
        "max_retries": 0,
        "max_pages": 1,
        "max_results": 5,
        "collection_status": "allowed",
        "robots_verification": "verified",
    }
    values.update(overrides)
    return CollectorConfig(**values)


def candidate(url: str, *, content_hash: str = "hash", title: str = "목포 자료") -> CollectedCandidate:
    return CollectedCandidate(
        document_id=f"auto-{abs(hash(url))}",
        source_id="official",
        source_url=url,
        title=title,
        publisher="공식 테스트 기관",
        published_date="",
        accessed_date="2026-07-30",
        language="ko",
        license_name="",
        license_url="https://official.example/policy",
        copyright_status="unknown",
        allowed_for_rag=False,
        allowed_for_training=False,
        redistribution_allowed=False,
        trust_grade="A",
        rag_priority_candidate=True,
        review_status="draft",
        raw_path="data/raw/collected/item.html",
        extracted_path="data/extracted/collected/item.txt",
        ocr_path="",
        content_sha256=content_hash,
        notes="테스트용 가상 후보",
    )


def test_collector_only_allows_configured_domains() -> None:
    collector = TestCollector(collector_config(), transport=FakeTransport({}), sleep=lambda _: None)
    assert collector.is_allowed_url("https://official.example/item/1")
    assert collector.is_allowed_url("https://sub.official.example/item/1")
    assert not collector.is_allowed_url("https://evil.example/item/1")
    assert not collector.is_allowed_url("file:///etc/passwd")


def test_unknown_license_is_forbidden_until_review(tmp_path) -> None:
    item = candidate("https://official.example/item/1")
    document = item.to_source_document()
    assert not item.allowed_for_rag
    assert not item.allowed_for_training
    assert not can_use_for_rag(document)
    assert not can_index_for_service(document)


def test_duplicate_url_is_removed(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "collected.jsonl")
    first = candidate("https://official.example/item?id=1&utm_source=test", content_hash="one")
    duplicate = replace(
        first,
        document_id="second",
        source_url="https://official.example/item?id=1",
        content_sha256="two",
    )
    assert registry.add_new((first, duplicate)) == [first]
    assert len(registry.list()) == 1


def test_network_failure_stops_safely(tmp_path) -> None:
    collector = TestCollector(collector_config(), transport=FakeTransport({}), sleep=lambda _: None)
    report = collector.collect(
        "목포", raw_dir=tmp_path / "raw", extracted_dir=tmp_path / "extracted"
    )
    assert report.candidates == ()
    assert report.errors


def test_collected_item_is_draft_and_not_service_indexed(tmp_path) -> None:
    robots = FetchResponse(
        "https://official.example/robots.txt",
        200,
        {"content-type": "text/plain"},
        b"User-agent: *\nAllow: /\n",
    )
    discovery = FetchResponse(
        "https://official.example/search",
        200,
        {"content-type": "text/html; charset=utf-8"},
        '<a href="/item/1">목포 개항 테스트 후보</a>'.encode("utf-8"),
    )
    detail = FetchResponse(
        "https://official.example/item/1",
        200,
        {"content-type": "text/html; charset=utf-8"},
        "<html><title>테스트용 가상 자료</title><body>실제 역사 사실이 아닙니다.</body></html>".encode(
            "utf-8"
        ),
    )
    collector = TestCollector(
        collector_config(),
        transport=FakeTransport(
            {
                "https://official.example/robots.txt": robots,
                "https://official.example/search": discovery,
                "https://official.example/item/1": detail,
            }
        ),
        sleep=lambda _: None,
    )
    report = collector.collect(
        "목포", raw_dir=tmp_path / "raw", extracted_dir=tmp_path / "extracted"
    )
    assert len(report.candidates) == 1
    item = report.candidates[0]
    assert item.review_status == "draft"
    assert item.copyright_status == "unknown"
    assert not can_index_for_service(item.to_source_document())
    assert (tmp_path / "raw" / "official").is_dir()
    assert (tmp_path / "extracted" / "official").is_dir()


def test_seed_sources_include_conservative_audit_fields() -> None:
    seed_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "source_catalog"
        / "seed_sources.json"
    )
    configs = load_collector_configs(seed_path)
    configs_by_id = {config.source_id: config for config in configs}
    assert len(configs) == 7
    assert all(config.robots_url for config in configs)
    assert {
        source_id: config.collection_status
        for source_id, config in configs_by_id.items()
    } == {
        "heritage_portal": "manual_review",
        "history_database": "blocked",
        "mokpo_city": "manual_review",
        "national_archives": "allowed",
        "public_nuri": "manual_review",
        "oak": "allowed",
        "kci": "blocked",
    }
    assert configs_by_id["national_archives"].robots_verification == "verified"
    assert configs_by_id["oak"].robots_verification == "verified"
    assert configs_by_id["history_database"].robots_verification == "verified"
    assert configs_by_id["kci"].robots_verification == "verified"
    assert configs_by_id["heritage_portal"].robots_verification == "unknown"
    assert configs_by_id["mokpo_city"].robots_verification == "unknown"
    assert configs_by_id["public_nuri"].robots_verification == "unknown"
    assert all(config.api_available in {"yes", "no", "unknown"} for config in configs)
    assert all(config.audit_date == "2026-07-30" for config in configs)


def test_unapproved_or_unverified_source_never_calls_network(tmp_path) -> None:
    for status, robots in (
        ("manual_review", "verified"),
        ("blocked", "verified"),
        ("unknown", "verified"),
        ("allowed", "unknown"),
    ):
        collector = TestCollector(
            collector_config(collection_status=status, robots_verification=robots),
            transport=FakeTransport({}),
            sleep=lambda _: None,
        )
        report = collector.collect(
            "목포", raw_dir=tmp_path / "raw", extracted_dir=tmp_path / "extracted"
        )
        assert report.candidates == ()
        assert report.errors and "건너뜀" in report.errors[0]


def test_access_barrier_is_skipped(tmp_path) -> None:
    robots = FetchResponse(
        "https://official.example/robots.txt",
        200,
        {"content-type": "text/plain"},
        b"User-agent: *\nAllow: /\n",
    )
    blocked = FetchResponse(
        "https://official.example/search",
        200,
        {"content-type": "text/html"},
        b'<html><form><input type="password"></form></html>',
    )
    collector = TestCollector(
        collector_config(),
        transport=FakeTransport(
            {
                "https://official.example/robots.txt": robots,
                "https://official.example/search": blocked,
            }
        ),
        sleep=lambda _: None,
    )
    report = collector.collect(
        "목포", raw_dir=tmp_path / "raw", extracted_dir=tmp_path / "extracted"
    )
    assert report.candidates == ()
    assert any("접근 장벽" in error for error in report.errors)


def test_direct_collector_call_cannot_exceed_two_results(tmp_path) -> None:
    robots = FetchResponse(
        "https://official.example/robots.txt",
        200,
        {"content-type": "text/plain"},
        b"User-agent: *\nAllow: /\n",
    )
    links = "".join(
        f'<a href="/item/{index}">목포 개항 테스트 후보 {index}</a>'
        for index in range(4)
    )
    responses = {
        "https://official.example/robots.txt": robots,
        "https://official.example/search": FetchResponse(
            "https://official.example/search",
            200,
            {"content-type": "text/html"},
            links.encode("utf-8"),
        ),
    }
    for index in range(4):
        url = f"https://official.example/item/{index}"
        responses[url] = FetchResponse(
            url,
            200,
            {"content-type": "text/html"},
            f"<title>테스트용 가상 자료 {index}</title>".encode("utf-8"),
        )
    collector = TestCollector(
        collector_config(max_results=100),
        transport=FakeTransport(responses),
        sleep=lambda _: None,
    )
    report = collector.collect(
        "목포", raw_dir=tmp_path / "raw", extracted_dir=tmp_path / "extracted"
    )
    assert len(report.candidates) == MAX_RESULTS_PER_SOURCE == 2


class StubCollector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config

    def collect(self, query: str, *, raw_dir: Path, extracted_dir: Path) -> CollectionReport:
        items = tuple(
            replace(
                candidate(
                    f"https://official.example/{self.config.source_id}/item/{index}",
                    content_hash=f"{self.config.source_id}-{index}",
                    title=f"{self.config.source_id} 테스트 자료 {index}",
                ),
                document_id=f"{self.config.source_id}-{index}",
                source_id=self.config.source_id,
                allowed_for_rag=True,
                allowed_for_training=True,
                review_status="reviewed",
            )
            for index in range(5)
        )
        return CollectionReport(items)


def test_pilot_enforces_total_ten_and_two_per_source(tmp_path) -> None:
    configs = [
        collector_config(
            source_id=f"source-{index}",
            name=f"공식 출처 {index}",
            discovery_urls=(f"https://official.example/source/{index}",),
        )
        for index in range(6)
    ]
    registry = CandidateRegistry(tmp_path / "collected.jsonl")
    result = run_pilot(
        configs,
        query="목포",
        raw_dir=tmp_path / "raw",
        extracted_dir=tmp_path / "extracted",
        registry=registry,
        collector_factory=StubCollector,
    )
    assert len(result.candidates) == MAX_TOTAL_RESULTS == 10
    assert all(count <= MAX_RESULTS_PER_SOURCE == 2 for count in result.per_source.values())
    assert len({item.source_id for item in result.candidates}) == 5
    assert all(item.review_status == "draft" for item in result.candidates)
    assert all(not item.allowed_for_rag for item in result.candidates)
    assert all(not item.allowed_for_training for item in result.candidates)


def test_pilot_plan_explains_scheduled_and_skipped_urls() -> None:
    allowed = collector_config(source_id="allowed")
    skipped = collector_config(
        source_id="skipped",
        collection_status="manual_review",
        robots_verification="unknown",
    )
    plan = build_pilot_plan([allowed, skipped])
    assert plan[0].eligible
    assert "출처별 최대 2건" in plan[0].reason
    assert not plan[1].eligible
    assert "manual_review" in plan[1].reason


def test_unknown_license_safety_overrides_usage_and_review_state() -> None:
    unsafe = replace(
        candidate("https://official.example/item/unsafe"),
        allowed_for_rag=True,
        allowed_for_training=True,
        review_status="reviewed",
    )
    safe = enforce_candidate_safety(unsafe)
    assert safe.review_status == "draft"
    assert not safe.allowed_for_rag
    assert not safe.allowed_for_training


def test_cli_defaults_to_dry_run_without_network(tmp_path, monkeypatch, capsys) -> None:
    seed = tmp_path / "seed.json"
    config = collector_config()
    seed.write_text(
        json.dumps({"sources": [asdict(config)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--seed", str(seed), "--output", str(tmp_path / "output.jsonl")],
    )
    collector_cli_main()
    output = capsys.readouterr().out
    assert "수집 예정" in output
    assert "dry-run 완료" in output
    assert not (tmp_path / "output.jsonl").exists()
