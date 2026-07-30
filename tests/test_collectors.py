from dataclasses import replace
from pathlib import Path

from history_chatbot.collectors.base import (
    BaseCollector,
    CollectedCandidate,
    CollectionError,
    CollectorConfig,
    FetchResponse,
    load_collector_configs,
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
    assert len(configs) == 7
    assert all(config.robots_url for config in configs)
    assert all(config.collection_status == "manual_review" for config in configs)
    assert all(config.robots_verification == "unknown" for config in configs)
    assert all(config.api_available in {"yes", "no", "unknown"} for config in configs)
    assert all(config.audit_date == "2026-07-30" for config in configs)
