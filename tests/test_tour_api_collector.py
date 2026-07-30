import json
import sys

import pytest

from history_chatbot.collectors.cli import main as collector_cli_main
from history_chatbot.collectors.tour_api import (
    MAX_RESULTS_PER_KEYWORD,
    MAX_TOTAL_RESULTS,
    TourApiCollector,
    TourApiError,
)
from history_chatbot.ingestion.source_registry import SourceRegistry


class FakeTourTransport:
    def __init__(self, search_items, details) -> None:
        self.search_items = search_items
        self.details = details
        self.calls = []

    def get_json(self, endpoint, params, *, timeout):
        self.calls.append((endpoint, dict(params)))
        if endpoint == "searchKeyword2":
            items = self.search_items.get(params["keyword"], [])
        else:
            detail = self.details.get(params["contentId"])
            items = [detail] if detail else []
        return {"response": {"body": {"items": {"item": items}}}}


def test_missing_key_exits_without_network(monkeypatch, capsys) -> None:
    monkeypatch.delenv("TOUR_API_SERVICE_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["collector", "tour-api", "dry-run"])
    with pytest.raises(SystemExit) as raised:
        collector_cli_main()
    assert "TOUR_API_SERVICE_KEY" in str(raised.value)
    assert "serviceKey=" not in str(raised.value)


def test_search_and_detail_are_separate_and_overview_required() -> None:
    transport = FakeTourTransport(
        {"목포": [{"contentid": "1", "title": "자료 1"}, {"contentid": "2", "title": "자료 2"}]},
        {
            "1": {"contentid": "1", "title": "자료 1", "overview": "역사 설명"},
            "2": {"contentid": "2", "title": "자료 2", "overview": ""},
        },
    )
    collector = TourApiCollector("secret-value", transport=transport)
    items = collector.dry_run(("목포",))
    assert [item.content_id for item in items] == ["1"]
    assert [call[0] for call in transport.calls] == [
        "searchKeyword2",
        "detailCommon2",
        "detailCommon2",
    ]


def test_limits_and_duplicate_content_ids_are_enforced() -> None:
    keywords = tuple(f"검색어-{index}" for index in range(6))
    search_items = {
        keyword: [
            {"contentid": f"{keyword}-{index}", "title": "자료"}
            for index in range(8)
        ]
        for keyword in keywords
    }
    details = {
        item["contentid"]: {
            "contentid": item["contentid"],
            "title": "자료",
            "overview": "본문",
        }
        for items in search_items.values()
        for item in items
    }
    collector = TourApiCollector(
        "secret-value", transport=FakeTourTransport(search_items, details)
    )
    results = collector.dry_run(keywords)
    assert len(results) == MAX_TOTAL_RESULTS == 20
    assert all(
        sum(item.keyword == keyword for item in results) <= MAX_RESULTS_PER_KEYWORD == 5
        for keyword in keywords
    )


def test_collection_uses_contentid_and_conservative_permissions(tmp_path) -> None:
    transport = FakeTourTransport(
        {"목포": [{"contentid": "123", "title": "목포 자료"}]},
        {
            "123": {
                "contentid": "123",
                "title": "목포 자료",
                "overview": "테스트용 설명 본문",
                "homepage": "https://korean.visitkorea.or.kr/example",
            }
        },
    )
    collector = TourApiCollector("secret-value", transport=transport)
    items = collector.dry_run(("목포",))
    catalog = tmp_path / "collected.jsonl"
    manifest = tmp_path / "sources.jsonl"
    result = collector.collect(
        raw_dir=tmp_path / "raw",
        extracted_dir=tmp_path / "extracted",
        catalog_path=catalog,
        manifest_path=manifest,
        prepared_items=items,
    )
    item = result.candidates[0]
    assert item.document_id == "tour-api-123"
    assert item.review_status == "draft"
    assert item.copyright_status == "unknown"
    assert not item.allowed_for_rag
    assert not item.allowed_for_training
    assert (tmp_path / "raw" / "tour-api-123.json").exists()
    assert (tmp_path / "extracted" / "tour-api-123.txt").exists()
    document = SourceRegistry(manifest).get("tour-api-123")
    assert document.review_status.value == "draft"
    assert not document.allowed_for_rag
    assert "secret-value" not in catalog.read_text(encoding="utf-8")
    assert "secret-value" not in manifest.read_text(encoding="utf-8")


def test_dry_run_cli_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    fake = FakeTourTransport(
        {"목포": [{"contentid": "1", "title": "자료"}]},
        {"1": {"contentid": "1", "title": "자료", "overview": "본문"}},
    )
    collector = TourApiCollector("secret", transport=fake)
    monkeypatch.setattr(
        TourApiCollector, "from_environment", classmethod(lambda cls: collector)
    )
    monkeypatch.setattr(
        collector, "dry_run", lambda keywords=(): collector.__class__(
            "secret", transport=fake
        ).dry_run(("목포",))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "tour-api",
            "dry-run",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
        ],
    )
    collector_cli_main()
    assert "dry-run 완료" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "manifest.jsonl").exists()
