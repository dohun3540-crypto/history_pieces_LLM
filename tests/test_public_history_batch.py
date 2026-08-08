"""All network responses in this module are fictional local fixtures."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from history_chatbot.collectors.public_history_batch import (
    ADAPTERS, FIXED_RIGHTS, BatchCandidate, BatchError, BatchPipeline, BatchResponse,
    DuplicateIndex, PublicSourceAdapter, RequestController, SOURCE_SPECS, SourceSpec,
    atomic_write, extract_payload, is_relevant, quality_decision, validate_media_type,
    validate_public_url,
)
from scripts.collect_public_history_batch import main


def candidate(number=1, source_id="national_archives", document_type="descriptive_document", **values):
    item = BatchCandidate(
        document_id="batch-doc-%d" % number,
        source_id=source_id,
        title="목포 개항 역사 기록 %d" % number,
        institution="테스트 공공기관",
        source_url="https://www.archives.go.kr/detail/%d" % number,
        canonical_url="https://www.archives.go.kr/detail/%d" % number,
        document_type=document_type,
        topic_tags=["목포 개항"],
        portal_name="fictional fixture portal",
        original_institution="테스트 공공기관",
    )
    for key, value in values.items():
        setattr(item, key, value)
    return item


def descriptive_text(number=1, length=340):
    prefix = "fictional test fixture %d: 목포 개항과 항만의 역사 변화를 설명하는 검증 문장입니다. " % number
    return (prefix + ("별도의 역사 설명 내용입니다. " * 30))[:length]


def html_response(item, text=None, final_url=None, content_type="text/html; charset=utf-8"):
    body = "<html><head><title>%s</title></head><body><header>메뉴</header><main><p>%s</p></main><script>bad</script><footer>footer</footer></body></html>" % (
        item.title, text if text is not None else descriptive_text()
    )
    return BatchResponse(final_url or item.source_url, 200, content_type, body.encode("utf-8"))


class FakeTransport:
    def __init__(self, responses, calls=None):
        self.responses = responses
        self.calls = calls if calls is not None else []

    def get(self, url, timeout, max_bytes):
        self.calls.append(url)
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return value


def controller(responses, calls=None, sleeps=None, clock=None, max_requests=75):
    shared_calls = calls if calls is not None else []
    return RequestController(
        max_requests, 1.2, lambda hosts: FakeTransport(responses, shared_calls),
        sleep=(sleeps.append if sleeps is not None else (lambda value: None)),
        clock=clock or (lambda: 0.0), max_retries=0,
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def paths(tmp_path):
    return {
        "manifest": tmp_path / "data/manifests/sources.jsonl",
        "catalog": tmp_path / "data/catalog/batch.jsonl",
        "extracted_dir": tmp_path / "data/extracted",
        "report_json": tmp_path / "reports/batch.json",
        "report_md": tmp_path / "reports/batch.md",
    }


def limits(**values):
    result = {"max_accepted": 10, "max_per_source": 2, "max_requests": 75, "delay_seconds": 1.2}
    result.update(values)
    return result


def execute(tmp_path, candidates, responses, custom_limits=None, replace_file=None):
    target = paths(tmp_path)
    write_jsonl(target["manifest"], [{"document_id": "existing", "title": "기존 자료", "source_url": "https://example.invalid/existing"}])
    write_jsonl(target["catalog"], [item.__dict__ for item in candidates])
    pipeline = BatchPipeline(ADAPTERS, controller(responses))
    kwargs = {}
    if replace_file is not None:
        kwargs["replace_file"] = replace_file
    report = pipeline.execute("batch-test", list(ADAPTERS), target["catalog"], target["manifest"],
                              target["extracted_dir"], target["report_json"], target["report_md"],
                              15, 1048576, custom_limits or limits(), "2026-08-02T00:00:00Z", **kwargs)
    return target, report


def test_dry_run_has_no_network_and_creates_no_files(tmp_path, monkeypatch):
    target = paths(tmp_path)
    monkeypatch.setattr("history_chatbot.collectors.public_history_batch.UrllibBatchTransport.get",
                        lambda *args: (_ for _ in ()).throw(AssertionError("network called")))
    result = BatchPipeline(ADAPTERS).dry_run(list(ADAPTERS), target, limits())
    assert result["network"] is False and result["files_created"] is False
    assert not any(value.exists() for value in target.values())


def test_cli_dry_run_has_no_files(tmp_path, capsys):
    target = paths(tmp_path)
    args = ["--batch-id", "batch-001", "--manifest", str(target["manifest"]), "--catalog", str(target["catalog"]),
            "--extracted-dir", str(target["extracted_dir"]), "--report-json", str(target["report_json"]),
            "--report-md", str(target["report_md"]), "--source", "national_archives", "--dry-run"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network"] is False
    assert result["limits"]["max_accepted"] == 10
    assert result["limits"]["max_per_source"] == 2
    assert not tmp_path.joinpath("data").exists()


@pytest.mark.parametrize("value,key", [(11, "max_accepted"), (3, "max_per_source"), (0, "max_requests")])
def test_batch_limits_are_bounded(value, key):
    with pytest.raises(BatchError):
        BatchPipeline(ADAPTERS).dry_run([], {}, limits(**{key: value}))


def test_request_limit_is_enforced():
    item = candidate()
    responses = {item.source_url: html_response(item)}
    control = controller(responses, max_requests=1)
    control.get(item.source_url, SOURCE_SPECS["national_archives"], 2, 10000)
    with pytest.raises(BatchError, match="maximum request"):
        control.get(item.source_url, SOURCE_SPECS["national_archives"], 2, 10000)


def test_per_host_delay_is_applied():
    item = candidate()
    sleeps = []
    ticks = iter([0.0, 0.0, 0.0, 0.0])
    control = controller({item.source_url: html_response(item)}, sleeps=sleeps, clock=lambda: next(ticks))
    control.get(item.source_url, SOURCE_SPECS["national_archives"], 2, 10000)
    control.get(item.source_url, SOURCE_SPECS["national_archives"], 2, 10000)
    assert sleeps == [1.2]


@pytest.mark.parametrize("url", ["http://www.archives.go.kr/x", "https://evil.example/x"])
def test_https_and_host_are_enforced(url):
    with pytest.raises(BatchError):
        validate_public_url(url, ("www.archives.go.kr",))


def test_redirect_host_change_is_rejected():
    item = candidate()
    control = controller({item.source_url: html_response(item, final_url="https://evil.example/x")})
    with pytest.raises(BatchError):
        control.get(item.source_url, SOURCE_SPECS["national_archives"], 2, 10000)


@pytest.mark.parametrize("media", ["text/html", "application/json", "application/xml", "text/xml", "text/csv"])
def test_supported_text_media_types(media):
    assert validate_media_type(media, ()) == media


@pytest.mark.parametrize("media", ["application/pdf", "image/jpeg", "application/zip"])
def test_binary_media_rejected_unless_explicit(media):
    with pytest.raises(BatchError):
        validate_media_type(media, ())
    assert validate_media_type(media, (media,)) == media


def test_html_json_xml_csv_extraction_removes_html_chrome():
    item = candidate()
    html_doc = extract_payload(html_response(item), item)
    assert "메뉴" not in html_doc.text and "bad" not in html_doc.text and "footer" not in html_doc.text
    fixtures = [
        BatchResponse(item.source_url, 200, "application/json", json.dumps({"text": descriptive_text()}).encode()),
        BatchResponse(item.source_url, 200, "application/xml", ("<root><text>%s</text></root>" % descriptive_text()).encode()),
        BatchResponse(item.source_url, 200, "text/csv", ("title,text\nfixture,%s\n" % descriptive_text()).encode()),
    ]
    assert all("목포" in extract_payload(value, item).text for value in fixtures)


@pytest.mark.parametrize("record_field,candidate_field", [
    ("document_id", "document_id"), ("source_url", "source_url"), ("canonical_url", "canonical_url")
])
def test_exact_id_and_url_duplicates_block(record_field, candidate_field):
    item = candidate()
    index = DuplicateIndex([{record_field: getattr(item, candidate_field)}])
    exact, unused = index.check(item, "body", "extracted")
    assert exact


def test_exact_body_and_extracted_hashes_block():
    item = candidate()
    index = DuplicateIndex([{"content_hash": "body", "extracted_sha256": "file"}])
    exact, unused = index.check(item, "body", "file")
    assert exact == ["body_hash", "extracted_hash"]


def test_similar_title_is_warning_not_duplicate():
    item = candidate()
    index = DuplicateIndex([{"title": item.title + " 설명"}])
    exact, warnings = index.check(item, "new-body", "new-file")
    assert exact == [] and any(value.startswith("similar_title") for value in warnings)


def test_body_similarity_is_warning():
    item = candidate()
    index = DuplicateIndex([])
    index.add_body(descriptive_text())
    exact, warnings = index.check(item, "new", "new2", descriptive_text() + " 추가")
    assert exact == [] and any(value.startswith("similar_body") for value in warnings)


@pytest.mark.parametrize("title,text,expected", [
    ("목포항 개발", "역사 설명", True),
    ("전라남도 보고서", "목포 개항과 항만 역사에서 목포가 핵심 도시로 다뤄진다.", True),
    ("전라남도 목록", "참고문헌에 목포가 있다.", False),
])
def test_mokpo_relevance(title, text, expected):
    assert is_relevant(title, text)[0] is expected


@pytest.mark.parametrize("text,decision", [
    ("", "rejected_empty"),
    ("404 Not Found 목포 " + "x" * 100, "rejected_quality"),
    ("통합검색 검색결과 목포 개항 " + "x" * 100, "rejected_quality"),
    ("로그인 필요 목포 개항 " + "x" * 100, "rejected_access_barrier"),
])
def test_empty_error_search_and_access_pages_rejected(text, decision):
    assert quality_decision(candidate(), text).decision == decision


def test_metadata_50_character_exception_and_descriptive_quality():
    metadata = candidate(document_type="metadata_document")
    text = "목포 개항 기록물 metadata fixture 설명으로서 식별과 연도를 포함합니다. " * 2
    assert quality_decision(metadata, text).decision == "accepted_metadata_only"
    assert quality_decision(candidate(), text).decision == "needs_review"
    assert quality_decision(candidate(), descriptive_text()).decision == "accepted_hackathon"


def test_execute_rights_hackathon_metadata_lf_and_hash(tmp_path):
    item = candidate()
    target, report = execute(tmp_path, [item], {item.source_url: html_response(item)})
    rows = [json.loads(line) for line in target["manifest"].read_text(encoding="utf-8").splitlines()]
    record = rows[-1]
    for key, value in FIXED_RIGHTS.items():
        assert record[key] == value
    assert record["review_status"] == "draft"
    assert record["allowed_for_rag"] is False
    assert record["source_id"] == item.source_id
    assert record["publisher"] == item.institution
    assert record["source_name"] == item.portal_name
    assert record["license_name"] == ""
    assert record["collection_metadata"]["allowed_for_hackathon_rag"] is False
    assert record["collection_metadata"]["allowed_for_public_production"] is False
    output = target["extracted_dir"] / (item.document_id + ".txt")
    assert b"\r\n" not in output.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == record["extracted_sha256"]
    assert report["counts"]["stored"] == 1


def test_existing_manifest_bytes_are_preserved(tmp_path):
    item = candidate()
    target = paths(tmp_path)
    original = b'{"z": 1, "a": 2}\n'
    target["manifest"].parent.mkdir(parents=True)
    target["manifest"].write_bytes(original)
    write_jsonl(target["catalog"], [item.__dict__])
    pipeline = BatchPipeline(ADAPTERS, controller({item.source_url: html_response(item)}))
    pipeline.execute("batch", [item.source_id], target["catalog"], target["manifest"], target["extracted_dir"],
                     target["report_json"], target["report_md"], 2, 10000, limits(), "2026-08-02")
    assert target["manifest"].read_bytes().startswith(original)


def test_per_source_limit_is_enforced(tmp_path):
    items = [candidate(number) for number in range(1, 4)]
    responses = {item.source_url: html_response(item, descriptive_text(number)) for number, item in enumerate(items, 1)}
    target, report = execute(tmp_path, items, responses, limits(max_per_source=2))
    assert report["counts"]["stored"] == 2


def test_total_pilot_limit_is_enforced(tmp_path):
    sources = [
        ("national_archives", "https://www.archives.go.kr/detail"),
        ("national_archives_html", "https://www.archives.go.kr/next/detail"),
        ("heritage_portal", "https://www.heritage.go.kr/detail"),
        ("mokpo_official", "https://www.mokpo.go.kr/detail"),
        ("data_portal", "https://www.data.go.kr/detail"),
    ]
    items = []
    responses = {}
    number = 1
    for source_id, base_url in sources:
        for unused in range(2):
            item = candidate(number, source_id=source_id)
            item.source_url = "%s/%d" % (base_url, number)
            item.canonical_url = item.source_url
            items.append(item)
            responses[item.source_url] = html_response(item, descriptive_text(number))
            number += 1
    overflow = candidate(number, source_id="heritage_wfs")
    overflow.source_url = "https://unused.invalid/detail/%d" % number
    overflow.canonical_url = overflow.source_url
    items.append(overflow)

    target, report = execute(tmp_path, items, responses)

    assert report["counts"]["stored"] == 10
    assert len(list(target["extracted_dir"].glob("*.txt"))) == 10


def test_disallowed_source_candidate_is_skipped(tmp_path):
    item = candidate(source_id="unlisted_source")
    target, report = execute(tmp_path, [item], {})
    assert report["counts"]["stored"] == 0
    assert not target["extracted_dir"].exists()


def test_execute_duplicate_is_skipped(tmp_path):
    item = candidate()
    target = paths(tmp_path)
    write_jsonl(target["manifest"], [{
        "document_id": item.document_id,
        "source_url": item.source_url,
        "canonical_url": item.canonical_url,
    }])
    write_jsonl(target["catalog"], [item.__dict__])
    pipeline = BatchPipeline(ADAPTERS, controller({item.source_url: html_response(item)}))

    report = pipeline.execute(
        "batch", [item.source_id], target["catalog"], target["manifest"],
        target["extracted_dir"], target["report_json"], target["report_md"],
        2, 10000, limits(), "2026-08-02",
    )

    assert report["counts"]["stored"] == 0
    assert report["counts"]["rejected_duplicate"] == 1
    assert not target["extracted_dir"].exists()


def test_partial_candidate_failure_is_reported_but_success_is_saved(tmp_path):
    good, bad = candidate(1), candidate(2)
    target, report = execute(tmp_path, [good, bad], {
        good.source_url: html_response(good),
        bad.source_url: BatchResponse(bad.source_url, 200, "text/html", b"<html><head><title>fixture</title></head><body></body></html>"),
    })
    assert report["counts"]["stored"] == 1
    assert report["counts"]["rejected_empty"] == 1
    assert (target["extracted_dir"] / (good.document_id + ".txt")).exists()
    assert not (target["extracted_dir"] / (bad.document_id + ".txt")).exists()


def test_failed_request_is_skipped_but_later_candidate_is_saved(tmp_path):
    failed, good = candidate(1), candidate(2)
    target, report = execute(tmp_path, [failed, good], {
        failed.source_url: BatchError("fictional request failure"),
        good.source_url: html_response(good, descriptive_text(2)),
    })
    assert report["counts"]["stored"] == 1
    assert report["counts"]["rejected_quality"] == 1
    assert not (target["extracted_dir"] / (failed.document_id + ".txt")).exists()
    assert (target["extracted_dir"] / (good.document_id + ".txt")).exists()


def test_batch_commit_failure_rolls_back_manifest_and_outputs(tmp_path):
    item = candidate()
    target = paths(tmp_path)
    write_jsonl(target["manifest"], [{"document_id": "existing"}])
    original = target["manifest"].read_bytes()
    write_jsonl(target["catalog"], [item.__dict__])
    calls = []
    def fail_late(source, destination):
        calls.append(destination)
        if len(calls) == 3:
            raise OSError("fictional transaction failure")
        Path(source).replace(Path(destination))
    pipeline = BatchPipeline(ADAPTERS, controller({item.source_url: html_response(item)}))
    with pytest.raises(OSError):
        pipeline.execute("batch", [item.source_id], target["catalog"], target["manifest"], target["extracted_dir"],
                         target["report_json"], target["report_md"], 2, 10000, limits(), "2026-08-02",
                         replace_file=fail_late)
    assert target["manifest"].read_bytes() == original
    assert not (target["extracted_dir"] / (item.document_id + ".txt")).exists()


def test_explicit_access_policy_is_rejected_without_request(tmp_path):
    item = candidate(discovery_metadata={"access_policy": "blocked"})
    target, report = execute(tmp_path, [item], {})
    assert report["counts"]["rejected_access_policy"] == 1


def test_missing_tour_api_key_stops_adapter():
    with pytest.raises(BatchError, match="TOUR_API_SERVICE_KEY"):
        ADAPTERS["tour_api"].discovery_urls(["목포"], {})


def test_tour_api_json_discovery_keeps_key_out_of_candidate():
    adapter = ADAPTERS["tour_api"]
    payload = {"response": {"body": {"items": {"item": {
        "contentid": "123", "title": "목포 근대역사관", "overview": "목포 근대 역사 설명"
    }}}}}
    response = BatchResponse("https://apis.data.go.kr/search", 200, "application/json",
                             json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    found = adapter.discover(response, response.final_url)
    assert found[0].document_id == "tour-api-123"
    assert "serviceKey" not in found[0].source_url
    assert "secret" in adapter.detail_url(found[0], {
        "TOUR_API_SERVICE_KEY": "secret", "TOUR_API_SERVICE_KEY_FORMAT": "decoding",
    })


def test_python_38_ast_compatibility():
    root = Path(__file__).resolve().parents[1]
    for name in ("src/history_chatbot/collectors/public_history_batch.py", "scripts/collect_public_history_batch.py"):
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 8))
