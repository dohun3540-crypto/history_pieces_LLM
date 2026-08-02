"""Network-free tests; every HTML sample below is fictional test fixture content."""
import ast
import hashlib
import json
from pathlib import Path, PureWindowsPath

import pytest

from history_chatbot.collectors.hackathon_metadata import (
    CANDIDATES,
    FIXED_RIGHTS,
    MAX_EXCERPT_LENGTH,
    CollectionError,
    FetchResponse,
    append_transaction,
    build_record,
    collect,
    dry_run,
    extract_page,
    validate_response,
    validate_url,
)


FICTIONAL_HTML = """<!doctype html><html><head><title>테스트용 기록 상세</title></head>
<body><header>테스트 메뉴 목포 메뉴</header><main>
<dl><dt>관리번호</dt><dd>CJA-TEST</dd><dt>문서번호</dt><dd>DOC-TEST</dd>
<dt>생산연도</dt><dd>1897</dd><dt>생산기관</dt><dd>테스트 기관</dd>
<dt>공개구분</dt><dd>공개</dd><dt>문서유형</dt><dd>문서</dd></dl>
<p>이 문장은 실제 사료가 아닌 fictional test fixture이며 목포 조계 관련 추출 동작만 검증합니다.</p>
</main><script>목포 비밀 스크립트</script><footer>목포 테스트 푸터</footer></body></html>"""


def response(body=FICTIONAL_HTML, content_type="text/html; charset=utf-8", final_url=None):
    return FetchResponse(
        final_url=final_url or CANDIDATES["archives-cja0002271-0027148187"]["source_url"],
        content_type=content_type,
        body=body.encode("utf-8"),
    )


def write_manifest(path, records=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def run_collect(tmp_path, candidate_ids=None, fetcher=None, replace_file=None):
    manifest = tmp_path / "data/provisional_hackathon/manifests/sources.jsonl"
    write_manifest(manifest, [{"source_id": "existing", "source_title": "기존 기록", "source_url": "https://example.invalid/existing"}])
    kwargs = {}
    if replace_file is not None:
        kwargs["replace_file"] = replace_file
    records = collect(
        candidate_ids or ["archives-cja0002271-0027148187"], manifest,
        tmp_path / "data/provisional_hackathon/extracted", 2, 1.0, 15, 1048576,
        fetcher=fetcher or (lambda url, timeout, size: response()),
        repository_root=tmp_path, collected_at="2026-08-02T00:00:00Z", **kwargs
    )
    return manifest, records


def test_dry_run_never_calls_network_or_creates_files(tmp_path):
    manifest = tmp_path / "missing/sources.jsonl"
    extracted = tmp_path / "missing/extracted"
    result = dry_run(["archives-cja0002271-0027148187"], manifest, extracted, 2, 1.2, 15, 1048576, tmp_path)
    assert result["maximum_requests_on_execute"] == 1
    assert not manifest.exists()
    assert not extracted.exists()


@pytest.mark.parametrize("url", ["http://theme.archives.go.kr/x", "https://evil.example/x"])
def test_url_policy_rejects_http_and_other_hosts(url):
    with pytest.raises(CollectionError):
        validate_url(url)


def test_redirect_to_other_host_is_rejected():
    with pytest.raises(CollectionError):
        validate_response(response(final_url="https://evil.example/page"), 1048576)


@pytest.mark.parametrize("content_type", ["application/pdf", "image/png", "application/zip"])
def test_non_html_content_types_are_rejected(content_type):
    with pytest.raises(CollectionError):
        validate_response(response(content_type=content_type), 1048576)


def test_oversized_and_empty_responses_are_rejected():
    with pytest.raises(CollectionError):
        validate_response(response(body="x" * 11), 10)
    with pytest.raises(CollectionError):
        validate_response(response(body=""), 10)


def test_missing_title_and_missing_mokpo_excerpt_are_rejected():
    with pytest.raises(CollectionError):
        extract_page(b"<html><body><p>fictional test fixture Mokpo</p></body></html>", CANDIDATES["archives-cja0002271-0027148187"])
    with pytest.raises(CollectionError):
        extract_page("<html><title>테스트</title><body>관련 없는 fictional test fixture 설명입니다.</body></html>".encode("utf-8"), CANDIDATES["archives-cja0002271-0027148187"])


def test_excerpt_is_bounded_and_removes_chrome_and_script():
    long_html = FICTIONAL_HTML.replace("추출 동작만 검증합니다.", "목포 관련 테스트 문장입니다. " * 80)
    item = extract_page(long_html.encode("utf-8"), CANDIDATES["archives-cja0002271-0027148187"])
    assert len(item["excerpt"]) <= MAX_EXCERPT_LENGTH
    assert "비밀 스크립트" not in item["excerpt"]
    assert "테스트 푸터" not in item["excerpt"]
    assert "테스트 메뉴" not in item["excerpt"]


@pytest.mark.parametrize("field,value", [
    ("source_id", "archives-cja0002271-0027148187"),
    ("source_url", CANDIDATES["archives-cja0002271-0027148187"]["source_url"]),
    ("canonical_url", CANDIDATES["archives-cja0002271-0027148187"]["canonical_url"]),
    ("source_title", CANDIDATES["archives-cja0002271-0027148187"]["title"]),
])
def test_exact_identifier_url_canonical_and_title_duplicates_block(tmp_path, field, value):
    manifest = tmp_path / "data/manifests/sources.jsonl"
    write_manifest(manifest, [{field: value}])
    with pytest.raises(CollectionError, match="중복 또는 유사"):
        collect(["archives-cja0002271-0027148187"], manifest, tmp_path / "out", 1, 1, 2, 10000,
                fetcher=lambda url, timeout, size: response(), repository_root=tmp_path)


def test_similar_title_blocks_by_default(tmp_path):
    manifest = tmp_path / "data/manifests/sources.jsonl"
    title = CANDIDATES["archives-cja0002271-0027148187"]["title"] + " 자료"
    write_manifest(manifest, [{"source_title": title}])
    with pytest.raises(CollectionError, match="similar title"):
        collect(["archives-cja0002271-0027148187"], manifest, tmp_path / "out", 1, 1, 2, 10000,
                fetcher=lambda url, timeout, size: response(), repository_root=tmp_path)


def test_excerpt_hash_duplicate_blocks_and_leaves_no_file(tmp_path):
    page = extract_page(FICTIONAL_HTML.encode("utf-8"), CANDIDATES["archives-cja0002271-0027148187"])
    manifest = tmp_path / "data/manifests/sources.jsonl"
    write_manifest(manifest, [{"excerpt_sha256": page["excerpt_sha256"]}])
    out = tmp_path / "out"
    with pytest.raises(CollectionError, match="hash exact"):
        collect(["archives-cja0002271-0027148187"], manifest, out, 1, 1, 2, 10000,
                fetcher=lambda url, timeout, size: response(), repository_root=tmp_path)
    assert not out.exists()


def test_parent_child_relationship_and_rights_are_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr("history_chatbot.collectors.hackathon_metadata.time.sleep", lambda delay: None)
    second = """<html><head><title>별도 테스트 기록철</title></head><body><main>
    <p>fictional test fixture: 목포 개항 뒤 형성된 거류지를 설명하는 완전히 별도의 검증 문장입니다.</p>
    </main></body></html>"""
    bodies = [FICTIONAL_HTML, second]
    calls = []
    def fetch(url, timeout, size):
        calls.append(url)
        return response(body=bodies[len(calls) - 1], final_url=url)
    manifest, records = run_collect(tmp_path, list(CANDIDATES), fetch)
    assert len(calls) == 2
    assert records[0]["related_document_id"] == records[1]["document_id"]
    assert records[1]["related_document_id"] == records[0]["document_id"]
    for record in records:
        for key, value in FIXED_RIGHTS.items():
            assert record[key] == value


def test_existing_manifest_bytes_unchanged_and_only_rows_appended(tmp_path):
    manifest = tmp_path / "manifest/sources.jsonl"
    original = b'{"z": 1, "a": 2}\n'
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(original)
    output = tmp_path / "extracted/new.txt"
    append_transaction(manifest, [(output, "test")], [{"new": True}])
    assert manifest.read_bytes().startswith(original)
    assert manifest.read_text(encoding="utf-8").splitlines()[0] == original.decode().strip()
    assert json.loads(manifest.read_text(encoding="utf-8").splitlines()[1]) == {"new": True}


def test_manifest_replace_failure_rolls_back_extracted(tmp_path):
    manifest = tmp_path / "manifest/sources.jsonl"
    write_manifest(manifest, [{"old": True}])
    original_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "extracted/new.txt"
    calls = []
    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("fictional manifest failure")
        Path(source).replace(Path(target))
    with pytest.raises(OSError):
        append_transaction(manifest, [(output, "test")], [{"new": True}], replace_file=fail_second)
    assert not output.exists()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == original_hash


def test_second_candidate_failure_leaves_no_partial_state(tmp_path, monkeypatch):
    monkeypatch.setattr("history_chatbot.collectors.hackathon_metadata.time.sleep", lambda delay: None)
    manifest = tmp_path / "data/manifests/sources.jsonl"
    write_manifest(manifest, [{"source_id": "old"}])
    original = manifest.read_bytes()
    count = [0]
    def fetch(url, timeout, size):
        count[0] += 1
        return response(final_url=url) if count[0] == 1 else response(body="", final_url=url)
    with pytest.raises(CollectionError):
        collect(list(CANDIDATES), manifest, tmp_path / "out", 2, 1, 2, 10000,
                fetcher=fetch, repository_root=tmp_path)
    assert manifest.read_bytes() == original
    assert not (tmp_path / "out").exists()


def test_extracted_output_is_minimal_and_no_raw_is_created(tmp_path):
    manifest, records = run_collect(tmp_path)
    output = tmp_path / records[0]["extracted_text_path"]
    text = output.read_text(encoding="utf-8")
    assert hashlib.sha256(output.read_bytes()).hexdigest() == records[0]["extracted_sha256"]
    assert "제목:" in text and "기관:" in text and "상세 URL:" in text
    assert "<html" not in text and "collection_metadata" not in text
    assert not (tmp_path / "data/provisional_hackathon/raw").exists()


def test_windows_paths_are_joined_without_platform_assumptions():
    path = PureWindowsPath(r"C:\repo\data\extracted") / ("doc" + ".txt")
    assert str(path).endswith(r"data\extracted\doc.txt")


def test_new_python_files_parse_as_python_38():
    root = Path(__file__).resolve().parents[1]
    for relative in ["src/history_chatbot/collectors/hackathon_metadata.py", "scripts/collect_hackathon_metadata.py"]:
        ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative, feature_version=(3, 8))


def test_registry_has_required_candidate_fields():
    required = {"document_id", "title", "institution", "source_url", "canonical_url", "topic_tags",
                "related_document_id", "raw_source_status", "robots_status", "expected_host", "document_type"}
    assert len(CANDIDATES) == 2
    assert all(required <= set(candidate) for candidate in CANDIDATES.values())
