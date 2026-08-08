"""Offline fixtures for the National Archives API and heritage WFS adapters."""
import json
import urllib.error

import pytest

from history_chatbot.collectors.public_history_batch import (
    ADAPTERS,
    ApiResponseError,
    BatchError,
    BatchPipeline,
    BatchResponse,
    HeritageWfsAdapter,
    RequestController,
    RequestDiagnosticError,
    SOURCE_SPECS,
)
from scripts.collect_public_history_batch import main


ARCHIVES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<response><header><resultCode>00</resultCode><resultMsg>NORMAL</resultMsg></header><body><items>
  <item><recordId>CJA-1</recordId><title>목포부 기록철</title><recordLevel>철</recordLevel>
    <producer>목포부</producer><productionYear>1932</productionYear><openYn>Y</openYn><originalYn>N</originalYn></item>
  <item><recordId>CJA-2</recordId><title>목포항 기록건</title><recordLevel>건</recordLevel>
    <parentRecordId>CJA-1</parentRecordId><producer>목포부</producer><productionYear>1932</productionYear></item>
  <item><recordId>CJA-2</recordId><title>중복 ID</title></item>
</items></body></response>"""

CAPABILITIES_XML = """<?xml version="1.0"?>
<wfs:WFS_Capabilities xmlns:wfs="http://www.opengis.net/wfs/2.0">
 <wfs:FeatureTypeList><wfs:FeatureType><wfs:Name>heritage:registered</wfs:Name></wfs:FeatureType></wfs:FeatureTypeList>
</wfs:WFS_Capabilities>"""

FEATURE_XML = """<?xml version="1.0"?>
<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:h="urn:test">
 <wfs:member><h:heritage gml:id="H-1"><h:name>구 목포 일본영사관</h:name><h:type>등록문화유산</h:type>
  <h:address>전라남도 목포시</h:address><h:designation>1981-09-25</h:designation><h:geometry><gml:Point><gml:pos>126.38 34.79</gml:pos></gml:Point></h:geometry>
 </h:heritage></wfs:member>
 <wfs:member><h:heritage gml:id="H-2"><h:geometry><gml:Point><gml:pos>126.39 34.80</gml:pos></gml:Point></h:geometry></h:heritage></wfs:member>
</wfs:FeatureCollection>"""


def response(body, status=200, url="https://apis.data.go.kr/api"):
    return BatchResponse(url, status, "application/xml; charset=utf-8", body.encode("utf-8"))


def test_archives_missing_key_is_safe_skip():
    with pytest.raises(BatchError, match="skipped_unverified_endpoint"):
        ADAPTERS["national_archives_api"].discovery_urls(["목포"], {})


def test_archives_public_urls_and_catalog_never_contain_key():
    adapter = ADAPTERS["national_archives_api"]
    public_url = adapter.spec.discovery_templates[0].format(query="fixture")
    assert "serviceKey" not in public_url and "fixture-secret" not in public_url
    candidates = adapter.discover(response(ARCHIVES_XML), public_url)
    serialized = json.dumps([item.__dict__ for item in candidates], ensure_ascii=False)
    assert "serviceKey" not in serialized and "fixture-secret" not in serialized
    assert "serviceKey=" in adapter.request_url(public_url, {
        "NATIONAL_ARCHIVES_API_KEY": "fixture-secret",
        "NATIONAL_ARCHIVES_API_KEY_FORMAT": "decoding",
    })


def test_archives_xml_fields_relationship_and_duplicate_id():
    found = ADAPTERS["national_archives_api"].discover(response(ARCHIVES_XML), "https://apis.data.go.kr/api")
    assert len(found) == 2
    assert found[0].document_type == "archival_metadata"
    assert found[0].discovery_metadata["producer"] == "목포부"
    assert found[1].parent_document_id == "CJA-1"
    assert found[1].discovery_metadata["production_year"] == "1932"


def test_archives_api_error_is_distinct_from_http_authentication():
    adapter = ADAPTERS["national_archives_api"]
    with pytest.raises(ApiResponseError) as auth_error:
        adapter.discover(response("<r><resultCode>30</resultCode><resultMsg>KEY ERROR</resultMsg></r>"), "https://apis.data.go.kr/api")
    assert auth_error.value.category == "api_authentication_error"
    with pytest.raises(ApiResponseError):
        adapter.discover(response("<r><resultCode>55</resultCode><resultMsg>INVALID REQUEST</resultMsg></r>"), "https://apis.data.go.kr/api")
    calls = []

    class AuthTransport:
        def get(self, url, timeout, max_bytes):
            calls.append(url)
            raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)

    control = RequestController(2, 1.2, lambda hosts: AuthTransport(), sleep=lambda value: None, max_retries=1)
    with pytest.raises(RequestDiagnosticError) as raised:
        control.get("https://apis.data.go.kr/api", SOURCE_SPECS["national_archives_api"], 1, 1000)
    assert raised.value.category == "http_401"
    assert len(calls) == 1


def test_archives_query_encoding_and_minimum_page_size():
    url = ADAPTERS["national_archives_api"].spec.discovery_templates[0].format(
        query="%EB%AA%A9%ED%8F%AC+%EA%B0%9C%ED%95%AD"
    )
    assert "%EB%AA%A9%ED%8F%AC+%EA%B0%9C%ED%95%AD" in url
    assert "numOfRows=10" in url


def test_wfs_capabilities_namespace_and_layers():
    adapter = ADAPTERS["heritage_wfs"]
    assert isinstance(adapter, HeritageWfsAdapter)
    assert adapter.capability_layers(response(CAPABILITIES_XML, url="https://heritage.example/wfs")) == ["heritage:registered"]


def test_wfs_getfeature_separates_geometry_and_spatial_metadata():
    adapter = ADAPTERS["heritage_wfs"]
    found = adapter.discover(response(FEATURE_XML, url="https://heritage.example/wfs"), "https://heritage.example/wfs?request=GetFeature")
    assert len(found) == 2
    assert all(item.document_type == "spatial_metadata" for item in found)
    assert found[0].discovery_metadata["attributes"]["name"] == "구 목포 일본영사관"
    assert found[0].discovery_metadata["geometry"]["pos"] == "126.38 34.79"
    assert found[1].discovery_metadata["historical_document_eligible"] is False


def test_wfs_builds_bounded_mokpo_filter_and_never_getmap():
    adapter = ADAPTERS["heritage_wfs"]
    env = {"HERITAGE_WFS_BASE_URL": "https://heritage.example/wfs"}
    capability = adapter.discovery_urls([], env)[0]
    feature = adapter.get_feature_url(env, "heritage:registered", 100)
    assert "GetCapabilities" in capability
    assert "GetFeature" in feature and "count=10" in feature and "%EB%AA%A9%ED%8F%AC" in feature
    assert "GetMap" not in capability + feature


def test_wfs_rejects_other_host_response():
    adapter = ADAPTERS["heritage_wfs"]
    spec = adapter.request_spec("https://heritage.example/wfs")
    control = RequestController(1, 1.2, lambda hosts: None, max_retries=0)
    with pytest.raises(BatchError):
        control.get("https://evil.example/wfs", spec, 1, 1000)


def test_cli_smoke_no_write_with_mocked_transport(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERITAGE_WFS_BASE_URL", "https://heritage.example/wfs")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "history_chatbot.collectors.public_history_batch.UrllibBatchTransport.get",
        lambda self, url, timeout, max_bytes: response(CAPABILITIES_XML, url=url),
    )
    assert main(["--batch-id", "stage-1", "--source", "heritage_wfs", "--smoke-test", "--no-write", "--max-requests", "2"]) == 0
    output = capsys.readouterr().out
    assert '"files_created": 0' in output
    assert not list(tmp_path.iterdir())


def test_minimum_delay_is_1_2_seconds():
    with pytest.raises(BatchError, match="1.2"):
        RequestController(1, 1.19, lambda hosts: None)


def test_archives_endpoint_is_explicitly_unverified_and_blocked():
    spec = SOURCE_SPECS["national_archives_api"]
    assert spec.endpoint_verification_status == "unverified"
    assert spec.endpoint_source == "estimated"
    assert spec.production_enabled is False
    assert ADAPTERS["national_archives_api"].readiness_status({
        "NATIONAL_ARCHIVES_API_KEY": "fixture",
        "NATIONAL_ARCHIVES_API_KEY_FORMAT": "decoding",
    }) == "skipped_unverified_endpoint"


def test_encoding_key_is_preserved_and_decoding_key_is_encoded_once():
    adapter = ADAPTERS["tour_api"]
    public_url = "https://apis.data.go.kr/example?x=1"
    encoded = adapter.request_url(public_url, {
        "TOUR_API_SERVICE_KEY": "fixture%2Bkey%3D",
        "TOUR_API_SERVICE_KEY_FORMAT": "encoding",
    })
    decoded = adapter.request_url(public_url, {
        "TOUR_API_SERVICE_KEY": "fixture+key=",
        "TOUR_API_SERVICE_KEY_FORMAT": "decoding",
    })
    assert "fixture%2Bkey%3D" in encoded and "%252B" not in encoded
    assert "fixture%2Bkey%3D" in decoded and "%252B" not in decoded


def test_unknown_key_format_skips_without_exposing_key():
    adapter = ADAPTERS["tour_api"]
    with pytest.raises(BatchError, match="skipped_unknown_key_format") as raised:
        adapter.request_url("https://apis.data.go.kr/example", {
            "TOUR_API_SERVICE_KEY": "fixture-secret",
        })
    assert "fixture-secret" not in str(raised.value)


@pytest.mark.parametrize("status,category", [(401, "http_401"), (403, "http_403"), (429, "http_429")])
def test_http_status_categories_and_no_auth_retry(status, category):
    calls = []

    class ErrorTransport:
        def get(self, url, timeout, max_bytes):
            calls.append(1)
            raise urllib.error.HTTPError(url, status, "fixture", {}, None)

    control = RequestController(2, 1.2, lambda hosts: ErrorTransport(),
                                sleep=lambda value: None, max_retries=1)
    with pytest.raises(RequestDiagnosticError) as raised:
        control.get("https://apis.data.go.kr/example", SOURCE_SPECS["tour_api"],
                    1, 1000, "area_code")
    assert raised.value.category == category
    assert len(calls) == 1
    assert control.events[0]["stage"] == "area_code"


def test_source_smoke_failure_isolated_and_telemetry_is_exact():
    calls = []

    class MixedTransport:
        def get(self, url, timeout, max_bytes):
            calls.append(url)
            if "apis.data.go.kr" in url:
                raise urllib.error.HTTPError(url, 401, "fixture", {}, None)
            if "GetCapabilities" in url:
                return response(CAPABILITIES_XML, url=url)
            return response(FEATURE_XML, url=url)

    control = RequestController(3, 1.2, lambda hosts: MixedTransport(),
                                sleep=lambda value: None, max_retries=1,
                                clock=lambda: 0.0)
    result = BatchPipeline(ADAPTERS, control).smoke_test(
        ["tour_api", "heritage_wfs"], {
            "TOUR_API_SERVICE_KEY": "fixture-secret",
            "TOUR_API_SERVICE_KEY_FORMAT": "decoding",
            "HERITAGE_WFS_BASE_URL": "https://heritage.example/wfs",
        }, 1, 100000,
    )
    assert result["request_count"] == 3
    assert result["source_request_counts"] == {"tour_api": 1, "heritage_wfs": 2}
    assert result["sources"][0]["status"] == "http_401"
    assert result["sources"][0]["failed_stage"] == "area_code"
    assert result["sources"][1]["status"] == "success"
    serialized = json.dumps(result)
    assert "fixture-secret" not in serialized and "serviceKey" not in serialized


def test_unverified_archives_and_unknown_tour_format_make_zero_requests():
    control = RequestController(2, 1.2, lambda hosts: (_ for _ in ()).throw(AssertionError()),
                                max_retries=0)
    result = BatchPipeline(ADAPTERS, control).smoke_test(
        ["national_archives_api", "tour_api"], {
            "NATIONAL_ARCHIVES_API_KEY": "fixture",
            "NATIONAL_ARCHIVES_API_KEY_FORMAT": "decoding",
            "TOUR_API_SERVICE_KEY": "fixture",
        }, 1, 1000,
    )
    assert result["request_count"] == 0
    assert [item["status"] for item in result["sources"]] == [
        "skipped_unverified_endpoint", "skipped_unknown_key_format",
    ]


def test_api_error_is_structured_and_echoed_secret_is_redacted():
    body = json.dumps({
        "response": {"header": {
            "resultCode": "30", "resultMsg": "serviceKey=fixture-secret invalid",
        }}
    }).encode("utf-8")

    class ApiErrorTransport:
        def get(self, url, timeout, max_bytes):
            return BatchResponse(url, 200, "application/json", body)

    control = RequestController(1, 1.2, lambda hosts: ApiErrorTransport(), max_retries=1)
    result = BatchPipeline(ADAPTERS, control).smoke_test(["tour_api"], {
        "TOUR_API_SERVICE_KEY": "fixture-secret",
        "TOUR_API_SERVICE_KEY_FORMAT": "decoding",
    }, 1, 10000)
    source = result["sources"][0]
    assert source["status"] == "api_authentication_error"
    assert source["api_result_code"] == "30"
    assert "fixture-secret" not in json.dumps(result)
