"""Bounded public-history discovery and provisional batch collection."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import socket
import tempfile
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urljoin, urlsplit, urlunsplit

from history_chatbot.collectors.hackathon_metadata import FIXED_RIGHTS, normalize_space, normalize_title


DECISIONS = (
    "accepted_hackathon", "accepted_metadata_only", "needs_review",
    "rejected_duplicate", "rejected_irrelevant", "rejected_empty",
    "rejected_access_policy", "rejected_access_barrier", "rejected_quality",
)
ACCEPTED_DECISIONS = {"accepted_hackathon", "accepted_metadata_only", "needs_review"}
ALLOWED_MEDIA_TYPES = {
    "text/html", "application/xhtml+xml", "application/json", "text/json",
    "application/xml", "text/xml", "text/csv", "application/csv",
}
BLOCKED_MEDIA_PREFIXES = ("application/pdf", "image/", "application/zip", "application/x-zip")
ERROR_PATTERNS = re.compile(
    r"(?:404\s*(?:not found|error)|500\s*(?:server error)?|페이지를 찾을 수 없|오류가 발생)", re.I
)
SEARCH_PAGE_PATTERNS = re.compile(r"(?:검색결과|search results?|통합검색)", re.I)
ACCESS_BARRIER_PATTERNS = re.compile(r"(?:captcha|recaptcha|paywall|로그인\s*(?:후|필요)|유료\s*(?:회원|구독))", re.I)
HISTORY_TERMS = (
    "개항", "해관", "세관", "조계", "거류지", "항만", "목포항", "철도", "목포역",
    "호남선", "근대", "일제강점기", "독립운동", "학생운동", "노동운동", "동양척식",
    "일본영사관", "면화", "미곡", "소금", "수산업", "문화유산", "도시 형성",
)
MOKPO_PLACES = (
    "목포", "유달산", "삼학도", "고하도", "만호동", "대의동", "양동", "북교동",
    "죽교동", "온금동", "목포역", "목포항", "근대역사관",
)
TRACKING_KEYS = {"fbclid", "gclid", "ref", "source", "utm_campaign", "utm_medium", "utm_source"}


class BatchError(RuntimeError):
    pass


class RequestDiagnosticError(BatchError):
    """A secret-free structured request or API failure."""

    def __init__(self, category: str, http_status: Optional[int] = None,
                 api_code: str = "", api_message: str = "", retryable: bool = False) -> None:
        BatchError.__init__(self, category)
        self.category = category
        self.http_status = http_status
        self.api_code = api_code
        self.api_message = api_message
        self.retryable = retryable


class AuthenticationError(RequestDiagnosticError):
    """An authentication failure which must never be retried."""


class ApiResponseError(RequestDiagnosticError):
    """A successful HTTP response containing an API-level error."""


class RedirectRejected(BatchError):
    """A redirect crossed the source allowlist and must stop the run."""


class GlobalSafetyError(BatchError):
    """A run-wide safety boundary which source isolation must not swallow."""


@dataclass
class BatchCandidate:
    document_id: str
    source_id: str
    title: str
    institution: str
    source_url: str
    canonical_url: str
    document_type: str = "descriptive_document"
    topic_tags: List[str] = field(default_factory=list)
    place_tags: List[str] = field(default_factory=lambda: ["목포"])
    published_date: str = ""
    parent_document_id: str = ""
    portal_name: str = ""
    original_institution: str = ""
    public_access_status: str = "public"
    license_name: str = ""
    discovery_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchCandidate":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass
class DetailDocument:
    candidate: BatchCandidate
    text: str
    final_url: str
    content_type: str
    response_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResponse:
    final_url: str
    status: int
    content_type: str
    body: bytes


@dataclass
class CandidateResult:
    document_id: str
    source_id: str
    title: str
    decision: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    institution: str
    allowed_hosts: Tuple[str, ...]
    discovery_templates: Tuple[str, ...]
    media_types: Tuple[str, ...] = tuple(ALLOWED_MEDIA_TYPES)
    api_key_environment: str = ""
    api_key_format_environment: str = ""
    portal_name: str = ""
    endpoint_verification_status: str = "verified"
    endpoint_source: str = "official"
    production_enabled: bool = True


SOURCE_SPECS = {
    "national_archives_html": SourceSpec(
        "national_archives_html", "국가기록원", ("www.archives.go.kr", "theme.archives.go.kr"),
        ("https://www.archives.go.kr/next/newsearch/searchTotal.do?keyword={query}",),
        portal_name="국가기록원",
    ),
    "national_archives_api": SourceSpec(
        "national_archives_api", "국가기록원", ("apis.data.go.kr", "www.archives.go.kr"),
        ("https://apis.data.go.kr/1741000/recordInformation/getrecordInformation?"
         "searchKeyword={query}&pageNo=1&numOfRows=10",),
        api_key_environment="NATIONAL_ARCHIVES_API_KEY",
        api_key_format_environment="NATIONAL_ARCHIVES_API_KEY_FORMAT",
        portal_name="나라기록물정보 OpenAPI",
        endpoint_verification_status="unverified", endpoint_source="estimated",
        production_enabled=False,
    ),
    "heritage_portal": SourceSpec(
        "heritage_portal", "국가유산청", ("www.heritage.go.kr", "heritage.go.kr"),
        ("https://www.heritage.go.kr/heri/cul/culSelectViewList.do?searchKeyword={query}",),
        portal_name="국가유산포털",
    ),
    "mokpo_official": SourceSpec(
        "mokpo_official", "목포시", ("www.mokpo.go.kr", "mokpo.go.kr"),
        ("https://www.mokpo.go.kr/search/?query={query}",), portal_name="목포시 공식 사이트",
    ),
    "tour_api": SourceSpec(
        "tour_api", "한국관광공사", ("apis.data.go.kr",),
        ("https://apis.data.go.kr/B551011/KorService2/searchKeyword2?MobileOS=ETC&MobileApp=MokpoHistoryRAG&_type=json&keyword={query}",),
        api_key_environment="TOUR_API_SERVICE_KEY",
        api_key_format_environment="TOUR_API_SERVICE_KEY_FORMAT",
        portal_name="공공데이터포털 TourAPI",
    ),
    "data_portal": SourceSpec(
        "data_portal", "공공데이터포털", ("www.data.go.kr", "data.go.kr"),
        ("https://www.data.go.kr/tcs/dss/selectDataSetList.do?keyword={query}",),
        portal_name="공공데이터포털",
    ),
    "heritage_wfs": SourceSpec(
        "heritage_wfs", "국가유산청", (), (),
        portal_name="국가유산 공간정보 WFS",
    ),
}

# Backward-compatible alias. New commands should use national_archives_html.
SOURCE_SPECS["national_archives"] = SourceSpec(
    "national_archives", SOURCE_SPECS["national_archives_html"].institution,
    SOURCE_SPECS["national_archives_html"].allowed_hosts,
    SOURCE_SPECS["national_archives_html"].discovery_templates,
    portal_name=SOURCE_SPECS["national_archives_html"].portal_name,
)


class BatchTransport(Protocol):
    def get(self, url: str, timeout: float, max_bytes: int) -> BatchResponse:
        """Perform one bounded GET without retries."""


class _RedirectPolicy(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: Sequence[str]) -> None:
        urllib.request.HTTPRedirectHandler.__init__(self)
        self.allowed_hosts = {item.lower() for item in allowed_hosts}

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        try:
            validate_public_url(newurl, self.allowed_hosts)
        except BatchError:
            raise RedirectRejected("redirect_rejected")
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )


class UrllibBatchTransport:
    def __init__(self, allowed_hosts: Sequence[str]) -> None:
        self.allowed_hosts = tuple(allowed_hosts)

    def get(self, url: str, timeout: float, max_bytes: int) -> BatchResponse:
        validate_public_url(url, self.allowed_hosts)
        opener = urllib.request.build_opener(_RedirectPolicy(self.allowed_hosts))
        request = urllib.request.Request(
            url, headers={"User-Agent": "MokpoHistoryHackathonBatch/1.0", "Accept": "text/html,application/json,application/xml,text/xml,text/csv"}
        )
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            validate_public_url(final_url, self.allowed_hosts)
            content_type = response.headers.get("Content-Type", "")
            validate_media_type(content_type, ())
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise BatchError("response size limit exceeded")
            return BatchResponse(final_url, getattr(response, "status", 200), content_type, body)


class _HtmlParser(HTMLParser):
    BLOCKED = {"head", "script", "style", "nav", "footer", "header", "aside", "form", "noscript"}

    def __init__(self) -> None:
        HTMLParser.__init__(self, convert_charrefs=True)
        self.hidden = 0
        self.title_depth = 0
        self.title = []  # type: List[str]
        self.text = []  # type: List[str]
        self.links = []  # type: List[Tuple[str, str]]
        self.href = ""
        self.anchor = []  # type: List[str]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.BLOCKED:
            self.hidden += 1
        if tag == "title":
            self.title_depth += 1
        if tag == "a" and not self.hidden:
            self.href = dict(attrs).get("href") or ""
            self.anchor = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self.href:
            self.links.append((self.href, normalize_space(" ".join(self.anchor))))
            self.href = ""
            self.anchor = []
        if tag in self.BLOCKED and self.hidden:
            self.hidden -= 1
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        value = html.unescape(data)
        if self.title_depth:
            self.title.append(value)
        if self.hidden:
            return
        self.text.append(value)
        if self.href:
            self.anchor.append(value)


def canonicalize_public_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(sorted(
        (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_KEYS
    ))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def validate_public_url(url: str, allowed_hosts: Iterable[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise BatchError("HTTPS only: " + url)
    if parts.username or parts.password:
        raise BatchError("URL credentials are forbidden")
    if (parts.hostname or "").lower() not in {item.lower() for item in allowed_hosts}:
        raise BatchError("host is not allowed: " + (parts.hostname or ""))


def validate_media_type(content_type: str, explicitly_allowed_binary: Sequence[str]) -> str:
    media = content_type.split(";", 1)[0].strip().lower()
    if media in explicitly_allowed_binary:
        return media
    if media.startswith(BLOCKED_MEDIA_PREFIXES) or media not in ALLOWED_MEDIA_TYPES:
        raise BatchError("response media type is not allowed: " + media)
    return media


def decode_body(body: bytes) -> str:
    for encoding in ("utf-8", "cp949"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise BatchError("response encoding is not supported")


def extract_payload(response: BatchResponse, candidate: BatchCandidate) -> DetailDocument:
    if response.status < 200 or response.status >= 300:
        raise BatchError("HTTP response is not successful")
    media = validate_media_type(response.content_type, ())
    source = decode_body(response.body)
    metadata = {}  # type: Dict[str, Any]
    if media in ("text/html", "application/xhtml+xml"):
        parser = _HtmlParser()
        parser.feed(source)
        title = normalize_space(" ".join(parser.title))
        text = normalize_space("\n".join(parser.text))
        if title:
            metadata["page_title"] = title
    elif media in ("application/json", "text/json"):
        value = json.loads(source)
        text = normalize_space(_flatten_values(value))
    elif media in ("application/xml", "text/xml"):
        root = ET.fromstring(source)
        text = normalize_space(" ".join(item for item in root.itertext()))
    else:
        rows = list(csv.reader(io.StringIO(source)))
        text = normalize_space(" ".join(" ".join(row) for row in rows))
    return DetailDocument(candidate, text, response.final_url, response.content_type, len(response.body), metadata)


def _flatten_values(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_values(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_values(item) for item in value)
    return str(value or "")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def _xml_fields(element: ET.Element) -> Dict[str, str]:
    values = {}  # type: Dict[str, str]
    for child in element.iter():
        if child is element or list(child):
            continue
        value = normalize_space(child.text or "")
        if value:
            values[_local_name(child.tag).lower()] = value
    return values


def _pick(values: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name.lower(), "")
        if value:
            return value
    return ""


def _api_error(root: ET.Element) -> Tuple[str, str]:
    fields = _xml_fields(root)
    code = _pick(fields, "resultCode", "returnReasonCode", "errorCode", "errCode")
    message = _pick(fields, "resultMsg", "returnAuthMsg", "errorMessage", "errMsg")
    return code, message


def _safe_api_message(value: str) -> str:
    compact = normalize_space(value)
    compact = re.sub(r"(?i)(serviceKey|authorization)\s*[=:]\s*\S+", r"\1=[redacted]", compact)
    compact = re.sub(r"https?://\S+", "[url redacted]", compact)
    return compact[:200]


def _encoded_service_key(key: str, key_format: str) -> str:
    if key_format == "decoding":
        return quote(key, safe="")
    if key_format != "encoding":
        raise BatchError("skipped_unknown_key_format")
    if not re.fullmatch(r"[A-Za-z0-9._~!$'()*+,;=:%/-]+", key):
        raise BatchError("invalid_encoding_key_format")
    if re.search(r"%(?![0-9A-Fa-f]{2})", key):
        raise BatchError("invalid_encoding_key_format")
    return key


def _json_api_error(value: Mapping[str, Any]) -> Tuple[str, str]:
    try:
        header = value["response"]["header"]
    except (KeyError, TypeError):
        return "", ""
    if not isinstance(header, dict):
        return "", ""
    return normalize_space(header.get("resultCode")), normalize_space(header.get("resultMsg"))


class PublicSourceAdapter:
    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec

    def discovery_urls(self, keywords: Sequence[str], environment: Mapping[str, str]) -> List[str]:
        if self.spec.api_key_environment:
            if not environment.get(self.spec.api_key_environment, "").strip():
                raise BatchError("skipped_missing_api_key:" + self.spec.api_key_environment)
        urls = []
        for keyword in keywords:
            for template in self.spec.discovery_templates:
                url = template.format(query=quote_plus(keyword))
                validate_public_url(url, self.spec.allowed_hosts)
                urls.append(url)
        return urls

    def readiness_status(self, environment: Mapping[str, str]) -> str:
        if not self.spec.production_enabled or self.spec.endpoint_verification_status != "verified":
            return "skipped_unverified_endpoint"
        if self.spec.api_key_environment and not environment.get(self.spec.api_key_environment, "").strip():
            return "skipped_missing_api_key"
        if self.spec.api_key_format_environment:
            key_format = environment.get(self.spec.api_key_format_environment, "").strip().lower()
            if key_format not in ("encoding", "decoding"):
                return "skipped_unknown_key_format"
        return "ready"

    def request_url(self, public_url: str, environment: Mapping[str, str]) -> str:
        """Insert credentials at the final request boundary only."""
        if not self.spec.api_key_environment:
            return public_url
        key = environment.get(self.spec.api_key_environment, "").strip()
        if not key:
            raise BatchError("skipped_missing_api_key:" + self.spec.api_key_environment)
        key_format = environment.get(self.spec.api_key_format_environment, "").strip().lower()
        if key_format not in ("encoding", "decoding"):
            raise BatchError("skipped_unknown_key_format")
        separator = "&" if "?" in public_url else "?"
        return public_url + separator + "serviceKey=" + _encoded_service_key(key, key_format)

    def request_spec(self, public_url: str) -> SourceSpec:
        return self.spec

    def discover(self, response: BatchResponse, request_url: str) -> List[BatchCandidate]:
        media = validate_media_type(response.content_type, ())
        if media not in ("text/html", "application/xhtml+xml"):
            return self._discover_structured(response)
        source = decode_body(response.body)
        parser = _HtmlParser()
        parser.feed(source)
        candidates = []
        for href, title in parser.links:
            if not href or not title:
                continue
            url = canonicalize_public_url(urljoin(response.final_url, href))
            try:
                validate_public_url(url, self.spec.allowed_hosts)
            except BatchError:
                continue
            if not is_relevant(title, title)[0]:
                continue
            candidates.append(self._candidate(url, title))
        return candidates

    def _discover_structured(self, response: BatchResponse) -> List[BatchCandidate]:
        document = extract_payload(response, self._candidate(response.final_url, "discovery"))
        # Structured discovery needs stable item links; keep it reviewable instead of inventing URLs.
        if not is_relevant(document.text, document.text)[0]:
            return []
        return []

    def fetch_detail(self, candidate: BatchCandidate, response: BatchResponse) -> DetailDocument:
        validate_public_url(response.final_url, self.spec.allowed_hosts)
        return extract_payload(response, candidate)

    def detail_url(self, candidate: BatchCandidate, environment: Mapping[str, str]) -> str:
        validate_public_url(candidate.source_url, self.spec.allowed_hosts)
        return candidate.source_url

    def _candidate(self, url: str, title: str) -> BatchCandidate:
        canonical = canonicalize_public_url(url)
        digest = hashlib.sha256((self.spec.source_id + "\0" + canonical).encode("utf-8")).hexdigest()[:20]
        return BatchCandidate(
            document_id="public-%s-%s" % (self.spec.source_id, digest),
            source_id=self.spec.source_id,
            title=normalize_space(title), institution=self.spec.institution,
            source_url=url, canonical_url=canonical, portal_name=self.spec.portal_name,
            original_institution=self.spec.institution,
        )


class TourApiAdapter(PublicSourceAdapter):
    def discover(self, response: BatchResponse, request_url: str) -> List[BatchCandidate]:
        value = json.loads(decode_body(response.body))
        code, message = _json_api_error(value)
        if code and code not in ("0", "00", "0000"):
            category = "api_authentication_error" if code in ("10", "20", "30") else (
                "api_rate_limit_error" if code in ("22", "429") else "api_application_error"
            )
            raise ApiResponseError(category, api_code=code, api_message=_safe_api_message(message))
        try:
            items = value["response"]["body"]["items"]["item"]
        except (KeyError, TypeError):
            return []
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return []
        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content_id = normalize_space(item.get("contentid"))
            title = normalize_space(item.get("title"))
            if not content_id or not title or not is_relevant(title, normalize_space(_flatten_values(item)))[0]:
                continue
            detail_url = "https://apis.data.go.kr/B551011/KorService2/detailCommon2?" + urlencode({
                "MobileOS": "ETC", "MobileApp": "MokpoHistoryRAG", "_type": "json",
                "contentId": content_id, "defaultYN": "Y", "overviewYN": "Y",
            })
            candidate = self._candidate(detail_url, title)
            candidate.document_id = "tour-api-" + content_id
            candidate.discovery_metadata["content_id"] = content_id
            candidates.append(candidate)
        return candidates

    def detail_url(self, candidate: BatchCandidate, environment: Mapping[str, str]) -> str:
        return self.request_url(candidate.source_url, environment)


class NationalArchivesApiAdapter(PublicSourceAdapter):
    ITEM_NAMES = {"item", "record", "row", "list"}

    def discovery_urls(self, keywords: Sequence[str], environment: Mapping[str, str]) -> List[str]:
        if self.readiness_status(environment) == "skipped_unverified_endpoint":
            raise BatchError("skipped_unverified_endpoint")
        if not environment.get(self.spec.api_key_environment, "").strip():
            raise BatchError("skipped_missing_api_key:" + self.spec.api_key_environment)
        return [self.spec.discovery_templates[0].format(query=quote_plus(keyword)) for keyword in keywords]

    def discover(self, response: BatchResponse, request_url: str) -> List[BatchCandidate]:
        if response.status in (401, 403):
            raise AuthenticationError(_http_category(response.status), response.status)
        root = ET.fromstring(decode_body(response.body))
        code, message = _api_error(root)
        if code and code not in ("0", "00", "0000", "INFO-0"):
            if code in ("10", "20", "30", "99") or "auth" in message.lower() or "key" in message.lower():
                raise ApiResponseError("api_authentication_error", api_code=code,
                                       api_message=_safe_api_message(message))
            category = "api_rate_limit_error" if code in ("22", "429") else "api_application_error"
            raise ApiResponseError(category, api_code=code, api_message=_safe_api_message(message))
        candidates = []
        seen = set()
        for element in root.iter():
            if _local_name(element.tag).lower() not in self.ITEM_NAMES:
                continue
            fields = _xml_fields(element)
            record_id = _pick(fields, "recordId", "record_id", "rcdId", "itemId", "id")
            title = _pick(fields, "title", "recordTitle", "sj", "name")
            if not record_id or not title or record_id in seen:
                continue
            seen.add(record_id)
            parent_id = _pick(fields, "parentRecordId", "seriesId", "fileId", "parentId")
            level = _pick(fields, "recordLevel", "recordType", "level", "type")
            public_url = "https://www.archives.go.kr/next/newsearch/showDetailPopup.do?rc_code=" + quote_plus(record_id)
            candidate = BatchCandidate(
                document_id="national-archives-" + re.sub(r"[^A-Za-z0-9_-]", "-", record_id),
                source_id=self.spec.source_id, title=title, institution="국가기록원",
                source_url=public_url, canonical_url=canonicalize_public_url(public_url),
                document_type="archival_metadata", parent_document_id=parent_id,
                portal_name=self.spec.portal_name, original_institution=_pick(fields, "producer", "agency", "productionAgency"),
            )
            candidate.published_date = _pick(fields, "productionYear", "year", "date")
            candidate.discovery_metadata.update({
                "record_id": record_id, "record_level": level,
                "record_type": _pick(fields, "recordType", "materialType", "type"),
                "producer": candidate.original_institution,
                "production_year": candidate.published_date,
                "public_status": _pick(fields, "publicStatus", "openYn", "isPublic"),
                "original_available": _pick(fields, "originalYn", "fileYn", "hasOriginal"),
                "parent_record_id": parent_id,
            })
            candidates.append(candidate)
        return candidates


class HeritageWfsAdapter(PublicSourceAdapter):
    GEOMETRY_NAMES = {"geometry", "geom", "the_geom", "shape", "boundedby", "point", "polygon", "multipolygon", "coordinates", "pos", "poslist"}

    def _base(self, environment: Mapping[str, str]) -> str:
        base = environment.get("HERITAGE_WFS_BASE_URL", "").strip()
        if not base:
            raise BatchError("skipped_missing_api_key:HERITAGE_WFS_BASE_URL")
        parts = urlsplit(base)
        validate_public_url(base, ((parts.hostname or ""),))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    def discovery_urls(self, keywords: Sequence[str], environment: Mapping[str, str]) -> List[str]:
        base = self._base(environment)
        separator = "&" if "?" in base else "?"
        return [base + separator + urlencode({"service": "WFS", "request": "GetCapabilities"})]

    def get_feature_url(self, environment: Mapping[str, str], type_name: str, count: int = 10) -> str:
        base = self._base(environment)
        separator = "&" if "?" in base else "?"
        return base + separator + urlencode({
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": type_name, "count": min(max(count, 1), 10),
            "CQL_FILTER": "address LIKE '%목포%'",
        })

    def request_spec(self, public_url: str) -> SourceSpec:
        host = (urlsplit(public_url).hostname or "").lower()
        return SourceSpec(self.spec.source_id, self.spec.institution, (host,), (), portal_name=self.spec.portal_name)

    def capability_layers(self, response: BatchResponse) -> List[str]:
        root = ET.fromstring(decode_body(response.body))
        layers = []
        for element in root.iter():
            if _local_name(element.tag).lower() == "featuretype":
                fields = _xml_fields(element)
                name = _pick(fields, "name")
                if name and name not in layers:
                    layers.append(name)
        return layers

    def discover(self, response: BatchResponse, request_url: str) -> List[BatchCandidate]:
        root = ET.fromstring(decode_body(response.body))
        if _local_name(root.tag).lower().endswith("capabilities"):
            return []
        candidates = []
        seen = set()
        for member in root.iter():
            if _local_name(member.tag).lower() not in ("member", "featuremember") or not list(member):
                continue
            feature = list(member)[0]
            fields = _xml_fields(feature)
            attributes = {k: v for k, v in fields.items() if k not in self.GEOMETRY_NAMES}
            geometry = {k: v for k, v in fields.items() if k in self.GEOMETRY_NAMES}
            identifier = _pick(attributes, "id", "fid", "heritageId", "ccbaKdcd", "manageNo") or feature.attrib.get("{http://www.opengis.net/gml/3.2}id", "")
            title = _pick(attributes, "name", "title", "heritageName", "ccbaMnm1")
            address = _pick(attributes, "address", "addr", "location", "ccbaCtcdNm")
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            candidate = BatchCandidate(
                document_id="heritage-wfs-" + re.sub(r"[^A-Za-z0-9_-]", "-", identifier),
                source_id=self.spec.source_id, title=title or identifier,
                institution=self.spec.institution, source_url=canonicalize_public_url(request_url),
                canonical_url=canonicalize_public_url(request_url) + "#" + quote_plus(identifier),
                document_type="spatial_metadata", portal_name=self.spec.portal_name,
                original_institution=self.spec.institution,
            )
            candidate.discovery_metadata.update({
                "heritage_id": identifier, "heritage_type": _pick(attributes, "type", "category", "ccbaKdcdNm"),
                "address": address, "designation": _pick(attributes, "designation", "designatedDate", "ccbaAsdt"),
                "attributes": attributes, "geometry": geometry,
                "historical_document_eligible": False,
            })
            candidates.append(candidate)
        return candidates


ADAPTERS = {}
for _name, _spec in SOURCE_SPECS.items():
    if _name == "tour_api":
        ADAPTERS[_name] = TourApiAdapter(_spec)
    elif _name == "national_archives_api":
        ADAPTERS[_name] = NationalArchivesApiAdapter(_spec)
    elif _name == "heritage_wfs":
        ADAPTERS[_name] = HeritageWfsAdapter(_spec)
    else:
        ADAPTERS[_name] = PublicSourceAdapter(_spec)


class RequestController:
    def __init__(self, max_requests: int, delay_seconds: float,
                 transport_factory: Callable[[Sequence[str]], BatchTransport],
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, max_retries: int = 1) -> None:
        if max_requests < 1 or max_requests > 500:
            raise BatchError("max-requests must be between 1 and 500")
        if delay_seconds < 1.2:
            raise BatchError("host delay must be at least 1.2 seconds")
        if max_retries < 0 or max_retries > 1:
            raise BatchError("max retries cannot exceed one")
        self.max_requests = max_requests
        self.delay_seconds = delay_seconds
        self.transport_factory = transport_factory
        self.sleep = sleep
        self.clock = clock
        self.max_retries = max_retries
        self.request_count = 0
        self.source_request_counts = {}  # type: Dict[str, int]
        self.events = []  # type: List[Dict[str, Any]]
        self.last_request = {}  # type: Dict[str, float]

    def get(self, url: str, spec: SourceSpec, timeout: float, max_bytes: int,
            stage: str = "request") -> BatchResponse:
        try:
            validate_public_url(url, spec.allowed_hosts)
        except BatchError:
            raise GlobalSafetyError("host_or_https_rejected")
        host = (urlsplit(url).hostname or "").lower()
        error = None  # type: Optional[Exception]
        for attempt in range(self.max_retries + 1):
            last = self.last_request.get(host)
            if last is not None:
                remaining = self.delay_seconds - (self.clock() - last)
                if remaining > 0:
                    self.sleep(remaining)
            if self.request_count >= self.max_requests:
                raise GlobalSafetyError("maximum request count exceeded")
            self.request_count += 1
            source_number = self.source_request_counts.get(spec.source_id, 0) + 1
            self.source_request_counts[spec.source_id] = source_number
            self.last_request[host] = self.clock()
            event = {
                "source_id": spec.source_id, "stage": stage,
                "request_number": self.request_count, "source_request_number": source_number,
                "retry": attempt > 0, "outcome": "pending", "http_status": None,
            }
            self.events.append(event)
            try:
                response = self.transport_factory(spec.allowed_hosts).get(url, timeout, max_bytes)
                try:
                    validate_public_url(response.final_url, spec.allowed_hosts)
                except BatchError:
                    event["outcome"] = "redirect_rejected"
                    raise RedirectRejected("redirect_rejected")
                event["http_status"] = response.status
                if response.status in (401, 403, 404, 429) or response.status >= 500:
                    raise RequestDiagnosticError(_http_category(response.status), response.status,
                                                 retryable=response.status >= 500)
                if response.status < 200 or response.status >= 300:
                    raise RequestDiagnosticError("http_error", response.status)
                validate_media_type(response.content_type, ())
                if len(response.body) > max_bytes:
                    raise BatchError("response size limit exceeded")
                event["outcome"] = "success"
                return response
            except Exception as exc:
                error = exc
                if isinstance(exc, RedirectRejected):
                    raise
                diagnostic = _request_diagnostic(exc)
                if diagnostic is not None:
                    event["outcome"] = diagnostic.category
                    event["http_status"] = diagnostic.http_status
                    if not diagnostic.retryable or attempt >= self.max_retries:
                        raise diagnostic
                else:
                    event["outcome"] = "request_error"
                if attempt >= self.max_retries:
                    break
        raise BatchError("request failed: " + type(error).__name__)


def _http_category(status: int) -> str:
    if status in (401, 403, 404, 429):
        return "http_%d" % status
    if status >= 500:
        return "http_5xx"
    return "http_error"


def _request_diagnostic(exc: Exception) -> Optional[RequestDiagnosticError]:
    if isinstance(exc, RequestDiagnosticError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        return RequestDiagnosticError(_http_category(exc.code), exc.code,
                                      retryable=exc.code >= 500)
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return RequestDiagnosticError("timeout", retryable=True)
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError)):
        return RequestDiagnosticError("timeout", retryable=True)
    return None


def is_relevant(title: str, text: str) -> Tuple[bool, List[str]]:
    combined = normalize_space(title + " " + text)
    title_direct = any(place in title for place in MOKPO_PLACES)
    mokpo_count = combined.count("목포")
    direct_place = any(place in combined for place in MOKPO_PLACES[1:])
    history = [term for term in HISTORY_TERMS if term in combined]
    relevant = title_direct or direct_place or mokpo_count >= 2 or (mokpo_count >= 1 and bool(history))
    reasons = []
    if title_direct:
        reasons.append("title_place_match")
    if direct_place:
        reasons.append("specific_mokpo_place")
    if mokpo_count >= 2:
        reasons.append("mokpo_repeated")
    if mokpo_count and history:
        reasons.append("mokpo_history_link:" + ",".join(history[:5]))
    return relevant, reasons


def quality_decision(candidate: BatchCandidate, text: str) -> CandidateResult:
    compact = normalize_space(text)
    if not compact:
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_empty", ["empty body"])
    if ACCESS_BARRIER_PATTERNS.search(compact):
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_access_barrier", ["login/captcha/paywall marker"])
    if ERROR_PATTERNS.search(compact):
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_quality", ["error page marker"])
    if SEARCH_PAGE_PATTERNS.search(compact) and len(compact) < 300:
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_quality", ["search results page"])
    relevant, relevance_reasons = is_relevant(candidate.title, compact)
    if not relevant:
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_irrelevant", ["Mokpo is not a substantive subject"])
    if len(compact) < 50:
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "rejected_quality", ["text shorter than 50 characters"])
    if candidate.document_type == "metadata_document":
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "accepted_metadata_only", relevance_reasons)
    if len(compact) < 300:
        return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "needs_review", relevance_reasons, ["descriptive text shorter than 300 characters"])
    return CandidateResult(candidate.document_id, candidate.source_id, candidate.title, "accepted_hackathon", relevance_reasons)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchError("invalid JSONL row %d: %s" % (number, exc))
        if not isinstance(value, dict):
            raise BatchError("JSONL row must be an object")
        rows.append(value)
    return rows


def _first(record: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return normalize_space(value)
    return ""


class DuplicateIndex:
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.ids = set()  # type: set
        self.urls = set()  # type: set
        self.hashes = set()  # type: set
        self.titles = []  # type: List[Tuple[str, str]]
        self.bodies = []  # type: List[str]
        for record in records:
            self.add_record(record)

    def add_record(self, record: Mapping[str, Any]) -> None:
        for name in ("document_id", "source_id"):
            value = _first(record, [name])
            if value:
                self.ids.add(value)
        for name in ("source_url", "canonical_url", "canonical_source_url"):
            value = _first(record, [name])
            if value:
                self.urls.add(canonicalize_public_url(value))
        for name in ("content_hash", "body_hash", "excerpt_sha256", "extracted_sha256"):
            value = _first(record, [name])
            if value:
                self.hashes.add(value)
        title = _first(record, ["title", "source_title"])
        if title:
            self.titles.append((normalize_title(title), title))

    def check(self, candidate: BatchCandidate, text_hash: str, extracted_hash: str,
              body_text: str = "") -> Tuple[List[str], List[str]]:
        duplicates = []
        warnings = []
        if candidate.document_id in self.ids:
            duplicates.append("document_id")
        for url_type, url in (("source_url", candidate.source_url), ("canonical_url", candidate.canonical_url)):
            if canonicalize_public_url(url) in self.urls:
                duplicates.append(url_type)
        if text_hash in self.hashes:
            duplicates.append("body_hash")
        if extracted_hash in self.hashes:
            duplicates.append("extracted_hash")
        normalized = normalize_title(candidate.title)
        for other, original in self.titles:
            ratio = SequenceMatcher(None, normalized, other).ratio()
            if ratio >= 0.88:
                warnings.append("similar_title:%.3f:%s" % (ratio, original))
        for other_body in self.bodies:
            ratio = SequenceMatcher(None, normalize_space(body_text), other_body).ratio()
            if ratio >= 0.90:
                warnings.append("similar_body:%.3f" % ratio)
        return sorted(set(duplicates)), sorted(set(warnings))

    def add_body(self, text: str) -> None:
        compact = normalize_space(text)
        if compact:
            self.bodies.append(compact)


def render_extracted(candidate: BatchCandidate, text: str) -> bytes:
    value = "제목: %s\n기관: %s\n상세 URL: %s\n\n%s\n" % (
        candidate.title, candidate.institution, candidate.source_url, normalize_space(text)
    )
    return value.encode("utf-8")


def build_manifest_record(candidate: BatchCandidate, detail: DetailDocument,
                          target: Path, decision: str, collected_at: str) -> Dict[str, Any]:
    extracted = render_extracted(candidate, detail.text)
    body_hash = hashlib.sha256(normalize_space(detail.text).encode("utf-8")).hexdigest()
    record = {
        "document_id": candidate.document_id, "source_id": candidate.source_id,
        "title": candidate.title, "source_title": candidate.title,
        "institution": candidate.institution, "publisher": candidate.institution,
        "source_name": candidate.portal_name or candidate.institution,
        "license_name": candidate.license_name, "source_url": candidate.source_url,
        "canonical_url": candidate.canonical_url, "document_type": candidate.document_type,
        "topic": candidate.topic_tags, "topic_tags": candidate.topic_tags,
        "place_tags": candidate.place_tags, "published_date": candidate.published_date,
        "related_document_id": candidate.parent_document_id,
        "collected_at": collected_at, "active": True, "network_requested": True,
        "extracted_text_path": target.as_posix(), "content_hash": body_hash,
        "extracted_sha256": hashlib.sha256(extracted).hexdigest(),
        "collection_metadata": {
            "batch_decision": decision, "portal_name": candidate.portal_name,
            "original_institution": candidate.original_institution,
            "public_access_status": candidate.public_access_status,
            "license_name": candidate.license_name,
            "allowed_for_hackathon_rag": False,
            "allowed_for_public_production": False,
            "response_content_type": detail.content_type.split(";", 1)[0],
            "response_bytes": detail.response_bytes,
            "collection_method": "public_history_batch",
        },
    }
    record.update(FIXED_RIGHTS)
    record["allowed_for_rag"] = False
    return record


def atomic_write(files: Mapping[Path, bytes], replace_file: Callable[[str, str], None] = os.replace) -> None:
    if not files:
        return
    anchor = next(iter(files)).parent
    anchor.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".public-history-batch-", dir=str(anchor)))
    backups = {}  # type: Dict[Path, Path]
    committed = []  # type: List[Path]
    try:
        staged = []
        for index, (target, content) in enumerate(files.items()):
            item = stage / ("new-%04d" % index)
            item.write_bytes(content)
            if target.exists():
                backup = stage / ("old-%04d" % index)
                shutil.copyfile(str(target), str(backup))
                backups[target] = backup
            staged.append((item, target))
        for item, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            replace_file(str(item), str(target))
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            backup = backups.get(target)
            try:
                if backup and backup.exists():
                    shutil.copyfile(str(backup), str(target))
                elif target.exists():
                    target.unlink()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(str(stage), ignore_errors=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values).encode("utf-8")


def markdown_report(report: Mapping[str, Any]) -> bytes:
    counts = report.get("counts", {})
    lines = ["# Public history batch report", "", "- Batch: `%s`" % report.get("batch_id", ""),
             "- Mode: `%s`" % report.get("mode", ""), "- Requests: %s" % report.get("request_count", 0), "",
             "## Counts", ""]
    for key in sorted(counts):
        lines.append("- %s: %s" % (key, counts[key]))
    lines.extend(["", "## Failures", ""])
    for item in report.get("failures", []):
        lines.append("- `%s`: %s" % (item.get("document_id", "unknown"), "; ".join(item.get("reasons", []))))
    if not report.get("failures"):
        lines.append("- none")
    return ("\n".join(lines) + "\n").encode("utf-8")


class BatchPipeline:
    def __init__(self, adapters: Mapping[str, PublicSourceAdapter], controller: Optional[RequestController] = None) -> None:
        self.adapters = dict(adapters)
        self.controller = controller

    def dry_run(self, source_ids: Sequence[str], paths: Mapping[str, Path], limits: Mapping[str, Any]) -> Dict[str, Any]:
        self._validate_limits(limits)
        specs = []
        for source_id in source_ids:
            adapter = self._adapter(source_id)
            specs.append({
                "source_id": source_id, "allowed_hosts": list(adapter.spec.allowed_hosts),
                "api_key_environment": adapter.spec.api_key_environment or None,
                "api_key_format_environment": adapter.spec.api_key_format_environment or None,
                "discovery_templates": list(adapter.spec.discovery_templates),
                "endpoint_verification_status": adapter.spec.endpoint_verification_status,
                "endpoint_source": adapter.spec.endpoint_source,
                "production_enabled": adapter.spec.production_enabled,
            })
        return {
            "mode": "dry-run", "network": False, "files_created": False,
            "sources": specs, "limits": dict(limits),
            "paths": {key: str(value) for key, value in paths.items()},
            "rights": dict(FIXED_RIGHTS),
            "hackathon_usage": {"allowed_for_hackathon_rag": False, "allowed_for_public_production": False},
            "unchanged": ["raw", "chunks", "index", "allowed_for_rag"],
        }

    def smoke_test(self, source_ids: Sequence[str], environment: Mapping[str, str],
                   timeout: float, max_bytes: int) -> Dict[str, Any]:
        """Run bounded probes and return only structural summaries; never write files."""
        controller = self._controller()
        source_results = []  # type: List[Dict[str, Any]]

        def probe(adapter: PublicSourceAdapter, public_url: str, stage: str) -> BatchResponse:
            request_url = adapter.request_url(public_url, environment)
            return controller.get(request_url, adapter.request_spec(public_url), timeout, max_bytes, stage)

        for source_id in source_ids:
            adapter = self._adapter(source_id)
            result = {
                "source_id": source_id, "status": "pending", "failed_stage": None,
                "http_status": None, "api_result_code": "", "api_result_message": "",
                "steps": [], "stopped_after_failure": False,
            }  # type: Dict[str, Any]
            source_results.append(result)
            readiness = adapter.readiness_status(environment)
            if readiness != "ready":
                result["status"] = readiness
                result["stopped_after_failure"] = True
                continue
            stage = "readiness"
            try:
                if source_id == "national_archives_api":
                    stage = "search"
                    public_url = adapter.discovery_urls(["목포"], environment)[0]
                    response = probe(adapter, public_url, stage)
                    result["steps"].append(_response_summary(source_id, stage, response))
                    found = adapter.discover(response, public_url)
                    if found:
                        stage = "detail"
                        detail = found[0].source_url
                        detail_response = controller.get(detail, adapter.request_spec(detail), timeout,
                                                         max_bytes, stage)
                        result["steps"].append(_response_summary(source_id, stage, detail_response))
                    else:
                        result["status"] = "empty_result"
                elif source_id == "tour_api":
                    base = "https://apis.data.go.kr/B551011/KorService2/"
                    area = base + "areaCode2?" + urlencode({"MobileOS": "ETC", "MobileApp": "MokpoHistoryRAG", "_type": "json"})
                    stage = "area_code"
                    response = probe(adapter, area, stage)
                    result["steps"].append(_response_summary(source_id, stage, response))
                    _raise_json_api_error(response)
                    stage = "keyword_search"
                    public_url = adapter.discovery_urls(["목포 근대역사"], environment)[0]
                    response = probe(adapter, public_url, stage)
                    result["steps"].append(_response_summary(source_id, stage, response))
                    found = adapter.discover(response, public_url)
                    if found:
                        stage = "common_detail"
                        response = probe(adapter, found[0].source_url, stage)
                        result["steps"].append(_response_summary(source_id, stage, response))
                        _raise_json_api_error(response)
                    else:
                        result["status"] = "empty_result"
                elif source_id == "heritage_wfs":
                    stage = "get_capabilities"
                    public_url = adapter.discovery_urls([], environment)[0]
                    response = probe(adapter, public_url, stage)
                    result["steps"].append(_response_summary(source_id, stage, response))
                    layers = adapter.capability_layers(response)  # type: ignore[attr-defined]
                    if layers:
                        stage = "get_feature"
                        response = probe(adapter, adapter.get_feature_url(environment, layers[0], 1), stage)  # type: ignore[attr-defined]
                        result["steps"].append(_response_summary(source_id, stage, response))
                        found = adapter.discover(response, public_url)
                        if not found:
                            result["status"] = "empty_result"
                    else:
                        result["status"] = "empty_result"
                else:
                    raise BatchError("smoke-test source is not supported: " + source_id)
                if result["status"] == "pending":
                    result["status"] = "success"
            except RequestDiagnosticError as exc:
                result.update({
                    "status": exc.category, "failed_stage": stage,
                    "http_status": exc.http_status, "api_result_code": exc.api_code,
                    "api_result_message": exc.api_message, "stopped_after_failure": True,
                })
                if controller.events:
                    controller.events[-1]["outcome"] = exc.category
            except (ValueError, json.JSONDecodeError, ET.ParseError, UnicodeError):
                result.update({"status": "parser_error", "failed_stage": stage,
                               "stopped_after_failure": True})
            except (RedirectRejected, GlobalSafetyError):
                raise
            except BatchError as exc:
                status = str(exc) if str(exc).startswith("skipped_") else "request_error"
                result.update({"status": status, "failed_stage": stage,
                               "stopped_after_failure": True})
            result["request_count"] = controller.source_request_counts.get(source_id, 0)
        for result in source_results:
            result.setdefault("request_count", controller.source_request_counts.get(result["source_id"], 0))
        return {
            "mode": "smoke-test", "network": True, "no_write": True,
            "files_created": 0, "request_count": controller.request_count,
            "source_request_counts": dict(controller.source_request_counts),
            "requests": list(controller.events), "sources": source_results,
        }

    def discover(self, batch_id: str, source_ids: Sequence[str], keywords: Sequence[str],
                 catalog: Path, report_json: Path, report_md: Path,
                 environment: Mapping[str, str], timeout: float, max_bytes: int,
                 limits: Mapping[str, Any], replace_file: Callable[[str, str], None] = os.replace) -> Dict[str, Any]:
        self._validate_limits(limits)
        controller = self._controller()
        candidates = []  # type: List[BatchCandidate]
        failures = []  # type: List[Dict[str, Any]]
        seen = set()
        for source_id in source_ids:
            adapter = self._adapter(source_id)
            try:
                urls = adapter.discovery_urls(keywords, environment)
            except BatchError as exc:
                failures.append({"source_id": source_id, "document_id": "", "reasons": [str(exc)]})
                continue
            for url in urls:
                try:
                    secured_url = adapter.request_url(url, environment)
                    response = controller.get(secured_url, adapter.request_spec(url), timeout, max_bytes)
                    for candidate in adapter.discover(response, url):
                        key = (candidate.document_id, candidate.canonical_url)
                        if key not in seen:
                            candidates.append(candidate)
                            seen.add(key)
                except BatchError as exc:
                    failures.append({"source_id": source_id, "document_id": "", "reasons": [str(exc)]})
        report = {
            "batch_id": batch_id, "mode": "discover", "request_count": controller.request_count,
            "counts": {"candidates": len(candidates), "failures": len(failures)}, "failures": failures,
        }
        atomic_write({catalog: _jsonl_bytes([asdict(item) for item in candidates]),
                      report_json: _json_bytes(report), report_md: markdown_report(report)}, replace_file)
        return report

    def execute(self, batch_id: str, source_ids: Sequence[str], catalog: Path, manifest: Path,
                extracted_dir: Path, report_json: Path, report_md: Path,
                timeout: float, max_bytes: int, limits: Mapping[str, Any], collected_at: str,
                environment: Optional[Mapping[str, str]] = None,
                replace_file: Callable[[str, str], None] = os.replace) -> Dict[str, Any]:
        self._validate_limits(limits)
        controller = self._controller()
        candidates = [BatchCandidate.from_dict(item) for item in read_jsonl(catalog)]
        candidates = [item for item in candidates if item.source_id in source_ids]
        existing_bytes = manifest.read_bytes() if manifest.exists() else b""
        existing_rows = read_jsonl(manifest)
        duplicates = DuplicateIndex(existing_rows)
        results = []  # type: List[CandidateResult]
        records = []  # type: List[Dict[str, Any]]
        output_files = {}  # type: Dict[Path, bytes]
        per_source = {}  # type: Dict[str, int]
        source_errors = {}  # type: Dict[str, int]
        max_accepted = int(limits["max_accepted"])
        max_per_source = int(limits["max_per_source"])
        environment = environment or {}
        for candidate in candidates:
            if len(records) >= max_accepted:
                break
            if per_source.get(candidate.source_id, 0) >= max_per_source:
                continue
            if source_errors.get(candidate.source_id, 0) >= 2:
                results.append(CandidateResult(candidate.document_id, candidate.source_id, candidate.title,
                                               "rejected_quality", ["source stopped after repeated errors"]))
                continue
            adapter = self._adapter(candidate.source_id)
            try:
                if candidate.discovery_metadata.get("access_policy") == "blocked":
                    results.append(CandidateResult(candidate.document_id, candidate.source_id, candidate.title,
                                                   "rejected_access_policy", ["explicit access policy prohibition"]))
                    continue
                request_url = adapter.detail_url(candidate, environment)
                response = controller.get(request_url, adapter.request_spec(request_url), timeout, max_bytes)
                detail = adapter.fetch_detail(candidate, response)
                result = quality_decision(candidate, detail.text)
                if result.decision in ACCEPTED_DECISIONS:
                    target = extracted_dir / (candidate.document_id + ".txt")
                    extracted = render_extracted(candidate, detail.text)
                    body_hash = hashlib.sha256(normalize_space(detail.text).encode("utf-8")).hexdigest()
                    extracted_hash = hashlib.sha256(extracted).hexdigest()
                    exact, warnings = duplicates.check(candidate, body_hash, extracted_hash, detail.text)
                    result.warnings.extend(warnings)
                    if exact:
                        result.decision = "rejected_duplicate"
                        result.reasons = exact
                    else:
                        record = build_manifest_record(candidate, detail, target, result.decision, collected_at)
                        records.append(record)
                        output_files[target] = extracted
                        duplicates.add_record(record)
                        duplicates.add_body(detail.text)
                        per_source[candidate.source_id] = per_source.get(candidate.source_id, 0) + 1
                results.append(result)
            except (BatchError, ValueError, ET.ParseError) as exc:
                source_errors[candidate.source_id] = source_errors.get(candidate.source_id, 0) + 1
                results.append(CandidateResult(candidate.document_id, candidate.source_id, candidate.title,
                                               "rejected_quality", [str(exc)]))
        counts = {decision: sum(1 for item in results if item.decision == decision) for decision in DECISIONS}
        counts["stored"] = len(records)
        report = {
            "batch_id": batch_id, "mode": "execute", "request_count": controller.request_count,
            "counts": counts, "results": [asdict(item) for item in results],
            "failures": [asdict(item) for item in results if item.decision not in ACCEPTED_DECISIONS],
        }
        manifest_data = existing_bytes
        if manifest_data and not manifest_data.endswith(b"\n"):
            manifest_data += b"\n"
        manifest_data += _jsonl_bytes(records)
        output_files[manifest] = manifest_data
        output_files[report_json] = _json_bytes(report)
        output_files[report_md] = markdown_report(report)
        atomic_write(output_files, replace_file)
        return report

    def _adapter(self, source_id: str) -> PublicSourceAdapter:
        if source_id not in self.adapters:
            raise BatchError("unsupported source: " + source_id)
        return self.adapters[source_id]

    def _controller(self) -> RequestController:
        if self.controller is None:
            raise BatchError("network controller is not configured")
        return self.controller

    @staticmethod
    def _validate_limits(limits: Mapping[str, Any]) -> None:
        accepted = int(limits["max_accepted"])
        per_source = int(limits["max_per_source"])
        requests = int(limits["max_requests"])
        delay = float(limits["delay_seconds"])
        if accepted < 1 or accepted > 10:
            raise BatchError("max-accepted cannot exceed 10")
        if per_source < 1 or per_source > 2:
            raise BatchError("max-per-source cannot exceed 2")
        if requests < 1 or requests > 500:
            raise BatchError("max-requests is invalid")
        if delay < 1.2:
            raise BatchError("delay-seconds must be at least 1.2")


def _raise_json_api_error(response: BatchResponse) -> None:
    value = json.loads(decode_body(response.body))
    if not isinstance(value, dict):
        raise ValueError("JSON response root must be an object")
    code, message = _json_api_error(value)
    if code and code not in ("0", "00", "0000"):
        category = "api_authentication_error" if code in ("10", "20", "30") else (
            "api_rate_limit_error" if code in ("22", "429") else "api_application_error"
        )
        raise ApiResponseError(category, api_code=code, api_message=_safe_api_message(message))


def _response_summary(source_id: str, stage: str, response: BatchResponse) -> Dict[str, Any]:
    media = validate_media_type(response.content_type, ())
    fields = []  # type: List[str]
    api_code = ""
    api_message = ""
    try:
        source = decode_body(response.body)
        if media in ("application/json", "text/json"):
            value = json.loads(source)
            if isinstance(value, dict):
                fields = sorted(str(key) for key in value.keys())[:30]
                api_code, api_message = _json_api_error(value)
        elif media in ("application/xml", "text/xml"):
            root = ET.fromstring(source)
            fields = sorted({_local_name(item.tag) for item in root.iter()})[:30]
            api_code, api_message = _api_error(root)
    except (ValueError, ET.ParseError, BatchError):
        fields = []
    return {
        "source_id": source_id, "stage": stage, "http_status": response.status,
        "response_format": media, "fields": fields,
        "encoding_ok": True, "api_result_code": api_code,
        "api_result_message": _safe_api_message(api_message),
    }
