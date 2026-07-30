"""도메인·robots·속도 제한을 강제하는 안전한 수집기 기반."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from abc import ABC
from dataclasses import asdict, dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from history_chatbot.ingestion.models import CopyrightStatus, ReviewStatus, SourceDocument


DEFAULT_USER_AGENT = "MokpoHistoryRAGCollector/0.1 (+non-commercial research prototype)"
TRACKING_PARAMETERS = {"fbclid", "gclid", "ref", "source", "utm_campaign", "utm_medium", "utm_source"}
ACCESS_BARRIER_PATTERNS = (
    re.compile(r"<input[^>]+type=[\"']password[\"']", re.IGNORECASE),
    re.compile(r"\b(?:captcha|recaptcha|hcaptcha|paywall)\b", re.IGNORECASE),
    re.compile(r"자동\s*입력\s*방지|유료\s*회원|구독\s*후\s*(?:이용|열람)"),
)


class CollectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FetchResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").lower()


class HttpTransport(Protocol):
    def request(self, url: str, *, timeout: float, user_agent: str) -> FetchResponse:
        """GET 요청을 수행한다."""


class UrllibTransport:
    def request(self, url: str, *, timeout: float, user_agent: str) -> FetchResponse:
        request = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return FetchResponse(
                    url=response.geturl(),
                    status=response.status,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=response.read(),
                )
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise CollectionError(f"네트워크 요청 실패: {url} ({error})") from error


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    source_id: str
    name: str
    collector_type: str
    base_url: str
    publisher: str
    trust_grade: str
    policy_url: str
    robots_url: str
    allowed_domains: tuple[str, ...]
    discovery_urls: tuple[str, ...]
    api_url: str = ""
    terms_url: str = "unknown"
    copyright_policy_url: str = "unknown"
    license_mark_location: str = "unknown"
    api_available: str = "unknown"
    api_docs_url: str = "unknown"
    api_auth_requirement: str = "unknown"
    collection_status: str = "manual_review"
    robots_verification: str = "unknown"
    audit_date: str = ""
    audit_notes: str = ""
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 10.0
    max_retries: int = 2
    max_pages: int = 2
    max_results: int = 20

    def __post_init__(self) -> None:
        if self.trust_grade not in {"A", "B", "C", "D"}:
            raise ValueError("trust_grade는 A, B, C, D 중 하나여야 합니다.")
        if self.api_available not in {"yes", "no", "unknown"}:
            raise ValueError("api_available은 yes, no, unknown 중 하나여야 합니다.")
        if self.collection_status not in {"allowed", "manual_review", "blocked", "unknown"}:
            raise ValueError(
                "collection_status는 allowed, manual_review, blocked, unknown 중 하나여야 합니다."
            )
        if not self.allowed_domains or not self.discovery_urls:
            raise ValueError("allowed_domains와 discovery_urls는 비어 있을 수 없습니다.")
        if not 1 <= self.max_pages <= 5:
            raise ValueError("max_pages는 1~5 범위여야 합니다.")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results는 1~100 범위여야 합니다.")
        if not 0.1 <= self.request_delay_seconds <= 60:
            raise ValueError("request_delay_seconds는 0.1~60초 범위여야 합니다.")
        if not 1 <= self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds는 1~60초 범위여야 합니다.")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries는 0~3 범위여야 합니다.")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CollectorConfig":
        values = dict(data)
        for name in ("allowed_domains", "discovery_urls"):
            values[name] = tuple(str(item) for item in values.get(name, []))
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CollectedCandidate:
    document_id: str
    source_id: str
    source_url: str
    title: str
    publisher: str
    published_date: str
    accessed_date: str
    language: str
    license_name: str
    license_url: str
    copyright_status: str
    allowed_for_rag: bool
    allowed_for_training: bool
    redistribution_allowed: bool
    trust_grade: str
    rag_priority_candidate: bool
    review_status: str
    raw_path: str
    extracted_path: str
    ocr_path: str
    content_sha256: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_source_document(self) -> SourceDocument:
        return SourceDocument(
            document_id=self.document_id,
            title=self.title,
            source_type=f"automatic_candidate:{self.source_id}",
            publisher=self.publisher,
            author="",
            source_url=self.source_url,
            local_path=self.raw_path,
            published_date=self.published_date,
            accessed_date=self.accessed_date,
            language=self.language,
            license_name=self.license_name,
            license_url=self.license_url,
            copyright_status=CopyrightStatus(self.copyright_status),
            allowed_for_rag=False,
            allowed_for_training=False,
            redistribution_allowed=False,
            attribution_required=False,
            attribution_text="",
            notes=self.notes,
            review_status=ReviewStatus.DRAFT,
            reviewed_by="",
            reviewed_at="",
            source_reliability=self.trust_grade,
            verification_notes="자동 수집 후보: 사실·권리·OCR 검수 전 사용 금지",
        )


@dataclass(frozen=True, slots=True)
class CollectionReport:
    candidates: tuple[CollectedCandidate, ...]
    errors: tuple[str, ...] = field(default_factory=tuple)


class _LinkAndTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._anchor_text: list[str] = []
        self.visible_text: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            self._href = attributes.get("href")
            self._anchor_text = []
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, _compact(" ".join(self._anchor_text))))
            self._href = None
            self._anchor_text = []
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        value = html.unescape(data)
        self.visible_text.append(value)
        if self._href is not None:
            self._anchor_text.append(value)
        if self._in_title:
            self.title_parts.append(value)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMETERS
        )
    )
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


class BaseCollector(ABC):
    """링크 탐색만 제공하며 출처별 모듈이 검색 진입점을 구성한다."""

    PRIORITY_KEYWORDS = (
        "목포 개항",
        "목포 해관",
        "외국인 거류지",
        "조계지",
        "근대역사문화공간",
        "구 일본영사관",
        "동양척식주식회사 목포지점",
        "근대 항만",
        "근대 철도",
        "일제강점기 목포",
        "독립운동",
        "종교",
        "교육",
        "의료",
        "목포 원도심",
    )

    def __init__(
        self,
        config: CollectorConfig,
        *,
        transport: HttpTransport | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibTransport()
        self.user_agent = user_agent
        self.sleep = sleep
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at = 0.0

    def collect(
        self, query: str, *, raw_dir: Path, extracted_dir: Path
    ) -> CollectionReport:
        skip_reason = collection_skip_reason(self.config)
        if skip_reason:
            return CollectionReport((), (skip_reason,))
        candidates: list[CollectedCandidate] = []
        errors: list[str] = []
        seen_urls: set[str] = set()
        result_limit = min(self.config.max_results, 2)
        discovery_urls = self.discovery_urls(query)[: self.config.max_pages]
        for discovery_url in discovery_urls:
            try:
                response = self._safe_fetch(discovery_url)
                for href, link_title in self._extract_links(response, query):
                    url = canonicalize_url(urljoin(response.url, href))
                    if url in seen_urls or not self.is_allowed_url(url):
                        continue
                    seen_urls.add(url)
                    try:
                        candidate = self._collect_detail(
                            url, link_title, raw_dir=raw_dir, extracted_dir=extracted_dir
                        )
                    except CollectionError as error:
                        errors.append(str(error))
                        continue
                    candidates.append(candidate)
                    if len(candidates) >= result_limit:
                        return CollectionReport(tuple(candidates), tuple(errors))
            except CollectionError as error:
                errors.append(str(error))
                break
        return CollectionReport(tuple(candidates), tuple(errors))

    def discovery_urls(self, query: str) -> list[str]:
        """공식 API가 설정되면 API를 우선하고, 아니면 제한된 진입 페이지만 조회한다."""
        if self.config.api_url:
            separator = "&" if "?" in self.config.api_url else "?"
            return [f"{self.config.api_url}{separator}query={urlencode({'q': query})[2:]}"]
        return list(self.config.discovery_urls)

    def is_allowed_url(self, url: str) -> bool:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        return (
            parts.scheme in {"http", "https"}
            and any(host == domain or host.endswith(f".{domain}") for domain in self.config.allowed_domains)
            and parts.username is None
            and parts.password is None
        )

    def _safe_fetch(self, url: str) -> FetchResponse:
        if not self.is_allowed_url(url):
            raise CollectionError(f"허용되지 않은 도메인입니다: {url}")
        if not self._robots_allows(url):
            raise CollectionError(f"robots.txt가 수집을 허용하지 않습니다: {url}")
        last_error: CollectionError | None = None
        for attempt in range(self.config.max_retries + 1):
            self._throttle()
            try:
                response = self.transport.request(
                    url,
                    timeout=self.config.timeout_seconds,
                    user_agent=self.user_agent,
                )
                self._last_request_at = time.monotonic()
                if not self.is_allowed_url(response.url):
                    raise CollectionError(f"허용되지 않은 도메인으로 리디렉션되었습니다: {response.url}")
                if response.status >= 400:
                    raise CollectionError(f"HTTP {response.status}: {url}")
                barrier = detect_access_barrier(response)
                if barrier:
                    raise CollectionError(f"접근 장벽 감지({barrier}): {response.url}")
                return response
            except CollectionError as error:
                last_error = error
                if attempt < self.config.max_retries:
                    self.sleep(min(2**attempt, 4))
        raise last_error or CollectionError(f"수집 실패: {url}")

    def _robots_allows(self, url: str) -> bool:
        host = urlsplit(url).netloc.lower()
        if host not in self._robots:
            if not self.is_allowed_url(self.config.robots_url):
                return False
            parser = RobotFileParser()
            parser.set_url(self.config.robots_url)
            try:
                self._throttle()
                response = self.transport.request(
                    self.config.robots_url,
                    timeout=self.config.timeout_seconds,
                    user_agent=self.user_agent,
                )
                self._last_request_at = time.monotonic()
            except CollectionError:
                return False
            if response.status >= 400:
                return False
            if detect_access_barrier(response):
                return False
            parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
            self._robots[host] = parser
        return self._robots[host].can_fetch(self.user_agent, url)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.request_delay_seconds - elapsed
        if self._last_request_at and remaining > 0:
            self.sleep(remaining)

    def _extract_links(self, response: FetchResponse, query: str) -> list[tuple[str, str]]:
        text = response.body.decode("utf-8", errors="replace")
        if "json" in response.content_type:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise CollectionError(f"공식 API JSON 해석 실패: {response.url}") from error
            return self._extract_json_links(payload, query)
        parser = _LinkAndTextParser()
        parser.feed(text)
        terms = {_compact(query).lower(), *(keyword.lower() for keyword in self.PRIORITY_KEYWORDS)}
        return [
            (href, title)
            for href, title in parser.links
            if title and any(term in title.lower() for term in terms if term)
        ]

    def _extract_json_links(self, payload: object, query: str) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        terms = {_compact(query).lower(), *(keyword.lower() for keyword in self.PRIORITY_KEYWORDS)}

        def walk(value: object) -> None:
            if isinstance(value, dict):
                url = next(
                    (value.get(key) for key in ("source_url", "detail_url", "url", "link") if value.get(key)),
                    None,
                )
                title = next(
                    (value.get(key) for key in ("title", "name", "subject") if value.get(key)),
                    None,
                )
                if isinstance(url, str) and isinstance(title, str):
                    if any(term in title.lower() for term in terms if term):
                        results.append((url, title))
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        walk(payload)
        return results

    def _collect_detail(
        self, url: str, fallback_title: str, *, raw_dir: Path, extracted_dir: Path
    ) -> CollectedCandidate:
        response = self._safe_fetch(url)
        identifier = hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()[:20]
        suffix = ".json" if "json" in response.content_type else ".html"
        source_raw_dir = raw_dir / self.config.source_id
        source_extracted_dir = extracted_dir / self.config.source_id
        source_raw_dir.mkdir(parents=True, exist_ok=True)
        source_extracted_dir.mkdir(parents=True, exist_ok=True)
        raw_path = source_raw_dir / f"{identifier}{suffix}"
        raw_path.write_bytes(response.body)

        parser = _LinkAndTextParser()
        decoded = response.body.decode("utf-8", errors="replace")
        if "json" in response.content_type:
            extracted_text = json.dumps(json.loads(decoded), ensure_ascii=False, indent=2)
            title = fallback_title or url
        else:
            parser.feed(decoded)
            extracted_text = _compact(" ".join(parser.visible_text))
            title = _compact(" ".join(parser.title_parts)) or fallback_title or url
        extracted_path = source_extracted_dir / f"{identifier}.txt"
        extracted_path.write_text(extracted_text, encoding="utf-8", newline="\n")
        return CollectedCandidate(
            document_id=f"auto-{self.config.source_id}-{identifier}",
            source_id=self.config.source_id,
            source_url=canonicalize_url(url),
            title=title,
            publisher=self.config.publisher,
            published_date="",
            accessed_date=date.today().isoformat(),
            language="ko",
            license_name="",
            license_url=self.config.policy_url,
            copyright_status=CopyrightStatus.UNKNOWN.value,
            allowed_for_rag=False,
            allowed_for_training=False,
            redistribution_allowed=False,
            trust_grade=self.config.trust_grade,
            rag_priority_candidate=self.config.trust_grade in {"A", "B"},
            review_status=ReviewStatus.DRAFT.value,
            raw_path=str(raw_path),
            extracted_path=str(extracted_path),
            ocr_path="",
            content_sha256=hashlib.sha256(response.body).hexdigest(),
            notes="자동 수집 후보입니다. 역사 사실과 저작권을 사람이 검수하기 전 사용 금지.",
        )


def load_collector_configs(path: Path) -> list[CollectorConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CollectorConfig.from_dict(item) for item in payload["sources"]]


def collection_skip_reason(config: CollectorConfig) -> str | None:
    if config.collection_status != "allowed":
        return f"수집 건너뜀: collection_status={config.collection_status}"
    if config.robots_verification != "verified":
        return f"수집 건너뜀: robots_verification={config.robots_verification}"
    return None


def detect_access_barrier(response: FetchResponse) -> str | None:
    path = urlsplit(response.url).path.lower()
    if any(token in path for token in ("/login", "/signin", "/captcha", "/paywall")):
        return "로그인·캡차·유료벽 URL"
    if "html" not in response.content_type:
        return None
    text = response.body[:1_000_000].decode("utf-8", errors="replace")
    if ACCESS_BARRIER_PATTERNS[0].search(text):
        return "로그인 비밀번호 입력"
    if ACCESS_BARRIER_PATTERNS[1].search(text):
        return "캡차 또는 유료벽"
    if ACCESS_BARRIER_PATTERNS[2].search(text):
        return "자동입력 방지 또는 유료 열람"
    return None
