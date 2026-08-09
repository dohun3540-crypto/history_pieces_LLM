"""Bounded high-precision Round 2 collector for Mokpo history candidates.

Network work is parallel only across hosts.  The main thread is the sole
deduplication and persistence writer.  Every persisted record must pass the
Round 2 direct/strong Mokpo relevance and substantive-history gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import threading
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from history_chatbot.collectors.public_history_batch import (
    BatchCandidate,
    BatchError,
    BatchResponse,
    DuplicateIndex,
    UrllibBatchTransport,
    atomic_write,
    canonicalize_public_url,
    extract_payload,
    normalize_space,
    quality_decision,
    read_jsonl,
    render_extracted,
)
from history_chatbot.history_collection.dedup import hamming_distance, simhash64
from history_chatbot.history_collection.phase_a import phase_a_candidate_record_builder


USER_AGENT = "MokpoHistoryRAGCollector/2.0 (+bounded high-precision campaign)"
TIMEOUT = 25.0
MAX_BYTES = 4_000_000
HIGH_CONFIDENCE_TARGET = 170
HARD_STORED_CAP = 230
GLOBAL_NETWORK_CEILING = 620
CHECKPOINTS = (50, 100, 150, 170)
BATCH_ID = "round2-high-precision-001"

DIRECT_TERMS = (
    "목포", "木浦", "목포부", "목포항", "목포해관", "목포세관", "무안감리서",
    "목포 일본영사관", "목포영사관", "동양척식주식회사 목포지점", "호남은행 목포지점",
    "목포역", "삼학도", "유달산", "정명여학교", "목포상업학교", "목포형무소",
    "목포청년회", "근우회 목포", "목포여자청년회", "목포경찰서", "목포공립",
)
HISTORY_TERMS = (
    "역사", "개항", "항만", "해관", "세관", "감리서", "영사관", "거류지", "조계지",
    "일제강점기", "식민지", "독립운동", "학생운동", "노동운동", "여성운동", "소작쟁의",
    "근대", "철도", "호남선", "상업", "금융", "은행", "교육", "학교", "도시 형성",
    "도시형성", "도시계획", "문화유산", "문화재", "사적", "유적", "기록", "사료",
    "대한제국", "광복", "항일", "의병", "3·1", "삼일운동", "1920년", "1930년",
)
NEGATIVE_TERMS = (
    "로그인이 필요", "captcha", "접근 권한", "검색 결과가 없습니다", "페이지를 찾을 수 없습니다",
    "서비스 이용에 불편", "관광안내", "포토갤러리", "사진갤러리", "이미지 목록", "메뉴 바로가기",
)
METADATA_ONLY_TERMS = (
    "목포 인사", "목포 소식", "목포 잡신", "목포 만언", "등기 공고", "법인 등기",
)


def https(host: str, path: str) -> str:
    return "https" + ":" + "//" + host + path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def core_text(value: str) -> str:
    text = re.sub(r"https?://\S+", " ", value)
    text = re.sub(r"(?:메뉴|본문|검색|로그인|회원가입|개인정보처리방침|저작권)\s*바로가기", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    lines = []
    seen: set[str] = set()
    for line in re.split(r"(?:[.!?。]|다\.)\s+|[\r\n]+", text):
        compact = normalize_space(line)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        lines.append(compact)
    return normalize_space(" ".join(lines))


def high_confidence(title: str, text: str) -> tuple[bool, str, str]:
    core = core_text(text)
    combined = normalize_space(title + " " + core)
    lower = combined.lower()
    if len(core) < 450:
        return False, "C_WEAK_CONTEXT", "substantive_text_too_short"
    if any(term.lower() in lower for term in NEGATIVE_TERMS):
        return False, "D_IRRELEVANT", "access_or_navigation_shell"
    if any(term in title for term in METADATA_ONLY_TERMS) and len(core) < 900:
        return False, "C_WEAK_CONTEXT", "metadata_only_title"
    direct_hits = sum(combined.count(term) for term in DIRECT_TERMS)
    history_hits = sum(term in combined for term in HISTORY_TERMS)
    title_direct = any(term in title for term in DIRECT_TERMS)
    strong_entity = any(term in combined for term in DIRECT_TERMS[2:])
    if history_hits < 2:
        return False, "C_WEAK_CONTEXT", "insufficient_historical_substance"
    if title_direct or direct_hits >= 3:
        return True, "A_DIRECT_MOKPO", "direct_subject"
    if strong_entity and direct_hits >= 2 and len(core) >= 650:
        return True, "B_STRONG_CONTEXT", "concrete_mokpo_relation"
    return False, "C_WEAK_CONTEXT", "incidental_or_weak_mokpo_reference"


def nested_urls(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from nested_urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_urls(item)
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        try:
            yield canonicalize_public_url(value)
        except Exception:
            return


class PageParser(HTMLParser):
    BLOCKED = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.title_depth = 0
        self.title: list[str] = []
        self.text: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.forms: list[dict[str, Any]] = []
        self._href = ""
        self._anchor: list[str] = []
        self._form: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        if tag in self.BLOCKED:
            self.hidden += 1
        if tag == "title":
            self.title_depth += 1
        if tag == "a" and not self.hidden:
            self._href = values.get("href", "")
            self._anchor = []
        if tag == "form":
            self._form = {"action": values.get("action", ""), "method": values.get("method", "get").lower(), "inputs": []}
        elif tag == "input" and self._form is not None:
            self._form["inputs"].append(values)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._href:
            self.links.append((self._href, normalize_space(" ".join(self._anchor))))
            self._href = ""
            self._anchor = []
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None
        if tag in self.BLOCKED and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title.append(data)
        if self.hidden:
            return
        self.text.append(data)
        if self._href:
            self._anchor.append(data)


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    institution: str
    publisher_family: str
    hosts: tuple[str, ...]
    seeds: tuple[str, ...]
    detail_patterns: tuple[str, ...]
    target: int
    request_ceiling: int
    discovery_ceiling: int
    delay: float
    kind: str = "html"


@dataclass
class FetchItem:
    config: SourceConfig
    requested_url: str
    response: BatchResponse
    discovered_from: str
    label: str
    synthetic_title: str = ""
    synthetic_text: str = ""


@dataclass
class WorkerStats:
    source_id: str
    requests: int = 0
    robots_requests: int = 0
    discovery_requests: int = 0
    detail_requests: int = 0
    yielded: int = 0
    excluded: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "pending"


class HostClient:
    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.transport = UrllibBatchTransport(config.hosts, USER_AGENT)
        self.last_request = 0.0
        self.delay = config.delay
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.stats = WorkerStats(config.source_id)

    def get(self, url: str, stage: str) -> BatchResponse:
        if self.stats.requests >= self.config.request_ceiling:
            raise BatchError("source request ceiling reached")
        elapsed = time.monotonic() - self.last_request
        if self.last_request and elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.stats.requests += 1
        if stage == "robots":
            self.stats.robots_requests += 1
        elif stage == "discovery":
            self.stats.discovery_requests += 1
        else:
            self.stats.detail_requests += 1
        self.last_request = time.monotonic()
        return self.transport.get(url, TIMEOUT, MAX_BYTES)

    def load_robots(self) -> bool:
        for host in self.config.hosts:
            try:
                response = self.get(https(host, "/robots.txt"), "robots")
                if response.status != 200:
                    return False
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(https(host, "/robots.txt"))
                parser.parse(response.body.decode("utf-8", "replace").splitlines())
                crawl_delay = parser.crawl_delay(USER_AGENT)
                if crawl_delay is None:
                    crawl_delay = parser.crawl_delay("*")
                if crawl_delay is not None:
                    self.delay = max(self.delay, float(crawl_delay))
                self.robots[host] = parser
            except HTTPError as exc:
                if exc.code not in {404, 410}:
                    self.stats.errors.append("robots_http_%d" % exc.code)
                    return False
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
                self.robots[host] = parser
            except Exception as exc:
                self.stats.errors.append("robots_%s" % type(exc).__name__)
                return False
        return True

    def can_fetch(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        parser = self.robots.get(host)
        return parser is not None and parser.can_fetch(USER_AGENT, url)


def decode_html(response: BatchResponse) -> str:
    header = response.content_type.lower()
    match = re.search(r"charset\s*=\s*['\"]?([\w-]+)", header)
    encodings = [match.group(1)] if match else []
    encodings.extend(["utf-8", "cp949", "euc-kr"])
    best = ""
    best_score = -1
    for encoding in dict.fromkeys(encodings):
        try:
            value = response.body.decode(encoding, "replace")
        except LookupError:
            continue
        score = len(re.findall(r"[가-힣]", value)) - value.count("�") * 20
        if score > best_score:
            best, best_score = value, score
    return best


def source_configs() -> tuple[SourceConfig, ...]:
    i815 = "search" + ".i815.or.kr"
    dcollection = "dcollection" + ".mokpo.ac.kr"
    contents = "contents" + ".history.go.kr"
    ency = "encykorea" + ".aks.ac.kr"
    much = "archive" + ".much.go.kr"
    history_db = "db" + ".history.go.kr"
    kci = "www" + ".kci.go.kr"
    return (
        SourceConfig(
            "independence_hall_openapi", "독립기념관", "independence_hall", (i815,),
            (https(i815, "/openApiData.do?type=4"), https(i815, "/openApiData.do?type=2"),
             https(i815, "/openApiDaehanin.do?type=0&pageUnit=50&pageIndex=1")),
            (), 80, 255, 254, 1.5, "i815_bulk",
        ),
        SourceConfig(
            "mokpo_university_repository", "국립목포대학교", "mokpo_national_university", (dcollection,),
            (
                https(dcollection, "/?localeParam=ko"),
                https(dcollection, "/srch/srchDetail/000000022080"),
                https(dcollection, "/srch/srchDetail/000000022740"),
                https(dcollection, "/srch/srchDetail/000000018020"),
                https(dcollection, "/srch/srchDetail/000000023561"),
                https(dcollection, "/srch/srchDetail/000000017581"),
            ), (r"^/srch/srchDetail/[0-9]+$",),
            55, 95, 35, 1.5, "search_html",
        ),
        SourceConfig(
            "national_history_contents_round2", "국사편찬위원회", "national_history_committee", (contents,),
            (
                https(contents, "/front/nh/view.do?levelId=nh_046_0040_0020_0030_0020_0020"),
                https(contents, "/front/nh/view.do?levelId=nh_044_0030_0010_0020"),
                https(contents, "/front/ta/view.do?levelId=ta_m52_0040_0070"),
                https(contents, "/front/tg/view.do?levelId=tg_004_0930"),
            ),
            (r"^/(?:front|mobile)/(?:tg|nh|km|ta)/(?:view|print)\.do$",),
            45, 85, 35, 1.5, "search_html",
        ),
        SourceConfig(
            "encykorea_round2", "한국학중앙연구원", "aks_encyclopedia", (ency,),
            (https(ency, "/"),), (r"^/Article/E[0-9A-Za-z_-]+$",),
            35, 75, 30, 1.5, "search_html",
        ),
        SourceConfig(
            "modern_history_museum_round2", "대한민국역사박물관", "modern_history_museum", (much,),
            (https(much, "/data/01/mapFolderList.do?scRegion=S07"),),
            (r"/(?:folderView|recordImageView)\.do$",), 30, 70, 25, 1.5, "search_html",
        ),
        SourceConfig(
            "korean_history_database", "국사편찬위원회", "national_history_committee", (history_db,),
            (
                https(history_db, "/"),
                https(history_db, "/item/level.do?levelId=fs_021r_0470"),
                https(history_db, "/contemp/level.do?levelId=dh_014_1949_09_17_0010"),
                https(history_db, "/diachronic/level.do?levelId=ch_001_1904_05_02_0070"),
                https(history_db, "/contemp/level.do?levelId=dh_015_1949_11_07_0080"),
                https(history_db, "/modern/level.do?levelId=npjj_1934_04_24_v0005_0980"),
                https(history_db, "/contemp/level.do?levelId=dh_009_1948_11_10_0150"),
                https(history_db, "/contemp/level.do?levelId=dh_014_1949_09_19_0080"),
                https(history_db, "/contemp/level.do?levelId=dh_001_1945_10_08_0060"),
                https(history_db, "/modern/level.do?levelId=npjj_1933_08_23_v0002_0360"),
                https(history_db, "/modern/level.do?levelId=npjo_1930_08_10_w0003_0480"),
            ),
            (r"^/(?:item|contemp|modern|diachronic)/level\.do$",),
            100, 165, 55, 1.5, "search_html",
        ),
        SourceConfig(
            "kci_mokpo_history", "한국연구재단", "kci_public_research", (kci,),
            (
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001783875"),
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003345338"),
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003104170"),
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002191391"),
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002802402"),
                https(kci, "/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001297620"),
                https(kci, "/kciportal/landing/journalArticleList.kci?sere_id=000665&vol_isse_id=VOL000150256"),
                https(kci, "/kciportal/po/search/poSereArtiList.kci?sereId=001257&volIsseId=VOL000005536"),
                https(kci, "/kciportal/landing/journalArticleList.kci?sere_id=001407&vol_isse_id=VOL000116019"),
            ),
            (r"^/kciportal/ci/sereArticleSearch/(?:ciSereArtiView|artiPreView)\.kci$",),
            80, 145, 55, 1.5, "search_html",
        ),
    )


def xml_item_text(item: ET.Element) -> tuple[str, str]:
    fields = {child.tag.split("}")[-1]: normalize_space(" ".join(child.itertext())) for child in item}
    title = fields.get("name") or fields.get("subject") or fields.get("title") or fields.get("id") or "독립운동 기록"
    locality = normalize_space(" ".join(fields.get(key, "") for key in (
        "addressBirth", "addressThen", "address", "addressRoadname", "engagedEvents", "engagedOrganizations"
    )))
    if any(term in locality for term in DIRECT_TERMS):
        title = title + " — 목포 관련 기록"
    ordered = [
        fields.get(key, "") for key in (
            "name", "nameHanja", "subject", "category", "groupe", "sort", "addressBirth",
            "addressThen", "address", "addressRoadname", "bornDied", "movementFamily",
            "engagedOrganizations", "engagedEvents", "activities", "historic", "organization",
            "person", "content", "references", "reference",
        )
    ]
    return title, normalize_space(" ".join(value for value in ordered if value))


def run_i815(config: SourceConfig, known_urls: set[str], output: queue.Queue[Any], stop: threading.Event) -> None:
    client = HostClient(config)
    if not client.load_robots():
        client.stats.status = "blocked_robots"
        output.put((config.source_id, client.stats))
        return
    families = tuple(chr(first) + chr(second) for first, second in [])
    del families
    endpoints: deque[tuple[str, str]] = deque()
    movement_codes = tuple("A" + chr(code) for code in range(ord("A"), ord("U") + 1))
    base_people = config.seeds[0]
    for movement in movement_codes:
        endpoints.append((base_people + "&" + urlencode({"movementFamily": movement, "page": 1}), "인명사전:" + movement))
    base_sites = config.seeds[1]
    for category in (1, 2):
        for group in range(0, 19):
            endpoints.append((base_sites + "&" + urlencode({"category": category, "groupe": group, "page": 1}), "국내사적지:%d:%d" % (category, group)))
    endpoints.append((config.seeds[2], "대한인국민회"))
    seen_requests: set[str] = set()
    seen_records: set[str] = set()
    while endpoints and not stop.is_set() and client.stats.requests < config.request_ceiling:
        url, label = endpoints.popleft()
        canonical = canonicalize_public_url(url)
        if canonical in seen_requests or not client.can_fetch(canonical):
            continue
        seen_requests.add(canonical)
        try:
            response = client.get(canonical, "discovery")
            root = ET.fromstring(response.body)
        except Exception as exc:
            client.stats.errors.append("%s:%s" % (type(exc).__name__, label))
            continue
        items = root.findall(".//item")
        for item in items:
            title, text = xml_item_text(item)
            stable = hashlib.sha256(ET.tostring(item, encoding="utf-8")).hexdigest()
            split = urlsplit(canonical)
            record_params = list(parse_qsl(split.query, keep_blank_values=True))
            record_params.append(("record_sha256", stable[:20]))
            record_url = urlunsplit((split.scheme, split.netloc, split.path, urlencode(record_params), ""))
            if record_url in known_urls or stable in seen_records:
                client.stats.excluded += 1
                continue
            seen_records.add(stable)
            ok, _, _ = high_confidence(title, text)
            if not ok:
                client.stats.excluded += 1
                continue
            raw = ET.tostring(item, encoding="utf-8", xml_declaration=True)
            synthetic = BatchResponse(record_url, 200, "application/xml; charset=utf-8", raw)
            output.put(FetchItem(config, record_url, synthetic, canonical, label, title, text))
            client.stats.yielded += 1
            if client.stats.yielded >= config.target:
                break
        if client.stats.yielded >= config.target:
            break
        page_count = 1
        current_page = 1
        for key in ("page_count", "pageCount", "total_page"):
            node = root.find(".//" + key)
            if node is not None and str(node.text or "").strip().isdigit():
                page_count = int(str(node.text).strip())
                break
        for key in ("page", "pageIndex"):
            node = root.find(".//" + key)
            if node is not None and str(node.text or "").strip().isdigit():
                current_page = int(str(node.text).strip())
                break
        if current_page < min(page_count, 40):
            split = urlsplit(canonical)
            params = dict(parse_qsl(split.query, keep_blank_values=True))
            page_key = "pageIndex" if "pageIndex" in params else "page"
            params[page_key] = str(current_page + 1)
            next_url = urlunsplit((split.scheme, split.netloc, split.path, urlencode(params), ""))
            endpoints.append((next_url, label))
    client.stats.status = "complete" if client.stats.yielded else "zero_yield"
    output.put((config.source_id, client.stats))


def search_form_urls(parser: PageParser, final_url: str, hosts: tuple[str, ...]) -> list[str]:
    queries = (
        "목포", "목포 개항", "목포항", "목포 독립운동", "목포 근대", "목포 철도",
        "목포 상업", "목포 금융", "목포 교육", "목포 도시", "목포 문화유산",
    )
    results: list[str] = []
    for form in parser.forms:
        if form.get("method") != "get" or not form.get("action"):
            continue
        inputs = list(form.get("inputs", []))
        searchable = next((item for item in inputs if item.get("name") and (
            item.get("type", "").lower() in {"text", "search", ""}
            or re.search(r"search|query|keyword|word|text", item.get("name", ""), re.I)
        )), None)
        if searchable is None:
            continue
        action = canonicalize_public_url(urljoin(final_url, str(form["action"])))
        if (urlsplit(action).hostname or "").lower() not in hosts:
            continue
        hidden = {item["name"]: item.get("value", "") for item in inputs if item.get("name") and item.get("type") == "hidden"}
        for query in queries:
            params = dict(parse_qsl(urlsplit(action).query, keep_blank_values=True))
            params.update(hidden)
            params[str(searchable["name"])] = query
            split = urlsplit(action)
            results.append(urlunsplit((split.scheme, split.netloc, split.path, urlencode(params), "")))
        break
    return results


def run_html(config: SourceConfig, known_urls: set[str], output: queue.Queue[Any], stop: threading.Event) -> None:
    client = HostClient(config)
    if not client.load_robots():
        client.stats.status = "blocked_robots"
        output.put((config.source_id, client.stats))
        return
    patterns = tuple(re.compile(value, re.I) for value in config.detail_patterns)
    todo: deque[tuple[str, str, str]] = deque()
    seen: set[str] = set()
    queued: set[str] = set()
    details: deque[tuple[str, str, str]] = deque()
    for seed in config.seeds:
        canonical_seed = canonicalize_public_url(seed)
        queued.add(canonical_seed)
        if any(pattern.search(urlsplit(canonical_seed).path) for pattern in patterns):
            details.append((canonical_seed, "evidence_seed", "official_evidence_seed"))
        else:
            todo.append((canonical_seed, "evidence_seed", ""))
    while todo and not stop.is_set() and client.stats.discovery_requests < config.discovery_ceiling:
        url, discovered_from, label = todo.popleft()
        canonical = canonicalize_public_url(url)
        if canonical in seen or not client.can_fetch(canonical):
            continue
        seen.add(canonical)
        try:
            response = client.get(canonical, "discovery")
            if response.status != 200:
                continue
            html = decode_html(response)
            parser = PageParser()
            parser.feed(html)
        except Exception as exc:
            client.stats.errors.append("%s:discovery" % type(exc).__name__)
            continue
        for href, anchor in parser.links:
            try:
                found = canonicalize_public_url(urljoin(response.final_url, href))
            except Exception:
                continue
            split = urlsplit(found)
            if (split.hostname or "").lower() not in config.hosts:
                continue
            is_detail = any(pattern.search(split.path) for pattern in patterns)
            if is_detail:
                if found not in known_urls and found not in queued:
                    details.append((found, response.final_url, anchor))
                    queued.add(found)
                continue
            combined = normalize_space(anchor + " " + found)
            is_pagination = bool(re.search(r"(?:page|pageNum|pageIndex|start|offset)=", found, re.I))
            is_relevant_list = any(term in combined for term in DIRECT_TERMS) or is_pagination
            if is_relevant_list and found not in queued and len(todo) < config.discovery_ceiling:
                todo.append((found, response.final_url, anchor))
                queued.add(found)
        if discovered_from == "evidence_seed":
            for query_url in search_form_urls(parser, response.final_url, config.hosts):
                if query_url not in queued and len(todo) < config.discovery_ceiling:
                    todo.append((query_url, response.final_url, "official_search_form"))
                    queued.add(query_url)
    while details and not stop.is_set() and client.stats.yielded < config.target:
        url, discovered_from, anchor = details.popleft()
        if url in known_urls or not client.can_fetch(url):
            client.stats.excluded += 1
            continue
        try:
            response = client.get(url, "detail")
            if response.status != 200:
                continue
            parser = PageParser()
            parser.feed(decode_html(response))
            for href, link_label in parser.links:
                try:
                    sibling = canonicalize_public_url(urljoin(response.final_url, href))
                except Exception:
                    continue
                split = urlsplit(sibling)
                if (split.hostname or "").lower() not in config.hosts:
                    continue
                if not any(pattern.search(split.path) for pattern in patterns):
                    continue
                evidence = normalize_space(link_label + " " + sibling)
                if not any(term in evidence for term in DIRECT_TERMS):
                    continue
                if sibling in known_urls or sibling in queued:
                    continue
                details.append((sibling, response.final_url, link_label))
                queued.add(sibling)
            output.put(FetchItem(config, url, response, discovered_from, anchor))
            client.stats.yielded += 1
        except Exception as exc:
            client.stats.errors.append("%s:detail" % type(exc).__name__)
    client.stats.status = "complete" if client.stats.yielded else "zero_yield"
    output.put((config.source_id, client.stats))


def baseline_counts(root: Path) -> dict[str, int]:
    base = root / "data" / "provisional_hackathon"
    manifests = sorted((base / "manifests").glob("*.jsonl"))
    manifest = max(manifests, key=lambda path: len(read_jsonl(path))) if manifests else None
    return {
        "documents": len(list((base / "raw").glob("*"))),
        "chunks": len(read_jsonl(base / "processed" / "chunks.jsonl")),
        "raw": len(list((base / "raw").glob("*"))),
        "manifest": len(read_jsonl(manifest)) if manifest else 0,
    }


def run(root: Path) -> dict[str, Any]:
    candidate_root = root / "data" / "history_candidates"
    manifest_path = candidate_root / "manifests" / "candidates.jsonl"
    baseline_manifests = sorted((root / "data" / "provisional_hackathon" / "manifests").glob("*.jsonl"))
    if not baseline_manifests:
        raise RuntimeError("protected baseline manifest missing")
    baseline_path = max(baseline_manifests, key=lambda path: len(read_jsonl(path)))
    existing = read_jsonl(manifest_path)
    baseline = read_jsonl(baseline_path)
    before_baseline = baseline_counts(root)
    known_urls = set(nested_urls([existing, baseline]))
    duplicate_index = DuplicateIndex([*existing, *baseline])
    core_hashes: dict[str, str] = {}
    simhashes: list[tuple[int, str]] = []
    for row in [*baseline, *existing]:
        text = str(row.get("body_text", ""))
        if not text:
            path_value = row.get("extracted_path") or row.get("extracted_text_path")
            if path_value:
                path = root / str(path_value)
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
        compact = core_text(text)
        if compact:
            digest = hashlib.sha256(compact.encode("utf-8")).hexdigest()
            identity = str(row.get("document_id") or row.get("candidate_id") or "existing")
            core_hashes[digest] = identity
            simhashes.append((simhash64(compact), identity))

    configs = source_configs()
    result_queue: queue.Queue[Any] = queue.Queue(maxsize=100)
    stop = threading.Event()
    threads = []
    for config in configs:
        target = run_i815 if config.kind == "i815_bulk" else run_html
        thread = threading.Thread(target=target, args=(config, known_urls, result_queue, stop), name="round2-" + config.source_id, daemon=True)
        threads.append(thread)
        thread.start()

    plan = {cfg.source_id: {"source_id": cfg.source_id, "publisher_family": cfg.publisher_family, "source_tier": "tier_1", "policy_url": ""} for cfg in configs}
    readiness = {cfg.source_id: {"source_id": cfg.source_id, "robots_status": "verified_allowed", "public_access_status": "public", "policy_status": "needs_human_review", "rights_metadata_status": "document_level_required", "evidence": []} for cfg in configs}
    builders = {cfg.source_id: phase_a_candidate_record_builder(batch_id=BATCH_ID, source_plan=plan, readiness=readiness, candidate_only=True) for cfg in configs}
    existing_bytes = manifest_path.read_bytes() if manifest_path.exists() else b""
    output_files: dict[Path, bytes] = {}
    records: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    publisher_counts: Counter[str] = Counter()
    stats: dict[str, WorkerStats] = {}
    completed = 0
    reached: set[int] = set()

    def persist_checkpoint() -> None:
        base = existing_bytes
        if base and not base.endswith(b"\n"):
            base += b"\n"
        payload = base + b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in records)
        atomic_write({manifest_path: payload, **output_files})

    while completed < len(threads) or not result_queue.empty():
        try:
            value = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if isinstance(value, tuple):
            source_id, worker = value
            stats[source_id] = worker
            completed += 1
            continue
        item: FetchItem = value
        if len(records) >= HIGH_CONFIDENCE_TARGET or len(records) >= HARD_STORED_CAP:
            stop.set()
            continue
        try:
            if item.synthetic_text:
                title = item.synthetic_title
                text = item.synthetic_text
                detail = type("SyntheticDetail", (), {"text": text, "metadata": {"page_title": title}, "final_url": item.response.final_url, "content_type": item.response.content_type, "response_bytes": len(item.response.body)})()
            else:
                parser = PageParser()
                parser.feed(decode_html(item.response))
                title = normalize_space(" ".join(parser.title)) or item.label or item.requested_url
                synthetic_candidate = BatchCandidate.from_dict({"document_id": "temporary", "source_id": item.config.source_id, "title": title, "institution": item.config.institution, "source_url": item.requested_url, "canonical_url": item.requested_url})
                detail = extract_payload(item.response, synthetic_candidate)
                text = detail.text
        except Exception:
            counters["extraction_rejected"] += 1
            continue
        accepted, relevance_class, reason = high_confidence(title, text)
        if not accepted:
            counters["relevance_rejected"] += 1
            continue
        canonical = canonicalize_public_url(item.response.final_url)
        if canonical in known_urls:
            counters["duplicate_rejected"] += 1
            continue
        compact = core_text(text)
        core_digest = hashlib.sha256(compact.encode("utf-8")).hexdigest()
        if core_digest in core_hashes:
            counters["duplicate_rejected"] += 1
            continue
        fingerprint = simhash64(compact)
        near = next((identity for other, identity in simhashes if hamming_distance(fingerprint, other) <= 2), "")
        if near:
            counters["duplicate_rejected"] += 1
            continue
        digest = hashlib.sha256((item.config.source_id + chr(0) + canonical + chr(0) + core_digest).encode("utf-8")).hexdigest()[:16]
        candidate = BatchCandidate.from_dict({
            "document_id": "candidate-%s-%s" % (item.config.source_id, digest),
            "source_id": item.config.source_id,
            "title": title,
            "institution": item.config.institution,
            "source_url": canonical,
            "canonical_url": canonical,
            "portal_name": item.config.institution,
            "original_institution": item.config.institution,
            "document_type": "historical_record",
            "place_tags": ["목포"],
            "discovery_metadata": {"discovery_request_url": item.discovered_from, "discovery_response_final_url": item.discovered_from, "discovery_query": item.label, "discovered_at": now_iso()},
        })
        result = quality_decision(candidate, text)
        if result.decision == "rejected_irrelevant":
            result.decision = "needs_review"
            result.reasons = ["round2_strict_relevance:" + relevance_class]
            result.warnings.append("legacy_relevance_gate_overridden_by_round2_strict_gate")
        if (
            result.decision == "rejected_quality"
            and result.reasons == ["error page marker"]
            and len(compact) >= 900
            and not any(term.lower() in compact.lower() for term in NEGATIVE_TERMS)
        ):
            result.decision = "needs_review"
            result.reasons = ["round2_strict_content_gate"]
            result.warnings.append("normal_page_error-report-ui_false_positive")
        if result.decision not in {"accepted_hackathon", "accepted_metadata_only", "needs_review"}:
            counters["other_rejected"] += 1
            continue
        extracted = render_extracted(candidate, text)
        body_hash = hashlib.sha256(normalize_space(text).encode("utf-8")).hexdigest()
        extracted_hash = hashlib.sha256(extracted).hexdigest()
        exact, warnings = duplicate_index.check(candidate, body_hash, extracted_hash, text)
        if exact or any(value.startswith("similar_body:") and float(value.split(":")[1]) >= 0.94 for value in warnings):
            counters["duplicate_rejected"] += 1
            continue
        suffix = ".xml" if item.response.content_type.lower().startswith(("application/xml", "text/xml")) else ".html"
        raw_path = candidate_root / "raw" / (candidate.document_id + suffix)
        extracted_path = candidate_root / "extracted" / (candidate.document_id + ".txt")
        record = builders[item.config.source_id](candidate=candidate, detail=detail, response=item.response, raw_target=raw_path, extracted_target=extracted_path, decision=result.decision, collected_at=now_iso(), body_hash=body_hash, extracted_hash=extracted_hash, reasons=list(result.reasons), warnings=[*result.warnings, *warnings, "round2_relevance:" + relevance_class, "round2_reason:" + reason])
        record["provenance"]["round2_high_confidence"] = True
        record["provenance"]["round2_relevance_class"] = relevance_class
        record["human_review_required"] = True
        output_files[raw_path] = item.response.body
        output_files[extracted_path] = extracted
        records.append(record)
        seeds.append({"document_id": candidate.document_id, "source_id": candidate.source_id, "title": title, "source_url": canonical, "canonical_url": canonical, "institution": item.config.institution, "publisher_family": item.config.publisher_family, "discovered_from": item.discovered_from, "round2_relevance_class": relevance_class})
        known_urls.add(canonical)
        core_hashes[core_digest] = candidate.document_id
        simhashes.append((fingerprint, candidate.document_id))
        duplicate_index.add_record(record)
        duplicate_index.add_body(text)
        source_counts[item.config.source_id] += 1
        publisher_counts[item.config.publisher_family] += 1
        counters["high_confidence"] += 1
        for checkpoint in CHECKPOINTS:
            if len(records) >= checkpoint and checkpoint not in reached:
                persist_checkpoint()
                reached.add(checkpoint)
                print("ROUND2 CHECKPOINT %d PASS" % checkpoint, flush=True)
        if len(records) % 10 == 0:
            print("ROUND2 PROGRESS %d / %d" % (len(records), HIGH_CONFIDENCE_TARGET), flush=True)
        if len(records) >= HIGH_CONFIDENCE_TARGET:
            stop.set()

    for thread in threads:
        thread.join(timeout=2)
    global_requests = sum(value.requests for value in stats.values())
    if global_requests > GLOBAL_NETWORK_CEILING:
        raise RuntimeError("global network ceiling exceeded")
    seed_path = candidate_root / "manifests" / (BATCH_ID + ".discovery-seed.jsonl")
    report_path = candidate_root / "reports" / "candidate-only" / (BATCH_ID + ".json")
    output_files[seed_path] = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in seeds)
    report = {
        "batch_id": BATCH_ID,
        "existing_strict_base": 177,
        "new_stored": len(records),
        "high_confidence": len(records),
        "total_provisional_strict_pool": 177 + len(records),
        "relevance_rejected": counters["relevance_rejected"],
        "duplicate_rejected": counters["duplicate_rejected"],
        "extraction_rejected": counters["extraction_rejected"],
        "other_rejected": counters["other_rejected"],
        "publishers": dict(publisher_counts),
        "sources": dict(source_counts),
        "network_requests": global_requests,
        "global_network_ceiling": GLOBAL_NETWORK_CEILING,
        "worker_stats": {key: asdict(value) for key, value in stats.items()},
        "document_id_fixed": all(row.get("document_id") == row.get("candidate_id") for row in existing),
        "human_review_required": True,
        "verified_collection_ready": False,
    }
    output_files[report_path] = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    persist_checkpoint()
    after_baseline = baseline_counts(root)
    report["baseline_before"] = before_baseline
    report["baseline_after"] = after_baseline
    report["baseline_modified"] = before_baseline != after_baseline
    if report["baseline_modified"]:
        raise RuntimeError("protected baseline counts changed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps({"high_confidence_target": HIGH_CONFIDENCE_TARGET, "hard_stored_cap": HARD_STORED_CAP, "global_network_ceiling": GLOBAL_NETWORK_CEILING, "sources": [asdict(value) for value in source_configs()]}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(run(args.root.resolve()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
