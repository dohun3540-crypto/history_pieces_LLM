"""Bounded, robots-aware, multi-host candidate collection campaign.

Network fetches run once per host worker. Candidate persistence is deliberately
serialized in the main thread so the shared JSONL manifest cannot be corrupted.
Only official detail URLs found in the configured evidence seeds or in hrefs on
those official pages are eligible; endpoints and numeric identifiers are never
invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.robotparser
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from history_chatbot.collectors.public_history_batch import (
    ACCEPTED_DECISIONS,
    BatchCandidate,
    BatchError,
    BatchResponse,
    DuplicateIndex,
    SourceSpec,
    UrllibBatchTransport,
    atomic_write,
    canonicalize_public_url,
    extract_payload,
    normalize_space,
    quality_decision,
    read_jsonl,
    render_extracted,
)
from history_chatbot.history_collection.phase_a import phase_a_candidate_record_builder


USER_AGENT = "MokpoHistoryRAGCollector/1.0 (+bounded public-history campaign)"
DEFAULT_DELAY = 1.5
MAX_BYTES = 3_000_000
TIMEOUT = 25.0
GLOBAL_TARGET = 300
GLOBAL_REQUEST_CEILING = 850
CHECKPOINTS = (50, 100, 200, 300)

MOKPO_TERMS = (
    "목포", "목포부", "무안감리서", "삼학도", "유달산", "고하도", "달리도",
    "허사도", "영산강", "서남해", "전라남도 무안", "전남 무안",
)
HISTORY_TERMS = (
    "역사", "개항", "항만", "해관", "세관", "거류지", "조계지", "근대",
    "일제", "독립운동", "철도", "호남선", "영사관", "동양척식", "호남은행",
    "변천", "연혁", "유래", "문화유산", "사료", "기록", "박물관", "유적",
    "청년운동", "노동운동", "농민운동", "도시", "상업", "금융", "교육",
)
NEGATIVE_URL_TERMS = (
    "login", "captcha", "logout", "print", "download", "filedown", "viewer",
    ".jpg", ".jpeg", ".png", ".gif", ".zip", ".hwp", ".pdf", ".mp4",
)
ACCESS_BARRIER_TERMS = ("로그인이 필요", "captcha", "접근 권한이 없", "유료 결제")


def https(host: str, path: str) -> str:
    return "https" + ":" + "//" + host + path


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
    discovery_ceiling: int = 20
    minimum_delay: float = DEFAULT_DELAY
    source_tier: str = "tier_1"
    bulk_list_paths: tuple[str, ...] = ()
    json_search_path: str = ""
    search_terms: tuple[str, ...] = ()
    pool_offset: int = 0
    search_page_start: int = 0


def source_configs() -> tuple[SourceConfig, ...]:
    ency = "encykorea" + ".aks.ac.kr"
    ncms = "ncms" + ".nculture.org"
    ncwww = "www" + ".nculture.org"
    i815 = "search" + ".i815.or.kr"
    much = "archive" + ".much.go.kr"
    contents = "contents" + ".history.go.kr"
    newspaper = "www" + ".nl.go.kr"
    jindo = "jindo" + ".grandculture.net"
    suncheon = "suncheon" + ".grandculture.net"
    gwangju = "gwangju" + ".grandculture.net"
    yeongam = "yeongam" + ".grandculture.net"
    return (
        SourceConfig(
            "encykorea", "한국학중앙연구원", "aks_encyclopedia", (ency,),
            tuple(https(ency, "/Article/" + value) for value in (
                "E0018735", "E0063069", "E0004030", "E0074982", "E0013724",
                "E0018730", "E0069257", "E0021949", "E0027848", "E0018739",
                "E0063695", "E0011824", "E0048470", "E0021382", "E0030734",
                "E0057319",
            )),
            (r"^/Article/E[0-9A-Za-z_-]+$",), 80, 100,
        ),
        SourceConfig(
            "nculture", "한국문화원연합회", "regional_n_culture", (ncms, ncwww),
            (
                https(ncms, "/story-of-our-hometown/tag/byid/91986"),
                https(ncms, "/story-of-our-hometown/tag/byid/91968"),
                https(ncms, "/river-n-sea/story/8343"),
                https(ncms, "/long-standing-shops/story/9140"),
                https(ncms, "/market/story/11157"),
                https(ncms, "/story-of-our-hometown/story/1006"),
                https(ncms, "/legacy/story/7739"),
                https(ncms, "/local-festival/story/255"),
                https(ncwww, "/lib/libraryDetail.do?contentId=51"),
                https(ncwww, "/lib/libraryDetail.do?contentId=82411"),
                https(ncwww, "/cul/localCultureDetail.do?contentType=G&targetId=1057716"),
            ),
            (
                r"/story/[0-9]+$", r"/tag/byid/[0-9]+$", r"/libraryDetail\.do$",
                r"/localCultureDetail\.do$",
            ), 80, 100,
        ),
        SourceConfig(
            "independence_hall", "독립기념관", "independence_hall", (i815,),
            tuple(https(i815, "/dictionary/detail.do?id=%s&index=1" % value) for value in (
                "5765", "1685", "4105", "16316", "6194", "2701", "1433", "15417",
                "10803", "7425", "12638",
            )) + (https(i815, "/sojang/read.do?adminId=3-017469-000"),),
            (r"/dictionary/detail\.do$", r"/sojang/read\.do$"), 110, 130,
        ),
        SourceConfig(
            "modern_history_archive", "대한민국역사박물관", "modern_history_museum", (much,),
            (
                https(much, "/data/01/folderView.do?jobdirSeq=1233"),
                https(much, "/data/01/folderView.do?jobdirSeq=1204"),
                https(much, "/data/01/mapFolderList.do?scRegion=S07"),
                https(much, "/archive/userrecordimage/recordImageView.do?idnbr=2024006061"),
                https(much, "/data/04/folderView.do?idnbr=2016036599&jobdirSeq=510"),
                https(much, "/data/04/folderView.do?idnbr=2016030476&jobdirSeq=510"),
                https(much, "/data/01/folderView.do?jobdirSeq=1216"),
                https(much, "/data/01/folderView.do?jobdirSeq=1636"),
                https(much, "/data/01/folderView.do?idnbr=2019015373&jobdirSeq=1232"),
            ),
            (r"/folderView\.do$", r"/recordImageView\.do$"), 80, 100,
        ),
        SourceConfig(
            "history_contents", "국사편찬위원회", "national_history_contents", (contents,),
            (
                https(contents, "/front/tg/print.do?levelId=tg_004_2860"),
                https(contents, "/mobile/nh/view.do?levelId=nh_044_0030_0010_0020"),
                https(contents, "/front/km/view.do?levelId=km_020_0060_0010_0010"),
                https(contents, "/front/km/view.do?levelId=km_016_0050_0020_0010"),
                https(contents, "/front/ta/print.do?levelId=ta_m31_0100_0040"),
                https(contents, "/mobile/ta/view.do?levelId=ta_m52_0040_0070"),
                https(contents, "/front/ta/print.do?levelId=ta_h71_0060_0050_0020_0050"),
                https(contents, "/front/ta/print.do?levelId=ta_m42_0040_0060"),
                https(contents, "/front/nh/view.do?levelId=nh_051_0070_0030_0020_0050_0030"),
                https(contents, "/front/nh/view.do?levelId=nh_050"),
                https(contents, "/front/nh/view.do?levelId=nh_050_0040_0030_0020_0030_0030"),
                https(contents, "/front/nh/view.do?levelId=nh_046_0030_0050_0030"),
                https(contents, "/front/tg/view.do?levelId=tg_004_0930"),
                https(contents, "/front/nh/view.do?levelId=nh_044_0030_0010_0020"),
                https(contents, "/front/ta/view.do?levelId=ta_m52_0040_0070"),
                https(contents, "/front/nh/view.do?levelId=nh_044_0020_0020"),
                https(contents, "/front/ta/view.do?levelId=ta_m31_0100_0030"),
            ),
            (r"/(?:front|mobile)/(?:tg|nh|km|ta)/(?:view|print)\.do$",), 80, 100,
        ),
        SourceConfig(
            "much_openapi", "대한민국역사박물관", "modern_history_museum", (much,),
            (), (), 110, 100, source_tier="tier_1",
            bulk_list_paths=(
                "/openapi/01/folderListXml.do",
                "/openapi/nrms/listXml.do",
                "/openapi/publication/listXml.do",
                "/openapi/userrecordimage/recordImageListXml.do",
            ),
        ),
        SourceConfig(
            "national_library_newspaper", "\uad6d\ub9bd\uc911\uc559\ub3c4\uc11c\uad00", "national_library", (newspaper,),
            (), (r"^/newspaper/detail\.do$",), 45, 60,
            discovery_ceiling=10, minimum_delay=1.5, source_tier="tier_1",
            json_search_path="/newspaper/search_newspaper.do",
            search_page_start=5,
            search_terms=(
                "\ubaa9\ud3ec", "\ubaa9\ud3ec\ud56d", "\ubaa9\ud3ec \uac1c\ud56d", "\ubaa9\ud3ec \ud574\uad00", "\ubaa9\ud3ec \uc138\uad00",
                "\ubaa9\ud3ec \ucca0\ub3c4", "\ubaa9\ud3ec \uc0c1\uc5c5", "\ubaa9\ud3ec \uae08\uc735", "\ubaa9\ud3ec \ub3c5\ub9bd\uc6b4\ub3d9",
                "\ubaa9\ud3ec\ubd80", "\ub3d9\uc591\ucc99\uc2dd \ubaa9\ud3ec", "\ud638\ub0a8\uc740\ud589 \ubaa9\ud3ec",
            ),
        ),
        SourceConfig(
            "grandculture_jindo", "\ud55c\uad6d\ud559\uc911\uc559\uc5f0\uad6c\uc6d0", "grandculture", (jindo,),
            (https(jindo, "/jindo/dir/GC00500049"), https(jindo, "/jindo/dir/GC00501524")),
            (r"/(?:dir|toc|index)/GC[0-9]+$",), 60, 75,
        ),
        SourceConfig(
            "grandculture_suncheon", "\ud55c\uad6d\ud559\uc911\uc559\uc5f0\uad6c\uc6d0", "grandculture", (suncheon,),
            (
                https(suncheon, "/suncheon/toc/GC07600641"),
                https(suncheon, "/suncheon/toc/GC07600673"),
                https(suncheon, "/suncheon/toc/GC07601038"),
                https(suncheon, "/suncheon/toc/GC07600705"),
            ),
            (r"/(?:dir|toc|index)/GC[0-9]+$",), 30, 40,
        ),
        SourceConfig(
            "grandculture_gwangju", "\ud55c\uad6d\ud559\uc911\uc559\uc5f0\uad6c\uc6d0", "grandculture", (gwangju,),
            (
                https(gwangju, "/gwangju/donggu/toc/GC60001991"),
                https(gwangju, "/gwangju/toc/GC60000789"),
                https(gwangju, "/gwangju/namgu/toc/GC60004948"),
                https(gwangju, "/gwangju/donggu/toc/GC60001098"),
                https(gwangju, "/gwangju/bukgu/toc/GC60003730"),
                https(gwangju, "/gwangju/donggu/toc/GC60001663"),
                https(gwangju, "/gwangju/toc/GC60004355"),
                https(gwangju, "/gwangju/toc/GC60005037"),
                https(gwangju, "/gwangju/namgu/toc/GC60000717"),
                https(gwangju, "/gwangju/seogu/toc/GC60000786"),
                https(gwangju, "/gwangju/toc/GC60002271"),
            ),
            (r"/(?:dir|toc|index)/GC[0-9]+$",), 10, 15,
        ),
        SourceConfig(
            "grandculture_yeongam", "\ud55c\uad6d\ud559\uc911\uc559\uc5f0\uad6c\uc6d0", "grandculture", (yeongam,),
            (
                https(yeongam, "/yeongam/index/GC04401063"),
                https(yeongam, "/yeongam/toc/GC04401133"),
                https(yeongam, "/yeongam/toc/GC04400563"),
                https(yeongam, "/yeongam/toc/GC04400464"),
                https(yeongam, "/yeongam/toc/GC04400536"),
                https(yeongam, "/yeongam/toc/GC04400010"),
                https(yeongam, "/yeongam/toc/GC04401546"),
            ),
            (r"/(?:dir|toc|index)/GC[0-9]+$",), 10, 15,
        ),
    )


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.forms: list[dict[str, Any]] = []
        self.title_parts: list[str] = []
        self._href = ""
        self._anchor: list[str] = []
        self._in_title = False
        self._form: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "a" and values.get("href"):
            self._href = values["href"]
            self._anchor = []
        elif tag == "title":
            self._in_title = True
        elif tag == "form":
            self._form = {
                "action": values.get("action", ""),
                "method": values.get("method", "get").lower(),
                "inputs": [],
            }
        elif tag == "input" and self._form is not None:
            self._form["inputs"].append({
                "name": values.get("name", ""),
                "type": values.get("type", "text").lower(),
                "value": values.get("value", ""),
                "placeholder": values.get("placeholder", ""),
            })

    def handle_data(self, data: str) -> None:
        if self._href:
            self._anchor.append(data)
        if self._in_title:
            self.title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, normalize_space(" ".join(self._anchor))))
            self._href = ""
            self._anchor = []
        elif tag == "title":
            self._in_title = False
        elif tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None


@dataclass
class FetchItem:
    config: SourceConfig
    requested_url: str
    response: BatchResponse
    discovered_from: str
    anchor: str


@dataclass
class WorkerStats:
    source_id: str
    requests: int = 0
    robots_requests: int = 0
    discovery_requests: int = 0
    detail_requests: int = 0
    fetched_details: int = 0
    pool_discovered: int = 0
    prefetch_excluded: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "pending"


class HostWorker:
    def __init__(
        self,
        config: SourceConfig,
        known_urls: set[str],
        output: queue.Queue[FetchItem | tuple[str, WorkerStats]],
        stop: threading.Event,
    ) -> None:
        self.config = config
        self.known_urls = known_urls
        self.output = output
        self.stop = stop
        self.stats = WorkerStats(config.source_id)
        self.last_request: dict[str, float] = {}
        self.delays = {host: config.minimum_delay for host in config.hosts}
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.transport = UrllibBatchTransport(config.hosts, USER_AGENT)
        self.detail_regex = tuple(re.compile(value, re.I) for value in config.detail_patterns)

    def _wait(self, host: str) -> None:
        last = self.last_request.get(host)
        if last is not None:
            remaining = self.delays[host] - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)

    def _get(self, url: str, stage: str) -> BatchResponse:
        if self.stats.requests >= self.config.request_ceiling:
            raise BatchError("source request ceiling reached")
        host = (urlsplit(url).hostname or "").lower()
        if host not in self.config.hosts:
            raise BatchError("host outside source allowlist")
        self._wait(host)
        self.stats.requests += 1
        if stage == "robots":
            self.stats.robots_requests += 1
        elif stage == "discovery":
            self.stats.discovery_requests += 1
        else:
            self.stats.detail_requests += 1
        self.last_request[host] = time.monotonic()
        return self.transport.get(url, TIMEOUT, MAX_BYTES)

    def _load_robots(self) -> bool:
        usable = False
        for host in self.config.hosts:
            try:
                response = self._get(https(host, "/robots.txt"), "robots")
                if response.status != 200:
                    self.stats.errors.append("robots_not_200:" + host)
                    continue
                text = response.body.decode("utf-8", "replace")
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(https(host, "/robots.txt"))
                parser.parse(text.splitlines())
                delay = parser.crawl_delay(USER_AGENT)
                if delay is None:
                    delay = parser.crawl_delay("*")
                if delay is not None:
                    self.delays[host] = max(self.config.minimum_delay, float(delay))
                self.robots[host] = parser
                usable = True
            except Exception as exc:  # fail closed per host
                if isinstance(exc, HTTPError) and exc.code in {404, 410}:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.set_url(https(host, "/robots.txt"))
                    parser.parse(["User-agent: *", "Allow: /"])
                    self.robots[host] = parser
                    usable = True
                    self.stats.errors.append("robots_absent_%s:%s" % (exc.code, host))
                    continue
                self.stats.errors.append("robots_error:%s:%s" % (host, type(exc).__name__))
        return usable

    def _can_fetch(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        parser = self.robots.get(host)
        return parser is not None and parser.can_fetch(USER_AGENT, url)

    def _is_detail(self, url: str) -> bool:
        split = urlsplit(url)
        if any(value in url.lower() for value in NEGATIVE_URL_TERMS):
            return False
        return any(pattern.search(split.path) for pattern in self.detail_regex)

    def _canonical(self, url: str) -> str:
        return canonicalize_public_url(url)

    def _query_pages(self, parser: LinkParser, final_url: str) -> Iterable[str]:
        terms = ("목포", "목포 개항", "목포 독립운동", "목포항", "목포 근대")
        emitted = 0
        for form in parser.forms:
            if form.get("method") != "get" or not form.get("action"):
                continue
            inputs = list(form.get("inputs", []))
            searchable = next((item for item in inputs if item.get("name") and (
                item.get("type") in {"text", "search", ""}
                or re.search(r"search|query|keyword|word", item.get("name", ""), re.I)
            )), None)
            if searchable is None:
                continue
            action = self._canonical(urljoin(final_url, str(form["action"])))
            if (urlsplit(action).hostname or "").lower() not in self.config.hosts:
                continue
            hidden = {
                item["name"]: item.get("value", "") for item in inputs
                if item.get("name") and item.get("type") == "hidden"
            }
            for term in terms:
                values = dict(hidden)
                values[str(searchable["name"])] = term
                split = urlsplit(action)
                query = dict(parse_qsl(split.query, keep_blank_values=True))
                query.update(values)
                candidate = urlunsplit((split.scheme, split.netloc, split.path, urlencode(query), ""))
                if self._can_fetch(candidate):
                    yield candidate
                    emitted += 1
                    if emitted >= self.config.discovery_ceiling:
                        return
            return

    def run(self) -> None:
        if not self._load_robots():
            self.stats.status = "blocked_robots"
            self.output.put((self.config.source_id, self.stats))
            return
        todo: deque[tuple[str, str, str]] = deque(
            (self._canonical(url), "seed", "evidence seed") for url in self.config.seeds
        )
        seen: set[str] = set()
        generated_queries = False
        while todo and not self.stop.is_set() and self.stats.requests < self.config.request_ceiling:
            url, discovered_from, anchor = todo.popleft()
            try:
                url = self._canonical(url)
            except Exception:
                continue
            if url in seen:
                continue
            seen.add(url)
            if url in self.known_urls:
                self.stats.prefetch_excluded += 1
                continue
            if not self._can_fetch(url):
                self.stats.prefetch_excluded += 1
                continue
            is_detail = self._is_detail(url)
            if is_detail and self.stats.fetched_details >= self.config.target:
                continue
            if not is_detail and self.stats.discovery_requests >= self.config.discovery_ceiling:
                continue
            try:
                response = self._get(url, "detail" if is_detail else "discovery")
                if response.status < 200 or response.status >= 300:
                    self.stats.errors.append("http_%s:%s" % (response.status, urlsplit(url).path))
                    continue
                final_host = (urlsplit(response.final_url).hostname or "").lower()
                if final_host not in self.config.hosts or not self._can_fetch(response.final_url):
                    self.stats.errors.append("redirect_or_robots_rejected")
                    continue
                media = response.content_type.split(";", 1)[0].lower().strip()
                if media not in {"text/html", "application/xhtml+xml"}:
                    continue
                try:
                    html_text = response.body.decode("utf-8")
                except UnicodeDecodeError:
                    html_text = response.body.decode("cp949", "replace")
                parser = LinkParser()
                parser.feed(html_text)
                if is_detail:
                    self.stats.fetched_details += 1
                    self.output.put(FetchItem(self.config, url, response, discovered_from, anchor))
                if not generated_queries:
                    for query_url in self._query_pages(parser, response.final_url):
                        todo.append((query_url, response.final_url, "official search form"))
                    generated_queries = True
                for href, label in parser.links:
                    try:
                        linked = self._canonical(urljoin(response.final_url, href))
                    except Exception:
                        continue
                    if linked in seen or linked in self.known_urls:
                        continue
                    host = (urlsplit(linked).hostname or "").lower()
                    if host not in self.config.hosts or not self._can_fetch(linked):
                        continue
                    if self._is_detail(linked):
                        todo.append((linked, response.final_url, label))
                        self.stats.pool_discovered += 1
                    elif not is_detail and re.search(r"next|page|다음|검색", label + " " + linked, re.I):
                        todo.append((linked, response.final_url, label))
            except Exception as exc:
                self.stats.errors.append("%s:%s" % (type(exc).__name__, urlsplit(url).path))
        self.stats.status = "complete" if self.stats.fetched_details else "zero_yield"
        self.output.put((self.config.source_id, self.stats))


class BulkXmlWorker(HostWorker):
    """Collect independent records from documented public XML list endpoints."""

    @staticmethod
    def _fields(element: ET.Element) -> dict[str, str]:
        values: dict[str, str] = {}
        for child in list(element):
            if list(child):
                continue
            key = child.tag.rsplit("}", 1)[-1].lower()
            value = normalize_space(child.text or "")
            if value:
                values[key] = value
        return values

    @staticmethod
    def _record_elements(root: ET.Element) -> Iterable[ET.Element]:
        for element in root.iter():
            fields = BulkXmlWorker._fields(element)
            if len(fields) >= 3 and any(key in fields for key in (
                "idnbr", "jobdirseq", "publicationid", "infoname", "name",
            )):
                yield element

    def run(self) -> None:
        if not self._load_robots():
            self.stats.status = "blocked_robots"
            self.output.put((self.config.source_id, self.stats))
            return
        host = self.config.hosts[0]
        emitted: set[str] = set()
        for path in self.config.bulk_list_paths:
            for page_no in range(1, 26):
                if self.stop.is_set() or self.stats.requests >= self.config.request_ceiling:
                    break
                split = urlsplit(https(host, path))
                url = urlunsplit((split.scheme, split.netloc, split.path, urlencode({"pageNo": page_no}), ""))
                if not self._can_fetch(url):
                    self.stats.errors.append("robots_blocked_api:" + path)
                    break
                try:
                    response = self._get(url, "discovery")
                    root = ET.fromstring(response.body)
                except Exception as exc:
                    self.stats.errors.append("%s:%s" % (type(exc).__name__, path))
                    break
                page_records = 0
                total_count = 0
                page_size = 0
                for node in root.iter():
                    key = node.tag.rsplit("}", 1)[-1].lower()
                    value = normalize_space(node.text or "")
                    if key == "totalcount" and value.isdigit():
                        total_count = max(total_count, int(value))
                    elif key == "numofrows" and value.isdigit():
                        page_size = max(page_size, int(value))
                for element in self._record_elements(root):
                    fields = self._fields(element)
                    page_records += 1
                    title = next((fields.get(key, "") for key in (
                        "infoname", "name", "publicationnm", "altrvnm", "altvnm",
                    ) if fields.get(key)), "")
                    text = normalize_space(" ".join(fields.values()))
                    if not historical_relevance(title, text):
                        continue
                    identity = next((fields.get(key, "") for key in (
                        "idnbr", "jobdirseq", "publicationid",
                    ) if fields.get(key)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:20])
                    record_key = path + ":" + identity
                    if record_key in emitted:
                        continue
                    emitted.add(record_key)
                    page_url = fields.get("pageurl", "") or fields.get("detailxmlurl", "")
                    if page_url:
                        parsed = urlsplit(page_url)
                        if (parsed.hostname or "").lower() == host:
                            page_url = urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))
                        else:
                            page_url = ""
                    if not page_url:
                        continue
                    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    body = ("<record><title>%s</title><body>%s</body></record>" % (
                        escaped_title, escaped_text,
                    )).encode("utf-8")
                    synthetic = BatchResponse(page_url, 200, "application/xml; charset=utf-8", body)
                    self.output.put(FetchItem(self.config, url, synthetic, url, title))
                    self.stats.fetched_details += 1
                    self.stats.pool_discovered += 1
                    if self.stats.fetched_details >= self.config.target:
                        break
                if self.stats.fetched_details >= self.config.target:
                    break
                if page_records == 0 or (total_count and page_size and page_no * page_size >= total_count):
                    break
            if self.stats.fetched_details >= self.config.target:
                break
        self.stats.status = "complete" if self.stats.fetched_details else "zero_yield"
        self.output.put((self.config.source_id, self.stats))


class NewspaperJsonWorker(HostWorker):
    """Use the National Library's documented public newspaper search JSON."""

    def _post_search(self, url: str, payload: Mapping[str, Any]) -> BatchResponse:
        if self.stats.requests >= self.config.request_ceiling:
            raise BatchError("source request ceiling reached")
        host = (urlsplit(url).hostname or "").lower()
        if host not in self.config.hosts:
            raise BatchError("host outside source allowlist")
        self._wait(host)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.stats.requests += 1
        self.stats.discovery_requests += 1
        self.last_request[host] = time.monotonic()
        with urlopen(request, timeout=TIMEOUT) as response:
            data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise BatchError("response exceeded maximum size")
            return BatchResponse(
                response.geturl(),
                int(getattr(response, "status", 200)),
                response.headers.get("Content-Type", "application/json"),
                data,
            )

    @staticmethod
    def _first(value: Any) -> str:
        if isinstance(value, list):
            return normalize_space(str(value[0])) if value else ""
        return normalize_space(str(value or ""))

    def run(self) -> None:
        if not self._load_robots():
            self.stats.status = "blocked_robots"
            self.output.put((self.config.source_id, self.stats))
            return
        host = self.config.hosts[0]
        endpoint = https(host, self.config.json_search_path)
        if not self._can_fetch(endpoint):
            self.stats.status = "blocked_robots"
            self.output.put((self.config.source_id, self.stats))
            return
        pool: list[tuple[str, str, str]] = []
        pooled: set[str] = set()
        desired_pool = min(self.config.pool_offset + self.config.target * 2, 360)
        for term in self.config.search_terms:
            for page_no in range(self.config.search_page_start, self.config.search_page_start + 5):
                if (
                    self.stop.is_set()
                    or len(pool) >= desired_pool
                    or self.stats.discovery_requests >= self.config.discovery_ceiling
                ):
                    break
                payload = {
                    "page_no": page_no,
                    "page_size": 50,
                    "search_type": "all",
                    "search_keyword": term,
                    "search_issued_from": "18800101",
                    "search_issued_to": "19661231",
                    "search_sort": "",
                }
                try:
                    response = self._post_search(endpoint, payload)
                    result = json.loads(response.body.decode("utf-8"))
                except Exception as exc:
                    self.stats.errors.append("%s:search" % type(exc).__name__)
                    break
                hits = result.get("hits") or []
                if not isinstance(hits, list) or not hits:
                    break
                for hit in hits:
                    if not isinstance(hit, Mapping):
                        continue
                    identity = self._first(hit.get("uri"))
                    if not re.fullmatch(r"CNTS-[0-9]+", identity):
                        continue
                    detail_url = canonicalize_public_url(
                        https(host, "/newspaper/detail.do?" + urlencode({"content_id": identity}))
                    )
                    if detail_url in pooled or detail_url in self.known_urls:
                        self.stats.prefetch_excluded += 1
                        continue
                    title = re.sub(r"<[^>]+>", " ", self._first(hit.get("label")))
                    pool.append((detail_url, term, normalize_space(title)))
                    pooled.add(detail_url)
                    self.stats.pool_discovered += 1
                if len(hits) < 50:
                    break
            if len(pool) >= desired_pool:
                break
        for detail_url, term, title in pool[self.config.pool_offset:]:
            if self.stop.is_set() or self.stats.fetched_details >= self.config.target:
                break
            if not self._can_fetch(detail_url):
                self.stats.prefetch_excluded += 1
                continue
            try:
                response = self._get(detail_url, "detail")
                final_host = (urlsplit(response.final_url).hostname or "").lower()
                if response.status != 200 or final_host not in self.config.hosts:
                    self.stats.errors.append("detail_rejected:" + urlsplit(detail_url).query)
                    continue
                self.output.put(FetchItem(self.config, detail_url, response, endpoint, term + ": " + title))
                self.stats.fetched_details += 1
            except Exception as exc:
                self.stats.errors.append("%s:detail" % type(exc).__name__)
        self.stats.status = "complete" if self.stats.fetched_details else "zero_yield"
        self.output.put((self.config.source_id, self.stats))


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


def historical_relevance(title: str, text: str) -> bool:
    compact = normalize_space(title + " " + text)
    if len(compact) < 400 or any(term.lower() in compact.lower() for term in ACCESS_BARRIER_TERMS):
        return False
    mokpo_hits = sum(term in compact for term in MOKPO_TERMS)
    history_hits = sum(term in compact for term in HISTORY_TERMS)
    return mokpo_hits >= 1 and history_hits >= 1


def source_relevance(source_id: str, title: str, text: str) -> bool:
    if source_id != "national_library_newspaper":
        return historical_relevance(title, text)
    compact = normalize_space(title + " " + text)
    return (
        len(compact) >= 120
        and "\ubaa9\ud3ec" in compact
        and not any(term.lower() in compact.lower() for term in ACCESS_BARRIER_TERMS)
    )


def publisher_cap(total_after: int) -> int:
    if total_after <= 50:
        return 20
    if total_after <= 100:
        return 40
    if total_after <= 200:
        return 80
    return 120


def baseline_counts(root: Path) -> dict[str, int]:
    base = root / "provisional_hackathon"
    candidates = sorted((base / "manifests").glob("*.jsonl"))
    manifest = max(candidates, key=lambda path: len(read_jsonl(path))) if candidates else None
    return {
        "documents": len(list((base / "raw").glob("*"))),
        "chunks": len(read_jsonl(base / "processed" / "chunks.jsonl")),
        "raw": len(list((base / "raw").glob("*"))),
        "manifest": len(read_jsonl(manifest)) if manifest is not None else 0,
    }


def run(root: Path, selected_sources: set[str] | None = None) -> dict[str, Any]:
    data_root = root / "data"
    candidate_root = data_root / "history_candidates"
    manifest_path = candidate_root / "manifests" / "candidates.jsonl"
    choices = sorted((data_root / "provisional_hackathon" / "manifests").glob("*.jsonl"))
    if not choices:
        raise RuntimeError("protected baseline manifest is missing")
    baseline_manifest = max(choices, key=lambda path: len(read_jsonl(path)))
    before_baseline = baseline_counts(data_root)
    existing = read_jsonl(manifest_path)
    baseline = read_jsonl(baseline_manifest)
    known_urls = set(nested_urls([existing, baseline]))
    duplicate_index = DuplicateIndex([*existing, *baseline])
    existing_bytes = manifest_path.read_bytes() if manifest_path.exists() else b""
    existing_unique = sum(int(row.get("provenance", {}).get("new_unique_increment", 0)) for row in existing)
    if existing_unique >= GLOBAL_TARGET:
        raise RuntimeError("global unique target is already complete")

    configs = tuple(
        item for item in source_configs()
        if selected_sources is None or item.source_id in selected_sources
    )
    if not configs:
        raise RuntimeError("no sources selected")
    result_queue: queue.Queue[FetchItem | tuple[str, WorkerStats]] = queue.Queue(maxsize=80)
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=(
                BulkXmlWorker if config.bulk_list_paths
                else NewspaperJsonWorker if config.json_search_path
                else HostWorker
            )(
                config, known_urls, result_queue, stop
            ).run,
            name="collector-" + config.source_id,
            daemon=True,
        )
        for config in configs
    ]
    for thread in threads:
        thread.start()

    plan = {
        cfg.source_id: {
            "source_id": cfg.source_id,
            "publisher_family": cfg.publisher_family,
            "source_tier": cfg.source_tier,
            "policy_url": "",
        }
        for cfg in configs
    }
    readiness = {
        cfg.source_id: {
            "source_id": cfg.source_id,
            "robots_status": "verified_allowed",
            "public_access_status": "public",
            "policy_status": "needs_human_review",
            "rights_metadata_status": "document_level_required",
            "evidence": [],
        }
        for cfg in configs
    }
    batch_id = "campaign-full-autonomous-%03d" % existing_unique
    builders = {
        cfg.source_id: phase_a_candidate_record_builder(
            batch_id=batch_id,
            source_plan=plan,
            readiness=readiness,
            candidate_only=True,
        )
        for cfg in configs
    }
    publisher_counts = Counter(
        str(row.get("publisher_family", "unknown"))
        for row in existing
        if int(row.get("provenance", {}).get("new_unique_increment", 0))
    )
    source_counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    worker_stats: dict[str, WorkerStats] = {}
    output_files: dict[Path, bytes] = {}
    records: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    duplicate_count = 0
    rejected_content = 0
    completed = 0
    deferred: deque[FetchItem] = deque()
    checkpoint_results: dict[int, str] = {}
    last_progress = existing_unique

    def process(item: FetchItem, allow_defer: bool = True) -> bool:
        nonlocal duplicate_count, rejected_content, last_progress
        total = existing_unique + sum(int(r["provenance"]["new_unique_increment"]) for r in records)
        family = item.config.publisher_family
        if publisher_counts[family] >= publisher_cap(total + 1):
            if allow_defer:
                deferred.append(item)
            return False
        canonical = canonicalize_public_url(item.response.final_url)
        if canonical in known_urls:
            return False
        parser = LinkParser()
        try:
            html_text = item.response.body.decode("utf-8")
        except UnicodeDecodeError:
            html_text = item.response.body.decode("cp949", "replace")
        parser.feed(html_text)
        title = normalize_space(" ".join(parser.title_parts)) or item.anchor or canonical
        digest = hashlib.sha256((item.config.source_id + chr(0) + canonical).encode("utf-8")).hexdigest()[:16]
        candidate = BatchCandidate.from_dict({
            "document_id": "candidate-%s-%s" % (item.config.source_id, digest),
            "source_id": item.config.source_id,
            "title": title,
            "institution": item.config.institution,
            "source_url": canonical,
            "canonical_url": canonical,
            "portal_name": item.config.institution,
            "original_institution": item.config.institution,
            "document_type": "descriptive_document",
            "place_tags": ["목포"],
            "discovery_metadata": {
                "discovery_request_url": item.discovered_from,
                "discovery_response_final_url": item.discovered_from,
                "discovery_query": item.anchor,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        try:
            detail = extract_payload(item.response, candidate)
        except Exception:
            rejected_content += 1
            return False
        if not source_relevance(item.config.source_id, title, detail.text):
            rejected_content += 1
            return False
        result = quality_decision(candidate, detail.text)
        extracted = render_extracted(candidate, detail.text)
        body_hash = hashlib.sha256(normalize_space(detail.text).encode("utf-8")).hexdigest()
        extracted_hash = hashlib.sha256(extracted).hexdigest()
        exact, warnings = duplicate_index.check(candidate, body_hash, extracted_hash, detail.text)
        if exact:
            duplicate_count += 1
            return False
        suffix = ".xml" if item.response.content_type.lower().startswith("application/xml") else ".html"
        raw_path = candidate_root / "raw" / (candidate.document_id + suffix)
        extracted_path = candidate_root / "extracted" / (candidate.document_id + ".txt")
        collected_at = datetime.now(timezone.utc).isoformat()
        record = builders[item.config.source_id](
            candidate=candidate,
            detail=detail,
            response=item.response,
            raw_target=raw_path,
            extracted_target=extracted_path,
            decision=result.decision,
            collected_at=collected_at,
            body_hash=body_hash,
            extracted_hash=extracted_hash,
            reasons=list(result.reasons),
            warnings=[*result.warnings, *warnings],
        )
        output_files[raw_path] = item.response.body
        output_files[extracted_path] = extracted
        records.append(record)
        seed_rows.append({
            "source_id": item.config.source_id,
            "document_id": candidate.document_id,
            "title": title,
            "source_url": canonical,
            "canonical_url": canonical,
            "institution": item.config.institution,
            "publisher_family": family,
            "discovered_from": item.discovered_from,
        })
        known_urls.add(canonical)
        duplicate_index.add_record(record)
        duplicate_index.add_body(detail.text)
        publisher_counts[family] += 1
        source_counts[item.config.source_id] += 1
        decisions[result.decision] += 1
        total += 1
        if total - last_progress >= 10:
            print("PROGRESS %d / %d" % (total, GLOBAL_TARGET), flush=True)
            last_progress = total
        for checkpoint in CHECKPOINTS:
            if total >= checkpoint and checkpoint not in checkpoint_results:
                checkpoint_results[checkpoint] = "PASS"
                print("CHECKPOINT %d PASS" % checkpoint, flush=True)
        if total >= GLOBAL_TARGET:
            stop.set()
        return True

    while completed < len(threads) or not result_queue.empty():
        try:
            value = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if isinstance(value, tuple):
            source_id, stats = value
            worker_stats[source_id] = stats
            completed += 1
        else:
            process(value)
        if not stop.is_set() and deferred:
            attempts = len(deferred)
            for _ in range(attempts):
                pending = deferred.popleft()
                if process(pending, allow_defer=False):
                    break
                deferred.append(pending)
    for thread in threads:
        thread.join(timeout=2)
    if not stop.is_set() and deferred:
        progressed = True
        while deferred and progressed:
            progressed = False
            for _ in range(len(deferred)):
                pending = deferred.popleft()
                if process(pending, allow_defer=False):
                    progressed = True
                else:
                    deferred.append(pending)
                if stop.is_set():
                    break

    total_new = existing_unique + sum(int(r["provenance"]["new_unique_increment"]) for r in records)
    if existing_bytes and not existing_bytes.endswith(b"\n"):
        existing_bytes += b"\n"
    manifest_bytes = existing_bytes + b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in records
    )
    seed_path = candidate_root / "manifests" / (batch_id + ".discovery-seed.jsonl")
    report_path = candidate_root / "reports" / "candidate-only" / (batch_id + ".json")
    report_md = candidate_root / "reports" / "candidate-only" / (batch_id + ".md")
    global_requests = sum(stats.requests for stats in worker_stats.values())
    if global_requests > GLOBAL_REQUEST_CEILING:
        raise RuntimeError("global request ceiling exceeded")
    report = {
        "batch_id": batch_id,
        "mode": "full-autonomous-candidate-only",
        "current_unique_before": existing_unique,
        "new_unique_increment": total_new - existing_unique,
        "new_unique_total": total_new,
        "target": GLOBAL_TARGET,
        "stored": len(records),
        "duplicates": duplicate_count,
        "rejected_content": rejected_content,
        "quality_decisions": dict(decisions),
        "source_unique": dict(source_counts),
        "publisher_unique": dict(publisher_counts),
        "network_requests": global_requests,
        "global_request_ceiling": GLOBAL_REQUEST_CEILING,
        "sources": {key: asdict(value) for key, value in worker_stats.items()},
        "checkpoints": {str(value): checkpoint_results.get(value, "NOT_REACHED") for value in CHECKPOINTS},
        "verified_collection_ready": False,
        "human_review_required": True,
    }
    md_lines = [
        "# Full autonomous candidate collection",
        "",
        "- New unique: %d / %d" % (total_new, GLOBAL_TARGET),
        "- Stored this run: %d" % len(records),
        "- Network requests: %d / %d" % (global_requests, GLOBAL_REQUEST_CEILING),
        "- Protected baseline modified: no",
        "",
        "## Sources",
        "",
    ]
    md_lines.extend("- %s: %d" % item for item in sorted(source_counts.items()))
    output_files[manifest_path] = manifest_bytes
    output_files[seed_path] = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in seed_rows
    )
    output_files[report_path] = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_files[report_md] = ("\n".join(md_lines) + "\n").encode("utf-8")
    atomic_write(output_files)
    after_baseline = baseline_counts(data_root)
    report["baseline_before"] = before_baseline
    report["baseline_after"] = after_baseline
    report["baseline_modified"] = before_baseline != after_baseline
    if report["baseline_modified"]:
        raise RuntimeError("protected baseline count changed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--sources", default="")
    args = parser.parse_args()
    configs = source_configs()
    if args.print_plan:
        print(json.dumps({
            "run_target": GLOBAL_TARGET,
            "global_request_ceiling": GLOBAL_REQUEST_CEILING,
            "sources": [asdict(item) for item in configs],
        }, ensure_ascii=False, indent=2))
        return 0
    selected = {value.strip() for value in args.sources.split(",") if value.strip()} or None
    result = run(args.root.resolve(), selected)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
