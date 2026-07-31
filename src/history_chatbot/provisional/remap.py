"""독립기념관 인명사전 구형 URL을 현재 공식 상세 경로와 대조한다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse


CURRENT_DETAIL_TEMPLATE = (
    "https://search.i815.or.kr/dictionary/detail/print.do?id={record_id}"
)
ERROR_MARKERS = ("페이지를 찾을 수 없습니다", "주소가 변경 혹은 삭제")


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    record_id: str
    title: str


class _I815Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.candidates: list[SearchCandidate] = []
        self._candidate_id = ""
        self._candidate_depth = 0
        self._candidate_parts: list[str] = []
        self._in_page_title = False
        self._page_title_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        classes = values.get("class", "").split()
        onclick = values.get("onclick", "")
        match = re.fullmatch(r"\s*goDetail\((\d+)\);\s*", onclick)
        if match and not self._candidate_id:
            self._candidate_id = match.group(1)
            self._candidate_depth = self.depth + 1
            self._candidate_parts = []
        if "dict_txt_title" in classes or "entry-name" in classes:
            self._in_page_title = True
            self._page_title_depth = self.depth + 1
        self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._in_page_title and self.depth == self._page_title_depth:
            self._in_page_title = False
        if self._candidate_id and self.depth == self._candidate_depth:
            title = " ".join("".join(self._candidate_parts).split())
            if title:
                self.candidates.append(SearchCandidate(self._candidate_id, title))
            self._candidate_id = ""
            self._candidate_parts = []
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._candidate_id:
            self._candidate_parts.append(data)
        if self._in_page_title:
            self.title_parts.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())

    @property
    def page_title(self) -> str:
        return " ".join("".join(self.title_parts).split())


def old_record_id(url: str) -> str:
    return str(parse_qs(urlparse(url).query).get("id", [""])[0])


def expected_name(source_title: str) -> str:
    match = re.match(r"^([가-힣A-Za-z·\s]+?)\s*-\s*독립운동인명사전$", source_title)
    return " ".join(match.group(1).split()) if match else ""


def search_key(record: dict) -> str:
    return expected_name(str(record.get("source_title", ""))) or old_record_id(
        str(record.get("source_url", ""))
    )


def parse_search_candidates(html: str) -> list[SearchCandidate]:
    parser = _I815Parser()
    parser.feed(html)
    unique: dict[tuple[str, str], SearchCandidate] = {}
    for candidate in parser.candidates:
        unique[(candidate.record_id, candidate.title)] = candidate
    return list(unique.values())


def classify_old_response(status: int, html: str) -> str:
    if status >= 500 and any(marker in html for marker in ERROR_MARKERS):
        return "source_removed"
    if status >= 400:
        return "not_found"
    return "available"


def classify_mapping(
    record: dict,
    *,
    status: int,
    content_type: str,
    html: str,
    search_candidates: list[SearchCandidate] | None = None,
) -> tuple[str, float, list[str], str]:
    parser = _I815Parser()
    parser.feed(html)
    text = parser.text
    current_title = parser.page_title
    record_id = old_record_id(str(record.get("source_url", "")))
    name = expected_name(str(record.get("source_title", "")))
    reasons = ["공식 print 상세 경로", "기존 내부 ID 유지"]

    if (
        status != 200
        or "html" not in content_type.lower()
        or any(marker in text for marker in ERROR_MARKERS)
        or len(text) < 120
    ):
        return "not_found", 0.0, ["정상 HTML 상세 본문 없음"], current_title

    same_name = [
        item for item in (search_candidates or []) if item.title.split(" ")[0] == name
    ]
    if name and len({item.record_id for item in same_name}) > 1:
        return "ambiguous", 0.5, ["동일 이름 후보가 여러 내부 ID로 검색됨"], current_title
    if name and name in current_title and "목포" in text:
        reasons.extend(["인물명 정확 일치", "본문에서 목포 관련성 확인"])
        return "remapped_exact", 1.0, reasons, current_title
    if name and name in current_title:
        return "remapped_probable", 0.8, reasons + ["인물명 일치", "목포 확인 부족"], current_title
    if not name and record_id and current_title:
        return (
            "manual_review_required",
            0.6,
            reasons + ["기존 제목에 확정 가능한 인물명 없음"],
            current_title,
        )
    return "not_found", 0.0, ["제목 또는 내부 ID를 확인하지 못함"], current_title


def ensure_unique_exact_urls(records: list[dict]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        if record.get("match_status") != "remapped_exact":
            continue
        url = str(record.get("current_detail_url", ""))
        source_id = str(record.get("source_id", ""))
        if url in seen and seen[url] != source_id:
            raise ValueError(f"서로 다른 source_id의 current URL이 중복됩니다: {url}")
        seen[url] = source_id
