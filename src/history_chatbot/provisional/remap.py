"""독립기념관 인명사전 구형 URL을 현재 공식 상세 경로와 대조한다."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
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


@dataclass(frozen=True, slots=True)
class DetailFields:
    name: str
    name_hanja: str
    born_died: str
    address_birth: str
    movement_family: str
    organizations: str
    events: str
    activities: str
    text: str


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


def _clean_html(value: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _table_field(html: str, label: str) -> str:
    match = re.search(
        rf"<th[^>]*>\s*{re.escape(label)}\s*</th>\s*<td[^>]*>(.*?)</td>",
        html,
        re.IGNORECASE | re.DOTALL,
    )
    return _clean_html(match.group(1)) if match else ""


def parse_detail_fields(html: str) -> DetailFields:
    parser = _I815Parser()
    parser.feed(html)
    hanja = re.search(
        r'class=["\'][^"\']*\bentry-name-hanja\b[^"\']*["\'][^>]*>(.*?)</',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    return DetailFields(
        name=parser.page_title,
        name_hanja=_clean_html(hanja.group(1)) if hanja else _table_field(html, "한자명"),
        born_died=_table_field(html, "생몰년월일"),
        address_birth=_table_field(html, "출신지"),
        movement_family=_table_field(html, "운동계열"),
        organizations=_table_field(html, "관련 단체"),
        events=_table_field(html, "관련 사건"),
        activities=_table_field(html, "주요 활동"),
        text=parser.text,
    )


def mokpo_evidence(text: str) -> list[str]:
    evidence: list[str] = []
    for item in re.split(r"(?<=[.!?。])\s+|[\r\n]+", text):
        sentence = " ".join(item.split())
        if "목포" not in sentence:
            continue
        position = sentence.index("목포")
        start = max(0, position - 100)
        excerpt = sentence[start : start + 260]
        if start:
            excerpt = "…" + excerpt
        if start + 260 < len(sentence):
            excerpt += "…"
        if excerpt not in evidence:
            evidence.append(excerpt)
        if len(evidence) == 3:
            break
    return evidence


def review_manual_record(
    record: dict,
    *,
    current_url: str,
    status: int,
    content_type: str,
    html: str,
    existing_results: list[dict] | None = None,
) -> dict:
    source_id = str(record.get("source_id", ""))
    record_id = old_record_id(str(record.get("source_url", "")))
    fields = parse_detail_fields(html)
    evidence = mokpo_evidence(fields.text)
    duplicates = [
        item
        for item in (existing_results or [])
        if str(item.get("source_id", "")) != source_id
        and (
            str(item.get("current_url", item.get("current_detail_url", "")))
            == current_url
            or str(item.get("record_id", item.get("current_record_id", "")))
            == record_id
        )
    ]
    reasons = ["기존 record_id와 현재 공식 print ID 일치"]
    if duplicates:
        review_status = "rejected"
        relevance = "duplicate"
        reasons.append("다른 source_id와 URL 또는 record_id 중복")
    elif (
        status == 200
        and "html" in content_type.lower()
        and fields.name
        and len(fields.text) >= 120
        and evidence
    ):
        review_status = "promoted_to_exact"
        relevance = "direct"
        reasons.extend(["공식 인명사전 본문 확인", "목포 직접 근거 확인"])
    elif status == 200 and fields.name and len(fields.text) >= 120:
        review_status = "reference_only"
        relevance = "none"
        reasons.append("본문에서 목포 직접 관련성을 확인하지 못함")
    else:
        review_status = "manual_review_required"
        relevance = "unclear"
        reasons.append("핵심 인물명 또는 본문 구조 확인 필요")
    return {
        "source_id": source_id,
        "record_id": record_id,
        "current_name": fields.name,
        "name_hanja": fields.name_hanja,
        "born_died": fields.born_died,
        "address_birth": fields.address_birth,
        "movement_family": fields.movement_family,
        "organizations": fields.organizations,
        "events": fields.events,
        "activities": fields.activities,
        "current_url": current_url,
        "old_title": str(record.get("source_title", "")),
        "review_status": review_status,
        "mokpo_relevance": relevance,
        "relevance_evidence": evidence,
        "duplicate_check": "duplicate" if duplicates else "unique",
        "review_reasons": reasons,
        "HTTP_status": status,
        "content_type": content_type,
        "body_exists": len(fields.text) >= 120,
    }
