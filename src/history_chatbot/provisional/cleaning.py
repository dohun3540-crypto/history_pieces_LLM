"""Local-only cleaning and semantic sectioning for provisional source text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CleanSection:
    """A source-derived section that must not merge across its boundary."""

    title: str
    paragraphs: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


_COMMON_UI_LINES = {
    "닫기",
    "프린트",
    "페이지 인쇄",
    "QR코드 바로보기",
    "전승자 정보 바로보기",
    "설명 비교하기",
    "목록으로 이동하기",
    "바로가기",
}
_HERITAGE_UI_LINES = _COMMON_UI_LINES | {
    "본문",
    "국가유산 검색",
    "페이지",
    "구성",
    "이 페이지의 구성",
    "의견 등록하기",
    "의견처리 결과 확인",
    "이전",
    "다음",
}
_HERITAGE_ADJACENT_NAV = re.compile(
    r"^(?:이전|다음)\s+(?:문화유산|국가유산|항목|페이지)$"
)
_REFERENCE_MARKERS = {"|참고문헌|", "참고문헌"}
_AUTHOR_MARKER = re.compile(r"^⋮.+⋮$")
_YEAR = re.compile(r"(?<!\d)((?:18|19|20)\d{2})년")
_BIOGRAPHY_ACTIVITY_FIELDS = (
    "운동계열",
    "관련 단체",
    "관련 사건",
    "주요 활동",
    "포상훈격(연도)",
)


def clean_sections(record: dict, text: str) -> list[CleanSection]:
    """Return cleaned, source-derived sections without inventing historical prose."""

    lines = _lines(text)
    institution = str(record.get("institution", ""))
    if institution == "독립기념관":
        return _independence_sections(lines)
    if "국가유산" in institution:
        return _heritage_sections(lines)
    if institution == "목포시" and "역사·지명유래" in str(
        record.get("source_title", "")
    ):
        return _mokpo_chronology_sections(lines)
    return _generic_sections(lines)


def _lines(text: str) -> list[str]:
    return [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def _generic_sections(lines: list[str]) -> list[CleanSection]:
    cleaned = [line for line in lines if line not in _COMMON_UI_LINES]
    cleaned = _before_references(cleaned)
    return [CleanSection("본문", tuple(cleaned))] if cleaned else []


def _before_references(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if line in _REFERENCE_MARKERS:
            return lines[:index]
    return lines


def _independence_sections(lines: list[str]) -> list[CleanSection]:
    cleaned = [
        line
        for line in _before_references(lines)
        if line not in _COMMON_UI_LINES
        and line != "독립운동인명사전 - 한국독립운동정보시스템"
        and line != "모바일 메뉴 열기/닫기"
        and not _AUTHOR_MARKER.match(line)
    ]
    if not cleaned:
        return []

    try:
        basic_index = cleaned.index("기본정보")
        subject_index = max(0, basic_index - 2)
        subject = cleaned[subject_index]
    except ValueError:
        return [CleanSection("본문", tuple(cleaned))]

    narrative_index = _biography_narrative_start(cleaned, basic_index)
    basic = cleaned[subject_index:narrative_index]
    sections: list[CleanSection] = []
    if basic:
        profile, source_metadata = _biography_profile(basic)
        sections.append(
            CleanSection(
                f"{subject} 기본정보",
                tuple(profile),
                {"source_profile_metadata": tuple(source_metadata)},
            )
        )
    sections.extend(_biography_narrative(subject, cleaned[narrative_index:]))
    return sections


def _biography_profile(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate searchable identity fields from source-preserved activity metadata."""

    activity_start = min(
        (
            lines.index(field_name)
            for field_name in _BIOGRAPHY_ACTIVITY_FIELDS
            if field_name in lines
        ),
        default=len(lines),
    )
    return lines[:activity_start], lines[activity_start:]


def _biography_narrative_start(lines: list[str], basic_index: int) -> int:
    try:
        award_index = lines.index("포상훈격(연도)", basic_index)
    except ValueError:
        return basic_index
    return min(len(lines), award_index + 2)


def _biography_narrative(subject: str, paragraphs: list[str]) -> list[CleanSection]:
    sections: list[CleanSection] = []
    current: list[str] = []
    first_year = ""
    last_year = ""

    def flush() -> None:
        nonlocal current, first_year, last_year
        if not current:
            return
        label = (
            f"{subject} {first_year}년 활동"
            if first_year
            else f"{subject} 활동"
        )
        sections.append(CleanSection(label, tuple(current)))
        current = []
        first_year = ""
        last_year = ""

    for paragraph in paragraphs:
        match = _YEAR.search(paragraph[:100])
        year = match.group(1) if match else ""
        is_award = paragraph.startswith("대한민국 정부")
        is_transition = paragraph.startswith(("출옥 이후", "풀려난 뒤", "석방된 뒤"))
        year_jump = bool(
            year and last_year and abs(int(year) - int(last_year)) >= 2
        )
        if current and (
            is_award
            or is_transition
            or year_jump
        ):
            flush()
        current.append(paragraph)
        if year:
            if not first_year:
                first_year = year
            last_year = year
        if is_award:
            flush()
    flush()
    return sections


def _is_heritage_ui_line(line: str) -> bool:
    return line in _HERITAGE_UI_LINES or bool(_HERITAGE_ADJACENT_NAV.fullmatch(line))


def _heritage_sections(lines: list[str]) -> list[CleanSection]:
    try:
        start = lines.index("기본 정보")
    except ValueError:
        start = 0
    cleaned = [line for line in lines[start:] if not _is_heritage_ui_line(line)]
    return [CleanSection("기본 정보", tuple(cleaned))] if cleaned else []


def _mokpo_chronology_sections(lines: list[str]) -> list[CleanSection]:
    try:
        start = lines.index("삼국 이전")
    except ValueError:
        return _generic_sections(lines)
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("담당자")
            or lines[index].startswith("이 페이지에서 제공하는 정보")
        ),
        len(lines),
    )
    body = lines[start:end]
    summary_index = _find(body, "요약")
    detailed = body[:summary_index] if summary_index is not None else body
    summary = body[summary_index:] if summary_index is not None else []

    sections: list[CleanSection] = []
    opening_index = _find_prefix(detailed, "1895년")
    occupation_index = _find(detailed, "일제강점기")
    liberation_index = _find(detailed, "8·15 광복 이후")
    if opening_index is None or occupation_index is None or liberation_index is None:
        return [CleanSection("역사 본문", tuple(body))]

    _append(sections, "전근대 역사", detailed[:opening_index])
    _append(sections, "개항·거류·무역", detailed[opening_index : opening_index + 1])
    _append(
        sections,
        "항만·도로·해상교통",
        detailed[opening_index + 1 : opening_index + 2],
    )
    _append(
        sections,
        "도시 기반시설·상수도",
        detailed[opening_index + 2 : occupation_index],
    )
    _append(
        sections,
        "일제강점기 도시 확장",
        detailed[occupation_index : occupation_index + 2],
    )
    _append(sections, "철도·교통", detailed[occupation_index + 2 : liberation_index])
    _append(
        sections,
        "광복 이후 현대 행정구역 변화",
        detailed[liberation_index:],
    )

    if summary:
        modern_summary = _find_prefix(summary, "고종32년")
        post_summary = _find(summary, "8·15 광복 이후")
        if modern_summary is not None and post_summary is not None:
            _append(sections, "원문 요약 전근대", summary[:modern_summary])
            _append(
                sections,
                "개항 이후 근대 도시 변화 요약",
                summary[modern_summary:post_summary],
            )
            _append(sections, "원문 요약 광복 이후", summary[post_summary:])
        else:
            _append(sections, "원문 요약", summary)
    return sections


def _find(lines: list[str], value: str) -> int | None:
    try:
        return lines.index(value)
    except ValueError:
        return None


def _find_prefix(lines: list[str], prefix: str) -> int | None:
    return next((index for index, line in enumerate(lines) if line.startswith(prefix)), None)


def _append(sections: list[CleanSection], title: str, paragraphs: list[str]) -> None:
    if paragraphs:
        sections.append(CleanSection(title, tuple(paragraphs)))
