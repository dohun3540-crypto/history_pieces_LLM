"""Deterministic content diagnostics and multi-label topic classification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from history_chatbot.history_collection.models import TopicCategory


TOPIC_TERMS = {
    TopicCategory.OPENING_TRADE: ("개항", "해관", "세관", "거류지", "조계", "무역", "개항장"),
    TopicCategory.PORT_MARITIME: ("목포항", "항만", "부두", "해운", "증기선", "선박", "창고"),
    TopicCategory.RAIL_TRANSPORT: ("철도", "호남선", "목포역", "기차", "철길"),
    TopicCategory.URBAN_INFRASTRUCTURE: ("도시계획", "상수도", "도로망", "공공시설", "하수도", "전기"),
    TopicCategory.COLONIAL_CITY: ("일제강점기", "식민지", "목포부", "행정구역", "도시 확장", "인구"),
    TopicCategory.ECONOMY_FINANCE: ("동양척식", "호남은행", "금융", "은행", "시장", "상업", "미곡", "면화", "토지"),
    TopicCategory.ARCHITECTURE_HERITAGE: ("일본영사관", "근대역사문화공간", "등록문화유산", "건축", "가옥", "교회", "창고"),
    TopicCategory.EDUCATION_RELIGION: ("학교", "정명여학교", "영흥학교", "목포상업학교", "교육", "선교", "교회", "청년회"),
    TopicCategory.INDEPENDENCE_STUDENT_MOVEMENT: ("독립운동", "학생운동", "3·1운동", "삼일운동", "광주학생", "만세운동"),
    TopicCategory.LABOR_PEASANT_MOVEMENT: ("노동운동", "소작쟁의", "농민운동", "파업", "소작인"),
    TopicCategory.PERSON_HISTORY: ("인물", "생애", "출생", "사망", "본관", "독립운동인명사전"),
    TopicCategory.POST_LIBERATION: ("광복", "해방", "미군정", "한국전쟁", "해방 이후", "광복 이후"),
}

HISTORY_TERMS = tuple(sorted({term for values in TOPIC_TERMS.values() for term in values}))
NAVIGATION_TERMS = ("로그인", "회원가입", "메뉴", "사이트맵", "검색", "이전글", "다음글", "목록", "홈으로")
DATE_PATTERN = re.compile(r"(?:18|19|20)\d{2}(?:[.년/-]\s*\d{1,2})?")
ENTITY_PATTERN = re.compile(r"[가-힣]{2,}(?:학교|은행|회사|항|역|관|청|부|군|면|동|회|조합|교회)")
REPLACEMENT_PATTERN = re.compile(r"[�□■]{2,}")
REPEATED_PATTERN = re.compile(r"(.)\1{9,}")


@dataclass(frozen=True, slots=True)
class ContentQuality:
    body_length: int
    semantic_body_ratio: float
    navigation_ratio: float
    repeated_boilerplate_ratio: float
    title_only: bool
    metadata_only: bool
    ocr_corruption_ratio: float
    suspicious_repeated_characters: bool
    extraction_success: bool
    language: str
    historical_entity_date_density: float
    score: int
    noise: bool


def normalize_body(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_topics(title: str, text: str) -> list[TopicCategory]:
    combined = normalize_body(title + " " + text).lower()
    return [topic for topic, terms in TOPIC_TERMS.items() if any(term.lower() in combined for term in terms)]


def evaluate_content(title: str, text: str, *, extraction_status: str = "success", language: str = "ko") -> ContentQuality:
    compact = normalize_body(text)
    length = len(compact)
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", compact)
    nav_hits = sum(compact.count(term) for term in NAVIGATION_TERMS)
    navigation_ratio = min(1.0, nav_hits * 8 / max(length, 1))
    lines = [normalize_body(line) for line in text.splitlines() if normalize_body(line)]
    repeated_chars = sum(len(line) for line in lines if lines.count(line) > 1)
    repeated_ratio = min(1.0, repeated_chars / max(sum(map(len, lines)), 1))
    corrupt = len(REPLACEMENT_PATTERN.findall(compact)) * 2
    ocr_ratio = min(1.0, corrupt / max(length, 1))
    title_compact = normalize_body(title)
    title_only = bool(compact) and (compact == title_compact or length <= len(title_compact) + 20)
    metadata_labels = sum(compact.count(term) for term in ("제목", "작성자", "등록일", "첨부파일", "분류", "관리번호"))
    metadata_only = length < 300 and metadata_labels >= 3 and len(DATE_PATTERN.findall(compact)) <= 1
    entity_date_count = len(DATE_PATTERN.findall(compact)) + len(ENTITY_PATTERN.findall(compact))
    density = min(1.0, entity_date_count / max(len(tokens), 1))
    semantic_ratio = max(0.0, 1.0 - navigation_ratio - repeated_ratio)
    extraction_success = extraction_status == "success" and bool(compact)
    suspicious = bool(REPEATED_PATTERN.search(compact))

    score = 0
    score += 3 if length >= 500 else 2 if length >= 300 else 1 if length >= 120 else 0
    score += 2 if semantic_ratio >= 0.8 else 1 if semantic_ratio >= 0.6 else 0
    score += 1 if navigation_ratio <= 0.1 else 0
    score += 1 if repeated_ratio <= 0.15 else 0
    score += 1 if ocr_ratio <= 0.01 and not suspicious else 0
    score += 1 if extraction_success else 0
    score += 1 if density >= 0.01 else 0
    if title_only or metadata_only:
        score = min(score, 3)
    noise = navigation_ratio > 0.35 or repeated_ratio > 0.5 or suspicious or ocr_ratio > 0.05
    return ContentQuality(length, round(semantic_ratio, 4), round(navigation_ratio, 4),
                          round(repeated_ratio, 4), title_only, metadata_only,
                          round(ocr_ratio, 4), suspicious, extraction_success,
                          language, round(density, 4), max(0, min(score, 10)), noise)
