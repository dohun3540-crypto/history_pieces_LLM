"""검색 질의의 원문을 보존하면서 한국어 NFC와 토큰을 정규화한다."""

from __future__ import annotations

import re
from dataclasses import dataclass

from history_chatbot.preprocessing.normalize_korean import normalize_korean


GENERIC_WORDS = frozenset(
    {
        "목포",
        "근대",
        "역사",
        "자료",
        "관련",
        "대해",
        "알려줘",
        "알려주세요",
        "무엇",
        "어떤",
        "누구",
        "현재",
        "언제",
        "어떻게",
        "역할",
        "발전",
        "방법",
        "설명",
        "공간",
        "사용",
        "알려",
        "주세요",
        "했나요",
    }
)

_QUERY_SUFFIXES = (
    "이었나요",
    "되었나요",
    "했나요",
    "인가요",
    "되는",
    "으로",
    "에서",
    "에는",
    "에게",
    "부터",
    "까지",
    "처럼",
    "보다",
    "이라고",
    "라고",
    "이며",
    "이고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "인",
)


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    informative_words: tuple[str, ...]
    informative_tokens: tuple[str, ...]


def tokenize(text: str) -> tuple[str, ...]:
    words = [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", normalize_korean(text).lower())
        if len(token) > 1
    ]
    features: list[str] = []
    for word in words:
        features.append(word)
        if re.fullmatch(r"[가-힣]+", word):
            features.extend(
                word[index : index + size]
                for size in (2, 3)
                for index in range(len(word) - size + 1)
            )
    return tuple(dict.fromkeys(features))


def content_words(text: str) -> tuple[str, ...]:
    """Return particle-normalized words with question boilerplate removed."""

    values: list[str] = []
    for raw in re.findall(r"[0-9A-Za-z가-힣]+", normalize_korean(text).lower()):
        if len(raw) <= 1:
            continue
        word = raw
        for suffix in _QUERY_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                word = word[: -len(suffix)]
                break
        if len(word) > 1 and word not in GENERIC_WORDS:
            values.append(word)
    return tuple(dict.fromkeys(values))


def normalize_query(text: str) -> NormalizedQuery:
    original = text
    normalized = normalize_korean(text)
    if not normalized:
        raise ValueError("검색 질문을 입력하세요.")
    tokens = tokenize(normalized)
    words = content_words(normalized)
    informative = tuple(
        dict.fromkeys(feature for word in words for feature in tokenize(word))
    )
    return NormalizedQuery(
        original,
        normalized,
        tokens,
        words,
        informative,
    )
