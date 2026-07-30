"""검색 질의의 원문을 보존하면서 한국어 NFC와 토큰을 정규화한다."""

from __future__ import annotations

import re
from dataclasses import dataclass

from history_chatbot.preprocessing.normalize_korean import normalize_korean


GENERIC_TOKENS = frozenset(
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
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedQuery:
    original: str
    normalized: str
    tokens: tuple[str, ...]
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


def normalize_query(text: str) -> NormalizedQuery:
    original = text
    normalized = normalize_korean(text)
    if not normalized:
        raise ValueError("검색 질문을 입력하세요.")
    tokens = tokenize(normalized)
    return NormalizedQuery(
        original,
        normalized,
        tokens,
        tuple(token for token in tokens if token not in GENERIC_TOKENS),
    )
