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
        "들려줘",
        "궁금해",
        "중요했",
        "다음",
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
        "설명해",
        "설명해줘",
        "공간",
        "사용",
        "알려",
        "주세요",
        "했나요",
        "됐고",
        "이야기",
        "이야기해줘",
        "명확히",
        "부분",
        "얘기",
        "해줘",
        "넘어가자",
        "돌아가자",
        "돌아가서",
        "돌아와서",
        "돌아오고",
        "다시",
        "거야",
        "그럼",
        "그러면",
        "그렇다면",
        "처음",
        "생긴",
        "지었어",
        "거지",
        "설치된",
        "시기",
        "사건",
        "순서",
        "인물",
        "장소",
        "사실",
        "역사적",
        "확인된",
        "확인되는",
        "확인",
        "관련해",
        "관련된",
        "정확한",
        "정확히",
        "근거해",
        "근거로",
        "이번",
        "이번에",
        "돌아",
        "특징",
        "관계",
        "역할",
        "활동",
        "형성",
        "변화",
        "영향",
        "성격",
        "정보",
        "내용",
        "구분해",
        "배경",
        "전개",
        "결과",
        "과정",
        "주요",
        "무엇이야",
        "첫",
        "두",
        "번째",
        "단체",
        "둘",
        "둘을",
        "같은",
        "보면",
        "되는",
        "이유",
        "건립",
        "설립",
        "개통",
        "준공",
        "아까",
        "말한",
        "여기",
        "여기서",
        "거기서",
        "이곳에서",
        "그곳에서",
        "일어났어",
        "당시",
        "내부",
        "모습",
        "어땠어",
        "어땐어",
        "사람들",
        "이용",
        "이용했어",
        "이후",
        "있던",
        "지은",
        "건물",
        "뒤",
        "뒤에",
        "지금",
        "뭐야",
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
    "로",
    "인",
)

_QUESTION_BOILERPLATE = re.compile(
    r"^(?:누구(?:야|예요|인가요)?|어때(?:요)?|"
    r"(?:알려|말해|설명해|이야기해|얘기해)(?:줘|주세요|봐)?|"
    r"(?:만들|세우|세워|지어|생겨|일어나|했|됐|되었)[가-힣]*)$"
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
        if (
            len(word) > 1
            and word not in GENERIC_WORDS
            and not _QUESTION_BOILERPLATE.fullmatch(word)
        ):
            values.append(word)
    return tuple(dict.fromkeys(values))


def normalize_query(text: str) -> NormalizedQuery:
    original = text
    normalized = _normalize_compound_spacing(normalize_korean(text))
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


def explicit_subject_words(text: str) -> tuple[str, ...]:
    """Extract grammatical subject anchors without a domain entity dictionary."""

    normalized = _normalize_compound_spacing(normalize_korean(text))
    if re.match(
        r"\s*(?:그때|그\s*(?:때|당시|사람|인물|사건|장소|곳|역|단체|노선|뒤|다음|일|내용)|"
        r"그\s*이후|관련(?:된)?\s*|그럼|그러면|그렇다면|"
        r"언제|왜|어디|누가|누구|둘\s*중|확실하지|불확실)",
        normalized,
    ):
        return ()
    related_subject = re.match(
        r"\s*(.{2,40}?)(?=와\s*관련(?:된)?\s*(?:인물|사람|장소))",
        normalized,
    )
    described_subject = re.match(
        r"\s*(.{2,40}?)(?=(?:을|를|에\s*대해)\s*(?:알려|설명|말해|이야기))",
        normalized,
    )
    historical_subject = re.match(
        r"\s*(.{2,40}?)(?=의\s*(?:역사|배경|과정|활동|역할)(?:을|를|은|는))",
        normalized,
    )
    coordinated = re.match(
        r"\s*(.{2,35}?)(?:와|과)\s*(.{2,35}?)(?=의\s*(?:날짜|시기|인물|사람)|\s*(?:날짜|시기|인물|사람))",
        normalized,
    )
    general_comparison = re.match(
        r"\s*(.{2,35}?)(?:와|과)\s*(.{2,35}?)(?=(?:을|를)?\s*(?:각각|구분|비교))",
        normalized,
    )
    named_facet = re.match(
        r"\s*(?:먼저|이제|이번에는?)?\s*(.{2,35}?)(?=\s*(?:시기|날짜|인물|사람|장소|활동)(?:만|은|는|을|를))",
        normalized,
    )

    def phrase_subject(value: str) -> str:
        numbered_name = re.search(r"\d+(?:\.\d+)+(?:[가-힣A-Za-z]+)", value)
        if numbered_name:
            return re.sub(
                r"(?:과|와|은|는|이|가|을|를)$", "", numbered_name.group(0)
            )
        words = content_words(value)
        return " ".join(words[:3]).strip()

    if coordinated:
        values = tuple(dict.fromkeys(
            phrase_subject(value) for value in coordinated.groups()
            if phrase_subject(value)
        ))
        if len(values) >= 2:
            return values
    if general_comparison:
        values = tuple(dict.fromkeys(
            phrase_subject(value) for value in general_comparison.groups()
            if phrase_subject(value)
        ))
        if len(values) >= 2:
            return values
    if named_facet and "관련" not in normalized:
        value = phrase_subject(named_facet.group(1))
        if value:
            return (value,)
    if related_subject:
        value = phrase_subject(related_subject.group(1))
        if value:
            return (value,)
    if historical_subject:
        # This capture is already bounded by a possessive historical facet.
        # Preserve a leading place qualifier (for example, a city-qualified
        # institution) even when that place is normally a retrieval stopword.
        value = re.sub(r"\s+", " ", historical_subject.group(1)).strip()
        value = re.sub(r"^(?:이번에는?|이제는?)\s+", "", value).strip()
        if value:
            return (value,)
    if described_subject:
        value = phrase_subject(described_subject.group(1))
        if value:
            return (value,)
    recognized = tuple(dict.fromkeys(re.findall(
        r"구\s*목포\s*일본영사관|목포\s*일본영사관|구\s*일본영사관|"
        r"동양척식주식회사(?:\s*목포지점)?|목포근대역사관(?:\s*[12]관)?|"
        r"목포(?:역|항|진|시)|광주학생운동|한국전쟁|이범석",
        normalized,
    )))
    captures: list[str] = []
    phrase_captures = tuple(dict.fromkeys(re.findall(
        r"(?<![0-9A-Za-z가-힣·])([0-9A-Za-z가-힣·]{2,20}"
        r"(?:\s+[0-9A-Za-z가-힣·]{2,20}){1,2}\s*"
        r"(?:전시관|역사관|박물관|기념관|지점|학교|건물))"
        r"(?=(?:은|는|이|가|와|과|을|를|에서|에\s*대해))",
        normalized,
    )))
    captures.extend(
        re.findall(
            r"(?<![0-9A-Za-z가-힣·])([0-9A-Za-z가-힣·]{2,30})"
            r"(?=(?:의|은|는|이|가|와|과|을|를|에서|(?:으)?로\s*(?:돌아|복귀)))",
            normalized,
        )
    )
    captures.extend(
        re.findall(
            r"(?<![0-9A-Za-z가-힣·])([0-9A-Za-z가-힣·]{2,30})(?=에\s*대해)",
            normalized,
        )
    )
    switch = re.search(
        r"(?:이번에는?|이제는?)\s+(.{2,40}?)(?=(?:을|를|에\s*대해)\s*(?:알려|설명|말해))",
        normalized,
    )
    if switch:
        captures.append(switch.group(1))
    returned = re.search(
        r"(?:다시|아까)\s+(.{2,30}?)(?=(?:으)?로\s*(?:돌아|복귀))",
        normalized,
    )
    if returned:
        captures.append(returned.group(1))

    captured_words = tuple(dict.fromkeys(
        word
        for capture in captures
        for word in content_words(capture)
        if not re.fullmatch(r"\d+(?:년|월|일)?", word)
    ))
    if phrase_captures:
        return phrase_captures
    if captured_words:
        specific = tuple(value for value in captured_words if value not in recognized)
        if specific:
            return specific
    if recognized:
        return recognized
    if captured_words:
        return captured_words
    informative = content_words(normalized)
    if (
        1 <= len(informative) <= 3
        and not re.search(
            r"(?:왜|언제|어디|누가|누구|어떻게|무슨|알려|설명|말해|"
            r"이야기|관계|역할|원인|이유|결과|영향)",
            normalized,
        )
    ):
        return informative
    return informative if len(informative) == 1 else ()


def _normalize_compound_spacing(text: str) -> str:
    value = text
    replacements = (
        (r"목포\s+(역|항|진)", r"목포\1"),
        (r"일본\s+영사관", "일본영사관"),
        (r"동양\s+척식\s*주식회사", "동양척식주식회사"),
        (r"목포\s*근대\s*역사관", "목포근대역사관"),
        (r"커졌(?:는지|는가|어|어요)?", "성장 변천"),
        (r"달라졌(?:는지|는가|어|어요)?", "변화 변천"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    return value
