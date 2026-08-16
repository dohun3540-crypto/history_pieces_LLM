"""Build a minimal, deterministic prompt for untrusted remote inference workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from history_chatbot.models.contract import LLMMessage
from history_chatbot.retrieval.base import RankedChunk


_WINDOWS_PATH = re.compile(r"(?i)(?<![\w])(?:[a-z]:\\|\\\\)[^\s<>|]+")
_UNIX_PATH = re.compile(r"(?<![\w])/(?:home|users|var|tmp|opt|srv|mnt|etc)/[^\s]+", re.I)
_URL = re.compile(r"(?i)\b(?:https?|ssh|git)://[^\s<>'\"]+")
_GIT_SCP = re.compile(r"(?i)\bgit@[^\s:]+:[^\s]+")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(r"(?<!\d)(?:\+?82[- .]?)?0?1[016789](?:[- .]?\d){7,8}(?!\d)")
_PRIVATE_IPV4 = re.compile(
    r"(?<!\d)(?:10(?:\.\d{1,3}){3}|127(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
)
_PRIVATE_IPV6 = re.compile(
    r"(?i)(?<![\w:])(?:::1|f[cd][0-9a-f]{2}(?::[0-9a-f]{0,4}){2,}|"
    r"fe[89ab][0-9a-f](?::[0-9a-f]{0,4}){2,})(?![\w:])"
)
_SESSION_ID = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
_DOCUMENT_ID = re.compile(
    r"(?i)(?<![\w-])(?:mokpo_hist_\d+|[\w-]+::\d+|(?:document|chunk|source)[_-]id\s*[:=]\s*[^\s]+)"
)
_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password|ssh[_-]?key|vpn[_-]?key)"
    r"\s*[:=]\s*[^\s,;]+"
)


@dataclass(frozen=True, slots=True)
class RemotePromptPolicy:
    history_enabled: bool = False
    history_max_turns: int = 1
    context_max_chars: int = 12_000
    chunk_max_chars: int = 1_600
    max_evidence_items: int = 4
    sanitize_enabled: bool = True

    def validate(self) -> None:
        if not 0 <= self.history_max_turns <= 3:
            raise ValueError("LLM_REMOTE_HISTORY_MAX_TURNS는 0~3이어야 합니다.")
        if not 1_024 <= self.context_max_chars <= 100_000:
            raise ValueError("LLM_REMOTE_CONTEXT_MAX_CHARS는 1024~100000이어야 합니다.")
        if not 128 <= self.chunk_max_chars <= 10_000:
            raise ValueError("LLM_REMOTE_CHUNK_MAX_CHARS는 128~10000이어야 합니다.")
        if not 1 <= self.max_evidence_items <= 10:
            raise ValueError("LLM_REMOTE_MAX_EVIDENCE_ITEMS는 1~10이어야 합니다.")


@dataclass(frozen=True, slots=True)
class RemotePrompt:
    system_prompt: str
    user_prompt: str
    messages: tuple[LLMMessage, ...]
    evidence_items: int
    total_chars: int


class EvidenceSupport(StrEnum):
    DIRECT = "direct"
    NEARBY_SUPPORTED = "nearby_supported"
    PARTIAL = "partial"
    RELATED_ONLY = "related_only"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DirectEvidenceAssessment:
    support: EvidenceSupport
    excerpts: tuple[str, ...]
    subject_relevant: bool
    intent_relevant: bool
    direct_sentence_count: int


@dataclass(frozen=True, slots=True)
class GroundedFact:
    subject: str
    intent: str
    source_sentence: str
    source_id: str
    supporting_sentence: str = ""


@dataclass(frozen=True, slots=True)
class VerifiedPersonFact:
    """A person and any relation supported by the same evidence sentence."""

    person: str
    source_id: str
    source_sentence: str
    relation_type: str = ""
    relation_value: str = ""


@dataclass(frozen=True, slots=True)
class GroundedFactPacket:
    subject: str
    intent: str
    facts: tuple[GroundedFact, ...]
    support: EvidenceSupport
    conflicting: bool = False

    @property
    def primary_sentences(self) -> tuple[str, ...]:
        return tuple(fact.source_sentence for fact in self.facts)

    @property
    def supporting_sentences(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            fact.supporting_sentence for fact in self.facts
            if fact.supporting_sentence
            and fact.supporting_sentence != fact.source_sentence
        ))


_PERSON_HANJA = re.compile(
    r"(?<![가-힣])([가-힣]{2,4})\s*"
    r"\(([\u3400-\u9fff\uf900-\ufaff]{2,4})(?:\s*,[^)]*)?\)"
)
_NON_PERSON_KOREAN_SUFFIX = re.compile(
    r"(?:학교|대학교|대학|위원회|협의회|연맹|조합|단체|기관|회사|정부|"
    r"지회|본부|박물관|전시관|역사관|적인|적|회|군|시|도|면|리|동|역)$"
)
_NON_PERSON_HANJA_SUFFIX = re.compile(r"[會校院社團黨部局館寺里洞郡市道國驛領洲]$")
_KOREAN_FAMILY_NAME = frozenset(
    "김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허유남심노하곽성차주우구"
)
_PERSON_BEARING_CONTEXT = re.compile(
    r"관계자|발기인|목사|선교사|박사|교수|회장|대표|위원장|간사|"
    r"총리|장관|대사|학장|총장|참여|활동|주도|조직|결성|역임|"
    r"선출|취임|태어|연행|구속|독립운동|만세운동|모여|발표"
)
_PERSON_TITLE = r"회장|지회장|대표|위원장|간사|총리|장관|대사|학장|총장|교수"
_ROLE_PREDICATE = r"되|되어|됐|맡|역임|취임"
_PERSON_PREFIX_NAME = re.compile(
    r"(?:학생이었던|학생인|주도하였던|주도한|참여한|목사|장로|선교사|박사|교수|회장|대표|위원장|간사|"
    r"총리|장관|대사|학장|총장)\s+([가-힣]{2,4}?)(?=(?:은|는|이|가|을|를|과|와|\s))"
)


def _verified_hanja_people(sentence: str, subject: str) -> tuple[str, ...]:
    compact_subject = re.sub(r"\s+", "", subject)
    if not _PERSON_BEARING_CONTEXT.search(sentence):
        return ()
    values: list[str] = []
    for match in _PERSON_HANJA.finditer(sentence):
        person, hanja = match.groups()
        if re.sub(r"\s+", "", person) == compact_subject:
            continue
        if _NON_PERSON_KOREAN_SUFFIX.search(person):
            continue
        if _NON_PERSON_HANJA_SUFFIX.search(hanja):
            continue
        if person[0] not in _KOREAN_FAMILY_NAME:
            continue
        values.append(person)
    for person in _PERSON_PREFIX_NAME.findall(sentence):
        if re.sub(r"\s+", "", person) == compact_subject:
            continue
        if _NON_PERSON_KOREAN_SUFFIX.search(person):
            continue
        if person[0] not in _KOREAN_FAMILY_NAME:
            continue
        values.append(person)
    return tuple(dict.fromkeys(values))


def verified_person_facts(packet: GroundedFactPacket) -> tuple[VerifiedPersonFact, ...]:
    """Return conservative person facts without joining facts across sources."""

    found: list[VerifiedPersonFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in packet.facts:
        sentence = fact.source_sentence
        candidates = list(_verified_hanja_people(sentence, packet.subject))

        for person in dict.fromkeys(candidates):
            relation_type = ""
            relation_value = ""
            role = re.search(
                rf"{re.escape(person)}(?:\([^)]*\))?(?:은|는|이|가)?"
                rf"[^.!?。！？]{{0,35}}?({_PERSON_TITLE})(?:이|가|을|를|으로)?\s*"
                rf"(?:{_ROLE_PREDICATE})",
                sentence,
            )
            if role:
                relation_type = "role"
                relation_value = role.group(1)
            elif re.search(
                rf"{re.escape(person)}(?:\([^)]*\))?[^.!?。！？]{{0,45}}"
                r"(?:훈장|포장)[^.!?。！？]{0,20}(?:추서|수훈|받)",
                sentence,
            ):
                relation_type = "award"
                relation_value = "award"
            key = (person, fact.source_id, sentence)
            if key not in seen:
                seen.add(key)
                found.append(VerifiedPersonFact(
                    person=person,
                    source_id=fact.source_id,
                    source_sentence=sentence,
                    relation_type=relation_type,
                    relation_value=relation_value,
                ))
    return tuple(found)


def build_grounded_fact_packet(
    chunks: Sequence[RankedChunk], *, subject: str, intent: str, question: str = ""
) -> GroundedFactPacket:
    """Select source-owned facts without joining relations across sentences."""

    assessment = assess_direct_evidence(
        chunks, subject=subject, intent=intent, question=question
    )
    candidates: list[GroundedFact] = []
    for item in _subject_scoped_chunks(chunks, subject):
        clauses = _grounded_clauses(item.chunk.text)
        title_supports_subject = _subject_matches(item.chunk.title, subject)
        subject_indexes = {
            index for index, clause in enumerate(clauses)
            if _subject_matches(clause, subject)
        }
        for index, sentence in enumerate(clauses):
            sentence_has_subject = _subject_matches(sentence, subject)
            local_subject_support = bool(
                intent == "people"
                and any(abs(index - subject_index) <= 1 for subject_index in subject_indexes)
                and _verified_hanja_people(sentence, subject)
            )
            local_place_support = bool(
                intent == "place"
                and any(abs(index - subject_index) <= 1 for subject_index in subject_indexes)
                and _INTENT_EVIDENCE["place"].search(sentence)
            )
            title_scoped_people = False
            title_scoped_role = False
            title_scoped_place = False
            if (
                not sentence_has_subject
                and not title_supports_subject
                and not local_subject_support
                and not local_place_support
            ):
                continue
            # A document title can safely restore the omitted subject for a
            # definition or dated lifecycle fact.  It cannot establish an
            # actor, cause, role, or effect relation inside an unrelated
            # archive sentence.
            if (
                not sentence_has_subject
                and not local_subject_support
                and not local_place_support
                and intent not in {"overview", "time", "current"}
            ):
                title_scoped_people = bool(
                    intent == "people"
                    and title_supports_subject
                    and (
                        re.search(r"관계자|회장|대표|위원장|간사|학장|총장", sentence)
                        or _verified_hanja_people(sentence, subject)
                    )
                )
                title_scoped_role = bool(
                    intent == "role"
                    and title_supports_subject
                    and re.search(
                        r"활동|참여|주도|조직|결성|역임|선출|취임|"
                        r"독립운동|노동운동|농민운동|만세운동",
                        sentence,
                    )
                )
                title_scoped_place = bool(
                    intent == "place"
                    and title_supports_subject
                    and _INTENT_EVIDENCE["place"].search(sentence)
                )
                if not (title_scoped_people or title_scoped_role or title_scoped_place):
                    continue
            if (
                not sentence_has_subject
                and not title_scoped_people
                and not title_scoped_role
                and not title_scoped_place
                and not local_place_support
                and re.match(r"\s*(?!이곳|그곳|해당)[가-힣A-Za-z\s]{2,24}(?:은|는|이|가)\s", sentence)
            ):
                continue
            if (not title_scoped_role and _competitor_matches(sentence, subject)) or not _intent_matches(
                sentence, subject=subject, intent=intent, question=question
            ):
                continue
            if not _usable_fact_sentence(sentence):
                continue
            supporting = ""
            if index + 1 < len(clauses):
                candidate = clauses[index + 1]
                if not _competitor_matches(candidate, subject):
                    supporting = candidate
            normalized_sentence = sentence
            if not sentence_has_subject:
                normalized_sentence = f"{subject} — {sentence}"
            candidates.append(GroundedFact(
                subject=subject,
                intent=intent,
                source_sentence=normalized_sentence,
                source_id=str(item.chunk.payload.get("source_id", item.chunk.document_id)),
                supporting_sentence=supporting,
            ))
    def quality(fact: GroundedFact) -> tuple[int, int, int, int, int, int]:
        sentence = fact.source_sentence
        noise = len(re.findall(
            r"전시|체험|사진|이정표|운행회수|승강객수|승차인원|통계|상세",
            sentence,
        ))
        direct_start = int(re.sub(r"\s+", "", sentence).startswith(
            re.sub(r"\s+", "", subject)
        ))
        identity = int(bool(re.search(
            r"(?:이다|입니다|곳이다|건물이다|기관이다|회사였다|철도|항구)", sentence
        )))
        people_strength = (
            len(_verified_hanja_people(sentence, subject))
            if intent == "people" else 0
        )
        time_strength = (
            int(bool(re.search(r"개통|개항|건립|설립|준공|완공|창립", sentence)))
            if intent == "time" else 0
        )
        place_strength = 0
        if intent == "place":
            # Prefer an administrative hierarchy (for example ``목포 양동``)
            # over a bare city/region mention or a merely related venue.  A
            # boolean location score let earlier generic sentences crowd the
            # more precise title-scoped location out of the two-fact packet.
            if re.search(
                r"[가-힣]{2,12}(?:시|군|도)?\s+[가-힣]{1,12}(?:동|리|읍|면)(?:과|와|,|\s|에|에서)",
                sentence,
            ):
                place_strength = 3
            elif re.search(
                r"[가-힣]{1,12}(?:동|리|읍|면)(?:과|와|,|\s|에|에서)", sentence
            ):
                place_strength = 2
            elif re.search(
                r"소재|위치|캠퍼스|항구|[가-힣]{1,12}(?:군|시|도)(?:과|와|,|\s|에|에서)",
                sentence,
            ):
                place_strength = 1
        return (
            noise,
            -people_strength,
            -time_strength,
            -place_strength,
            -direct_start,
            -identity,
        )

    facts = sorted(candidates, key=quality)[:2]
    signatures: dict[str, set[str]] = {}
    if intent == "time":
        for fact in facts:
            if re.search(r"신축|증축|재건|개축|이전", fact.source_sentence):
                continue
            if re.search(r"구간|차례로", fact.source_sentence):
                continue
            predicate = next((
                value for value in ("건립", "준공", "개통", "개항", "설립", "개관", "영업")
                if value in fact.source_sentence
            ), "")
            dates = set(re.findall(r"(?:18|19|20)\d{2}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?", fact.source_sentence))
            if predicate and dates:
                signatures.setdefault(predicate, set()).update(dates)
    conflicting = any(len(values) > 1 for values in signatures.values())
    packet_support = EvidenceSupport.DIRECT if facts else assessment.support
    return GroundedFactPacket(subject, intent, tuple(facts), packet_support, conflicting)


_SECTION_BOUNDARY = re.compile(
    r"(?:정의|개설|개관|변천|형성\s*및\s*변천|구성|현황|내용|연원\s*및\s*변천|"
    r"역사적\s*배경|자연환경|위치|특징)\s*닫기"
)


def _grounded_clauses(text: str) -> list[str]:
    """Recover source-owned clauses from encyclopedia/archive section prose."""

    sectioned = _SECTION_BOUNDARY.sub("\n", text)
    clauses: list[str] = []
    for sentence in _sentences(sectioned):
        pieces = re.split(
            r"(?<=다)[,;]\s+|(?<=했다)[,;]\s+|(?<=하였다)[,;]\s+",
            sentence,
        )
        for piece in pieces:
            value = piece.strip(" \t\r\n-·")
            if value:
                clauses.append(value)
    return clauses


def _usable_fact_sentence(sentence: str) -> bool:
    if re.search(
        r"제목\s*:|상세\s*URL|화면묘사|자막|정의\s*닫기|형성\s*및\s*변천\s*닫기|"
        r"(?:^|\s)#\S+|생산국가|생산기관|촬영장소|자료시기|"
        r"서비스가\s*임시\s*중단|사실\s*확인\s*및\s*보완|주관적\s*서술\s*문제",
        sentence,
    ):
        return False
    if len(sentence) < 12 or len(sentence) > 700:
        return False
    for left, right in (("(", ")"), ("[", "]"), ("“", "”")):
        if sentence.count(left) != sentence.count(right):
            return False
    if sentence.count('"') % 2:
        return False
    return True


def sanitize_remote_text(value: str, *, enabled: bool = True) -> str:
    """Conservatively redact common local identifiers without pretending to be DLP."""

    text = str(value)
    if not enabled:
        return text.strip()
    replacements = (
        (_WINDOWS_PATH, "[경로 제거]"),
        (_UNIX_PATH, "[경로 제거]"),
        (_URL, "[URL 제거]"),
        (_GIT_SCP, "[Git 주소 제거]"),
        (_EMAIL, "[이메일 제거]"),
        (_PHONE, "[전화번호 제거]"),
        (_PRIVATE_IPV4, "[내부 주소 제거]"),
        (_PRIVATE_IPV6, "[내부 주소 제거]"),
        (_SESSION_ID, "[세션 식별자 제거]"),
        (_DOCUMENT_ID, "[내부 식별자 제거]"),
        (_SECRET, "[비밀정보 제거]"),
    )
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    return text.strip()


def serialize_remote_prompt(
    *,
    system_prompt: str,
    user_query: str,
    chunks: Sequence[RankedChunk],
    history: Sequence[tuple[str, str]] = (),
    policy: RemotePromptPolicy | None = None,
    question_subject: str = "",
    question_intent: str = "overview",
    transform: bool = False,
) -> RemotePrompt:
    """Serialize only anonymous evidence labels and bounded text for a remote worker."""

    resolved = policy or RemotePromptPolicy()
    resolved.validate()
    clean_system = sanitize_remote_text(system_prompt, enabled=resolved.sanitize_enabled)
    clean_query = sanitize_remote_text(user_query, enabled=resolved.sanitize_enabled)
    if not clean_system or not clean_query:
        raise ValueError("remote-safe prompt의 시스템 지침 또는 질문이 비어 있습니다.")

    clean_subject = sanitize_remote_text(
        question_subject, enabled=resolved.sanitize_enabled
    )
    clean_intent = re.sub(r"[^a-z_]", "", question_intent.casefold()) or "overview"
    request_label = "같은 기록을 표현만 바꿔 답해 주세요." if transform else "기록으로 질문에 답해 주세요."
    question_section = (
        f"{request_label}\n"
        f"{clean_subject or '직전 주제'}에 관한 {clean_intent} 질문입니다.\n"
        f"{clean_query}"
    )
    fixed_size = (
        len(clean_system)
        + len(question_section)
        + len("[검색 근거]\n")
        + len("\n\n")
    )
    if fixed_size >= resolved.context_max_chars:
        raise ValueError("remote-safe context 상한이 시스템 지침과 현재 질문보다 작습니다.")

    evidence_blocks: list[str] = []
    seen_evidence: set[str] = set()
    scoped_chunks = _subject_scoped_chunks(chunks, clean_subject)
    global_assessment = assess_direct_evidence(
        scoped_chunks,
        subject=clean_subject,
        intent=clean_intent,
        question=clean_query,
    )
    fact_packet = build_grounded_fact_packet(
        scoped_chunks,
        subject=clean_subject,
        intent=clean_intent,
        question=clean_query,
    )
    for item in scoped_chunks:
        if len(evidence_blocks) >= resolved.max_evidence_items:
            break
        local_assessment = assess_direct_evidence(
            (item,),
            subject=clean_subject,
            intent=clean_intent,
            question=clean_query,
        )
        if (
            global_assessment.support in {EvidenceSupport.DIRECT, EvidenceSupport.PARTIAL}
            and local_assessment.support
            not in {EvidenceSupport.DIRECT, EvidenceSupport.PARTIAL}
        ):
            continue
        title = _truncate(
            sanitize_remote_text(item.chunk.title, enabled=resolved.sanitize_enabled), 160
        )
        publisher = _truncate(
            sanitize_remote_text(item.chunk.publisher, enabled=resolved.sanitize_enabled), 160
        )
        raw_evidence = sanitize_remote_text(
            item.chunk.text, enabled=resolved.sanitize_enabled
        )
        evidence = _truncate(
            _evidence_excerpt(
                raw_evidence,
                subject=clean_subject,
                intent=clean_intent,
                question=clean_query,
            ),
            resolved.chunk_max_chars,
        )
        if not evidence or evidence in seen_evidence:
            continue
        seen_evidence.add(evidence)
        label = len(evidence_blocks) + 1
        prefix = f"[자료{label}]\n제목: {title or '(제목 없음)'}\n제공 기관: {publisher or '(기관 없음)'}\n근거: "
        remaining = resolved.context_max_chars - fixed_size - sum(
            len(block) + 2 for block in evidence_blocks
        ) - len(prefix)
        if remaining < 32:
            break
        evidence_blocks.append(prefix + _truncate(evidence, remaining))

    if not evidence_blocks:
        raise ValueError("remote-safe prompt에 전송할 수 있는 근거가 없습니다.")
    fact_sections: list[str] = []
    if fact_packet.primary_sentences:
        fact_sections.append(
            "[PRIMARY FACT]\n"
            + "\n".join(f"- {value}" for value in fact_packet.primary_sentences)
        )
    if fact_packet.supporting_sentences:
        fact_sections.append(
            "[SUPPORTING CONTEXT]\n"
            + "\n".join(
                f"- {value}" for value in fact_packet.supporting_sentences[:2]
            )
        )
    user_prompt = (
        "\n\n".join(fact_sections) + "\n\n" + question_section
        if fact_sections
        else "[검색 근거]\n" + "\n\n".join(evidence_blocks) + "\n\n" + question_section
    )

    messages: list[LLMMessage] = []
    if resolved.history_enabled and resolved.history_max_turns:
        for user, assistant in history[-resolved.history_max_turns :]:
            pair = (
                LLMMessage(
                    "user",
                    _truncate(
                        sanitize_remote_text(user, enabled=resolved.sanitize_enabled), 500
                    ),
                ),
                LLMMessage(
                    "assistant",
                    _truncate(
                        sanitize_remote_text(assistant, enabled=resolved.sanitize_enabled), 500
                    ),
                ),
            )
            pair_size = sum(len(message.content) for message in pair)
            if _total_chars(clean_system, user_prompt, messages) + pair_size > resolved.context_max_chars:
                continue
            messages.extend(pair)

    total = _total_chars(clean_system, user_prompt, messages)
    return RemotePrompt(clean_system, user_prompt, tuple(messages), len(evidence_blocks), total)


_CONFUSABLE_ENTITY = re.compile(
    r"구\s*목포\s*일본영사관|목포\s*일본영사관|구\s*일본영사관|일본영사관|"
    r"동양척식주식회사(?:\s*목포지점)?|목포근대역사관\s*[12]관"
)

_SUBJECT_ALIAS_GROUPS = (
    (
        re.compile(r"(?:구\s*)?목포\s*일본영사관|일본영사관|목포근대역사관\s*1관"),
        re.compile(r"동양척식주식회사(?:\s*목포지점)?|목포근대역사관\s*2관"),
    ),
    (
        re.compile(r"동양척식주식회사(?:\s*목포지점)?|목포근대역사관\s*2관"),
        re.compile(r"(?:구\s*)?목포\s*일본영사관|일본영사관|목포근대역사관\s*1관"),
    ),
)


def _subject_scoped_chunks(
    chunks: Sequence[RankedChunk], subject: str
) -> tuple[RankedChunk, ...]:
    """Drop later mixed-facility chunks only when the best chunk is subject-clean."""

    values = tuple(chunks)
    if not subject or not values:
        return values
    subject_compact = re.sub(r"\s+", "", subject)

    for target_pattern, competitor_pattern in _SUBJECT_ALIAS_GROUPS:
        if not target_pattern.search(subject):
            continue
        clean = tuple(
            item for item in values
            if target_pattern.search(f"{item.chunk.title} {item.chunk.text[:1600]}")
            and not competitor_pattern.search(f"{item.chunk.title} {item.chunk.text[:1600]}")
        )
        if clean:
            return clean

    def competitors(item: RankedChunk) -> set[str]:
        found = {
            re.sub(r"\s+", "", value)
            for value in _CONFUSABLE_ENTITY.findall(
                f"{item.chunk.title} {item.chunk.text[:900]}"
            )
        }
        return {
            value for value in found
            if value not in subject_compact and subject_compact not in value
        }

    if competitors(values[0]):
        return values
    filtered = tuple(item for item in values if not competitors(item))
    return filtered or values[:1]


_INTENT_EVIDENCE = {
    "time": re.compile(r"(?:18|19|20)\d{2}년|건립|설립|개통|준공|완공"),
    "cause": re.compile(r"이유|원인|배경|목적|위해|때문|계기로|필요"),
    "people": re.compile(
        r"인물|사람|총리|장관|대사|교수|학생|학장|총장|참석|참여|주도|관계자|회장|대표|위원장|간사"
    ),
    "place": re.compile(
        r"장소|지역|소재|위치|캠퍼스|본부|에서|에\s*있는|"
        r"(?:[가-힣]+(?:동|리|읍|면|군|시|도))(?:과|와|,|\s|에|에서)"
    ),
    "result": re.compile(r"이후|뒤|결과|영향|변화|사용|개관|지정"),
    "role": re.compile(
        r"역할|업무|수행|관리|운영|담당|수탈|매입|대부|징수|개발|"
        r"활동|참여|주도|조직|결성|역임|선출|취임"
    ),
    "current": re.compile(r"현재|지금|오늘날|개관|박물관|역사관|사용"),
}

_IDENTITY_EVIDENCE = re.compile(
    r"(?:은|는|이|가)\s*.{0,50}(?:역|항구|건물|기관|회사|시설|장소|곳|인물|총리|장관|철도|산|섬|학교|목사|독립운동가)"
    r"|(?:역|항구|건물|기관|회사|시설|장소|곳|인물|총리|장관|철도|산|섬|학교|목사|독립운동가)(?:이다|입니다|였다|였습니다|[.]?$)"
)
_HISTORICAL_ROLE = re.compile(r"수탈|매입|대부|식민|토지|농민|징수")
_CURRENT_MUSEUM = re.compile(r"현재|지금|박물관|역사관|전시|관람|체험|포토존|전시실")


def assess_direct_evidence(
    chunks: Sequence[RankedChunk], *, subject: str, intent: str, question: str = ""
) -> DirectEvidenceAssessment:
    """Separate topical relevance from answer-bearing evidence."""

    if not chunks or not subject:
        return DirectEvidenceAssessment(EvidenceSupport.NONE, (), False, False, 0)
    excerpts: list[str] = []
    subject_relevant = False
    intent_relevant = False
    direct_count = 0
    local_count = 0
    for item in _subject_scoped_chunks(chunks, subject):
        sentences = _sentences(item.chunk.text)
        subject_indexes = {
            index for index, sentence in enumerate(sentences)
            if _subject_matches(sentence, subject)
        }
        subject_relevant = subject_relevant or bool(subject_indexes)
        for index, sentence in enumerate(sentences):
            intent_match = not _competitor_matches(sentence, subject) and _intent_matches(
                sentence, subject=subject, intent=intent, question=question
            )
            intent_relevant = intent_relevant or intent_match
            if not intent_match:
                continue
            if index in subject_indexes:
                direct_count += 1
                excerpts.append(sentence)
                if index + 1 < len(sentences):
                    excerpts.append(sentences[index + 1])
                continue
            if any(abs(index - subject_index) <= 1 for subject_index in subject_indexes):
                local_count += 1
                for neighbor in (index - 1, index, index + 1):
                    if 0 <= neighbor < len(sentences):
                        excerpts.append(sentences[neighbor])
    unique = tuple(dict.fromkeys(value for value in excerpts if value))
    if direct_count:
        support = EvidenceSupport.DIRECT
    elif local_count:
        support = EvidenceSupport.PARTIAL
    elif subject_relevant:
        support = EvidenceSupport.RELATED_ONLY
    else:
        support = EvidenceSupport.NONE
    return DirectEvidenceAssessment(
        support, unique, subject_relevant, intent_relevant, direct_count
    )


def _sentences(text: str) -> list[str]:
    protected = re.sub(
        r"\b([A-Z])\.(?=\s*[A-Z][a-z])", r"\1<INITIAL_DOT>", text
    )
    values = [
        value.strip().replace("<INITIAL_DOT>", ".")
        for value in re.split(r"(?<=[.!?。！？])\s+|\n+", protected)
        if value.strip()
    ]
    return values


def _subject_aliases(subject: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", subject)
    if re.search(r"일본영사관|목포근대역사관\s*1관", subject):
        return (compact, "일본영사관", "목포근대역사관1관")
    if re.search(r"동양척식주식회사|목포근대역사관\s*2관", subject):
        return (compact, "동양척식주식회사", "목포근대역사관2관")
    return (compact,)


def _subject_matches(sentence: str, subject: str) -> bool:
    compact_sentence = re.sub(r"\s+", "", sentence)
    return any(alias and alias in compact_sentence for alias in _subject_aliases(subject))


def _competitor_matches(sentence: str, subject: str) -> bool:
    if re.search(r"일본영사관|목포근대역사관\s*1관", subject):
        return bool(re.search(r"동양척식주식회사|목포근대역사관\s*2관", sentence))
    if re.search(r"동양척식주식회사|목포근대역사관\s*2관", subject):
        return bool(re.search(r"일본영사관|목포근대역사관\s*1관", sentence))
    if re.search(r"비교|뒤지지|견주|영향력|구별", sentence):
        leading = re.sub(r"\s+", "", sentence)[: len(re.sub(r"\s+", "", subject))]
        if leading != re.sub(r"\s+", "", subject):
            return True
    comparison_body = re.sub(
        rf"^\s*{re.escape(subject)}\s*(?:[-—:])\s*", "", sentence, count=1
    )
    compared_event = re.match(
        r"\s*([0-9.가-힣]+(?:운동|사건))", comparison_body
    )
    if compared_event:
        candidate = re.sub(r"\s+", "", compared_event.group(1))
        target = re.sub(r"\s+", "", subject)
        sentence_compact = re.sub(r"\s+", "", comparison_body)
        if candidate != target and (
            re.search(r"비교|뒤지지|견주|영향력|구별", sentence)
            or target not in sentence_compact
        ):
            return True
    return False


def _intent_matches(
    sentence: str, *, subject: str, intent: str, question: str = ""
) -> bool:
    if intent in {"overview", "transform"}:
        if re.search(
            r"운행회수|승강객수|승차인원|전시물|제\s*\d\s*전시실|"
            r"사진|이정표|포토존|체험|게임",
            sentence,
        ):
            return False
        return bool(_IDENTITY_EVIDENCE.search(sentence))
    pattern = _INTENT_EVIDENCE.get(intent)
    if pattern is None or not pattern.search(sentence):
        return False
    if intent == "people" and re.search(r"관련(?:된)?\s*인물|인물은|누구|누가", question):
        # A generic collective does not answer who was involved. Require an
        # identifiable name or a person attached to a named office.
        named_people = _verified_hanja_people(sentence, subject)
        office_relation = re.search(
            r"(?:회장|대표|위원장|간사|총리|장관|대사|학장|총장)\s*[가-힣]{2,4}|"
            r"[가-힣]{2,4}\s*(?:회장|대표|위원장|간사|총리|장관|대사|학장|총장)",
            sentence,
        )
        return bool(named_people or office_relation)
    if intent == "time":
        if re.search(r"짓|지어|세워|만들|건립|완공|준공", question):
            return bool(
                re.search(r"(?:18|19|20)\d{2}년", sentence)
                and re.search(r"짓|지어|세워|만들|건립|완공|준공", sentence)
            )
        if "개항" in question:
            return bool(
                re.search(r"(?:18|19|20)\d{2}년", sentence) and "개항" in sentence
            )
        if "개통" in question:
            return bool(
                re.search(r"(?:18|19|20)\d{2}년", sentence) and "개통" in sentence
            )
    if intent == "cause":
        if re.search(r"중요|의미", question):
            return bool(re.search(r"중요|의미|가치|의의", sentence))
        if re.search(r"짓|지어|세워|만들|건립|설립", question):
            return bool(
                re.search(r"위해|때문|목적|배경|원인|따라|계기로|필요", sentence)
                and re.search(r"짓|지어|세워|만들|건립|설립", sentence)
            )
        if re.search(r"가|왔|내려|도착|참석", question):
            return bool(
                re.search(r"위해|때문|목적|참석", sentence)
                and re.search(r"가|왔|내려|도착|참석", sentence)
            )
    if intent == "result" and re.search(r"영향|어떤\s*변화", question):
        return bool(re.search(r"영향|변화|증가|발전|확대|성장|형성|촉진", sentence))
    if (
        intent == "role"
        and re.search(r"동양척식주식회사", subject)
        and _CURRENT_MUSEUM.search(sentence)
        and not _HISTORICAL_ROLE.search(sentence)
    ):
        return False
    if intent == "cause" and re.search(r"전시|살펴볼|체험|게임", sentence):
        return False
    if intent == "role" and re.search(
        r"어떤\s*활동|무슨\s*일|무엇을\s*했|뭘\s*했", question
    ):
        return bool(re.search(
            r"업무|수행|관리|운영|담당|활동|참여|주도|조직|결성|역임|선출|취임|"
            r"독립운동|노동운동|농민운동|만세운동",
            sentence,
        ))
    return True


def _evidence_excerpt(
    text: str, *, subject: str, intent: str, question: str = ""
) -> str:
    """Put subject- and intent-bearing source sentences first without inventing facts."""

    sentences = _sentences(text)
    if len(sentences) < 3:
        return text
    compact_subject = re.sub(r"\s+", "", subject)
    aliases: tuple[str, ...] = (
        ("일본영사관", "목포근대역사관1관")
        if re.search(r"일본영사관|목포근대역사관\s*1관", subject)
        else ("동양척식주식회사", "목포근대역사관2관")
        if re.search(r"동양척식주식회사|목포근대역사관\s*2관", subject)
        else ()
    )
    subject_indexes: set[int] = set()
    intent_indexes: set[int] = set()
    for index, sentence in enumerate(sentences):
        compact_sentence = re.sub(r"\s+", "", sentence)
        if bool(
            compact_subject and compact_subject in compact_sentence
        ) or any(re.sub(r"\s+", "", alias) in compact_sentence for alias in aliases):
            subject_indexes.add(index)
        if not _competitor_matches(sentence, subject) and _intent_matches(
            sentence, subject=subject, intent=intent, question=question
        ):
            intent_indexes.add(index)
    direct = subject_indexes & intent_indexes
    chosen: list[int] = []
    def priority(index: int) -> tuple[int, int, int]:
        sentence = sentences[index]
        noise = len(re.findall(r"사진|정면|좌측면|우측면|항공|해설\s*안내", sentence))
        specific = int(bool(re.search(r"착공|완공|건립|설립|개통|준공", sentence)))
        return (-specific, noise, index)
    if direct:
        chosen.extend(sorted(direct, key=priority)[:3])
    else:
        local_intent = {
            index for index in intent_indexes
            if any(abs(index - subject_index) <= 1 for subject_index in subject_indexes)
        }
        if local_intent:
            chosen.extend(sorted(local_intent, key=priority)[:2])
            chosen.extend(
                subject_index for subject_index in subject_indexes
                if any(abs(subject_index - index) <= 1 for index in local_intent)
            )
        else:
            chosen.extend(sorted(subject_indexes, key=priority)[:2])
    if not chosen:
        return text
    for index in tuple(chosen):
        if index in direct and index + 1 < len(sentences):
            chosen.append(index + 1)
    return " ".join(sentences[index] for index in dict.fromkeys(chosen))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _total_chars(system_prompt: str, user_prompt: str, messages: Sequence[LLMMessage]) -> int:
    return len(system_prompt) + len(user_prompt) + sum(len(item.content) for item in messages)
