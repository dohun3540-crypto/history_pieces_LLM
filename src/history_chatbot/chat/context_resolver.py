"""Resolve conversational references without treating chat history as evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from history_chatbot.chat.session import ChatSession


PLACE_LABELS = {
    "mokpo-station": "목포역", "mokpo_station": "목포역", "mokpo-yeok": "목포역",
    "mokpo-port": "목포항", "mokpo_port": "목포항",
    "samhakdo": "삼학도", "yudalsan": "유달산",
}
_PLACEHOLDER_CONTEXT = re.compile(
    r"^(?:(?:demo|test|dummy|placeholder)(?:[-_ ](?:place|piece))?"
    r"|(?:place|piece)[-_ ](?:demo|test|dummy|placeholder))"
    r"(?:[-_ ][a-z0-9]+)*$",
    re.IGNORECASE,
)
FOLLOWUP = re.compile(
    r"그\s*(?:때|당시|사람|사건|학교|건물|역|회사|곳|장소|뒤|과정|자료)|"
    r"여기|이곳|거기|아까|방금|이후에는?|왜\s*(?:그랬|왔|온)|"
    r"누가\s*(?:참여|관여|주도)|그래서|그건|"
    r"관련(?:된)?\s*(?:인물|사람)|"
    r"그\s*(?:이유|결과|영향|의미|배경)|"
    r"그럼|그러면|그렇다면|이어서|계속해서|또\s*있어|다른\s*건"
)
ELLIPTICAL_FOLLOWUP = re.compile(
    r"(?:(?:그럼|그러면|그렇다면)\s*)?(?:좀\s*)?(?:더\s*)?"
    r"(?:왜|언제|어디서|누가|누구|어떻게|무슨\s*이유(?:로)?|"
    r"이유|결과|영향|의미|배경|그다음|다음)"
    r"(?:은|는|이|가|부터|까지|였어|였어요|인가요|야|예요|지|요|"
    r"\s*(?:알려\s*줘|설명해\s*줘|말해\s*줘|지었어|만들었어|왔던\s*거야))?[?.!]*|"
    r"(?:좀\s*)?(?:더\s*)?(?:쉽게|자세히|짧게|간단히|다시)\s*"
    r"(?:설명해|알려|말해|요약해)(?:\s*줘|\s*주세요|요)?[?.!]*|"
    r"(?:또\s*있어|다른\s*건|관련\s*(?:인물|사람)은?)[?.!]*",
    re.IGNORECASE,
)
PLACE_REFERENCE = re.compile(r"여기|이곳|거기|그곳|그\s*장소|이\s*장소")
PERSON_REFERENCE = re.compile(r"(?:그|이|저)\s*(?:사람|인물)")
EXPLICIT_PLACE = re.compile(
    r"구\s*목포\s*일본영사관|구\s*일본영사관|목포(?:역|항|부|진|해관|세관)|삼학도|유달산|"
    r"무안감리서|동양척식주식회사(?:\s*목포지점)?|호남은행|일본영사관|목포"
)
EXPLICIT_PERSON = re.compile(
    r"(?<![가-힣])([가-힣]{2,4})(?=(?:은|는|이|가)"
    r"[^?.!]{0,30}(?:어떤\s*사람|누구|참석|도착|왔|온\s*(?:이유|사건)|갔|간\s*이유))"
)
GENERIC_PEOPLE_FOLLOWUP = re.compile(
    r"관련(?:된)?\s*(?:인물|사람)(?:은|이)?\s*누구(?:인가요|예요|야)?[?.!]?$"
)
_PERSON_STOPWORDS = {
    "사람", "인물", "누구", "당시", "당시에", "그때", "여기", "결과",
    "언제였어", "왜", "어떻게", "전시관", "건물", "장소",
}
EVENT = re.compile(r"(?:학생운동|독립운동|노동운동|개항|시위|파업|사건)")
PERIOD = re.compile(r"(?:18|19|20)\d{2}년|일제강점기|대한제국|근대")
TRANSFORMATION = re.compile(
    r"(?:좀\s*)?(?:더\s*)?(?:쉽게|짧게|간단히|한\s*문장으로|다시)\s*"
    r"(?:설명해|알려|말해|요약해|정리해)(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"아까\s*답(?:을|변을)?\s*다시\s*(?:설명해|말해)(?:\s*줘|\s*주세요)?[?.!]*$|"
    r"(?:정리해서|요약해서)\s*(?:알려|말해)(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"(?:초등학생|아이)(?:도|가)?\s*이해할\s*수\s*있게\s*"
    r"(?:설명해|알려|말해)(?:\s*줘|\s*주세요|요)?[?.!]*$",
    re.IGNORECASE,
)
DETAIL_EXPANSION = re.compile(
    r"(?:(?:좀|조금)\s*)?더\s*(?:자세히\s*)?(?:설명해|알려|말해)"
    r"(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"(?:그\s*(?:사건|행사|내용)|관련\s*내용)(?:에\s*대해|도)?\s*더\s*"
    r"(?:자세히\s*)?(?:설명해|알려|말해)?(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"(?:배경|결과|과정|영향)(?:과|와|까지|도|을|를|부터)?[^?.!]{0,24}"
    r"(?:자세히|구체적으로|더\s*알려)[^?.!]*[?.!]*$|"
    r"(?=[^?.!]*(?:배경|결과|과정|영향))[^?.!]*"
    r"(?:자세히|구체적으로|더\s*알려)[^?.!]*[?.!]*$|"
    r"구체적으로\s*(?:설명해|알려|말해)(?:\s*줘|\s*주세요|요)?[?.!]*$",
    re.IGNORECASE,
)
CORRECTION = re.compile(
    r"(?:^|\s)(?:아니|그게\s*아니라|내가\s*묻는\s*건|내\s*말은|"
    r"그\s*사람이\s*아니라|다시\s*확인)",
    re.IGNORECASE,
)


class ConversationRequestKind(StrEnum):
    INDEPENDENT = "independent"
    FACTUAL_FOLLOWUP = "factual_followup"
    TRANSFORM_PREVIOUS_ANSWER = "transform_previous_answer"
    EXPAND_PREVIOUS_ANSWER = "expand_previous_answer"
    CORRECTION = "correction"


def is_placeholder_context(value: str | None) -> bool:
    """Return whether a journey identifier is non-authoritative demo/test state."""

    if value is None:
        return True
    normalized = value.strip().casefold()
    if not normalized or normalized == "unknown":
        return True
    return bool(_PLACEHOLDER_CONTEXT.fullmatch(normalized))


def context_label(value: str | None) -> str:
    if is_placeholder_context(value):
        return ""
    assert value is not None
    normalized = value.strip().casefold()
    direct = PLACE_LABELS.get(normalized)
    if direct:
        return direct
    for prefix, label in PLACE_LABELS.items():
        if normalized.startswith(prefix + "-"):
            return label
    return value.replace("_", " ").replace("-", " ")


@dataclass(frozen=True, slots=True)
class ResolvedContext:
    current_user_query: str
    resolved_question: str
    search_query: str
    active_place: str
    active_piece: str
    active_topic: str
    recent_entities: tuple[str, ...]
    recent_people: tuple[str, ...]
    recent_event: str
    recent_period: str
    followup_resolved: bool
    request_kind: ConversationRequestKind
    needs_new_evidence: bool


class ConversationContextResolver:
    def resolve(
        self,
        query: str,
        session: ChatSession,
        *,
        current_place_id: str | None,
        current_piece_id: str | None,
    ) -> ResolvedContext:
        journey_place = context_label(current_place_id)
        journey_piece = context_label(current_piece_id)
        conversational_place = context_label(session.active_place)
        conversational_piece = context_label(session.active_piece)
        conversational_topic = (
            "" if is_placeholder_context(session.active_topic) else session.active_topic
        )
        conversational_event = (
            "" if is_placeholder_context(session.recent_event) else session.recent_event
        )
        conversational_period = (
            "" if is_placeholder_context(session.recent_period) else session.recent_period
        )
        explicit_places = tuple(dict.fromkeys(EXPLICIT_PLACE.findall(query)))
        explicit_people = tuple(
            value
            for value in dict.fromkeys(EXPLICIT_PERSON.findall(query))
            if value not in _PERSON_STOPWORDS
        )
        if re.match(r"\s*(?:그럼|그러면|그렇다면)\s+", query):
            short_match = re.fullmatch(
                r"(?:그럼|그러면|그렇다면)\s+([가-힣]{2,4}?)(?:은|는|이|가)[?.!]?",
                query.strip(),
            ) or re.fullmatch(
                r"(?:그럼|그러면|그렇다면)\s+([가-힣]{2,4})[?.!]?",
                query.strip(),
            )
            short_people = [short_match.group(1)] if short_match else []
            explicit_people = tuple(
                value for value in dict.fromkeys((*short_people, *explicit_people))
                if value not in _PERSON_STOPWORDS
            )
        elif session.recent_people and re.fullmatch(
            r"[가-힣]{2,4}(?:은|는|이|가)?[?.!]?", query.strip()
        ):
            short_people = re.findall(r"^([가-힣]{2,4}?)(?:은|는|이|가)?[?.!]?$", query.strip())
            explicit_people = tuple(
                value for value in dict.fromkeys((*short_people, *explicit_people))
                if value not in _PERSON_STOPWORDS
            )
        explicit_entities = (*explicit_places, *explicit_people)
        is_transformation = bool(session.turns and TRANSFORMATION.fullmatch(query.strip()))
        is_expansion = bool(session.turns and DETAIL_EXPANSION.fullmatch(query.strip()))
        is_correction = bool(session.turns and CORRECTION.search(query))
        explicit_followup = bool(FOLLOWUP.search(query))
        elliptical_followup = bool(
            session.turns
            and not explicit_entities
            and (
                ELLIPTICAL_FOLLOWUP.fullmatch(query.strip())
                or re.fullmatch(
                    r"(?:왜|언제|누가|누구|어떻게|그럼|그러면|그렇다면)"
                    r"[^?.!]{0,24}[?.!]?",
                    query.strip(),
                )
            )
        )
        is_followup = bool(
            session.turns
            and (explicit_followup or elliptical_followup or is_transformation or is_expansion or is_correction)
        )
        refers_to_place = bool(PLACE_REFERENCE.search(query))
        generic_people_followup = bool(
            session.turns and GENERIC_PEOPLE_FOLLOWUP.fullmatch(query.strip())
        )

        if explicit_places:
            active_place = explicit_places[0]
        elif journey_place and refers_to_place:
            active_place = journey_place
        else:
            active_place = conversational_place or journey_place
        active_piece = journey_piece or conversational_piece
        terms: list[str] = []
        parallel_question = self._parallel_question(query, session, explicit_people)
        referenced_person = self._referenced_person(session) if PERSON_REFERENCE.search(query) else ""
        if is_transformation or is_expansion:
            # These requests operate on the immediately preceding answer. Its factual
            # claims remain non-evidence; the orchestrator reuses only retrieved chunks.
            terms.extend((active_place, conversational_event, conversational_topic))
        elif parallel_question:
            terms.extend((explicit_people[0], conversational_event, active_place))
        elif elliptical_followup:
            # Chain follow-ups keep stable structured context instead of recursively
            # appending the immediately previous wording. Assistant prose is never used.
            terms.extend((active_place, conversational_event, conversational_topic))
            if not any(terms):
                terms.append(session.turns[-1].user)
        elif generic_people_followup:
            terms.extend((active_place, conversational_event, conversational_topic))
            if not any(terms):
                terms.append(session.turns[-1].user)
        elif explicit_entities and is_followup:
            terms.extend((*explicit_entities, conversational_event, conversational_topic))
        elif (
            active_place
            and (is_followup or refers_to_place)
            and (not PERSON_REFERENCE.search(query) or refers_to_place)
        ):
            terms.append(active_place)
        if active_piece and active_piece not in terms and re.search(
            r"이\s*조각|현재\s*조각|방금\s*본", query
        ):
            terms.append(active_piece)
        if (
            is_followup and not explicit_entities
            and not generic_people_followup and not elliptical_followup
        ):
            if PERSON_REFERENCE.search(query):
                terms.append(referenced_person)
            if conversational_event:
                terms.append(conversational_event)
            if conversational_period:
                terms.append(conversational_period)
            if conversational_topic:
                terms.append(conversational_topic)
            if not terms:
                terms.append(session.turns[-1].user)
        ordered = tuple(dict.fromkeys(value for value in terms if value))
        resolved_question = parallel_question or query
        if is_expansion:
            grounded_user = (
                session.evidence_turns[-1].user
                if session.evidence_turns else session.turns[-1].user
            )
            resolved_question = f"{grounded_user} — {query}"
        if referenced_person:
            resolved_question = PERSON_REFERENCE.sub(referenced_person, resolved_question)
        if (is_correction or elliptical_followup) and resolved_question == query and ordered:
            resolved_question = " ".join((*ordered, query))
        returns_to_named_topic = bool(
            explicit_entities and re.search(r"(?:돌아가|돌아오|다시\s*이야기)", query)
        )
        prefix_terms = tuple(
            value
            for value in ordered
            if value not in resolved_question
            or (returns_to_named_topic and value in explicit_entities)
        )
        search_query = (
            " ".join((*prefix_terms, resolved_question))
            if prefix_terms else resolved_question
        )
        events = EVENT.findall(query)
        periods = PERIOD.findall(query)
        topic = (
            events[0]
            if events
            else explicit_entities[0]
            if explicit_entities
            else conversational_topic
        )
        recent_entities = tuple(
            dict.fromkeys(
                value
                for value in (*explicit_entities, *session.recent_entities)
                if value and not is_placeholder_context(value)
            )
        )[:8]
        recent_people = tuple(dict.fromkeys((*explicit_people, *session.recent_people)))[:4]
        if is_transformation:
            request_kind = ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
        elif is_expansion:
            request_kind = ConversationRequestKind.EXPAND_PREVIOUS_ANSWER
        elif is_correction:
            request_kind = ConversationRequestKind.CORRECTION
        elif is_followup:
            request_kind = ConversationRequestKind.FACTUAL_FOLLOWUP
        else:
            request_kind = ConversationRequestKind.INDEPENDENT
        return ResolvedContext(
            current_user_query=query,
            resolved_question=resolved_question,
            search_query=search_query,
            active_place=active_place,
            active_piece=active_piece,
            active_topic=topic,
            recent_entities=recent_entities,
            recent_people=recent_people,
            recent_event=events[0] if events else conversational_event,
            recent_period=periods[0] if periods else conversational_period,
            followup_resolved=is_followup,
            request_kind=request_kind,
            # Expansion is conservatively retrieval-eligible until the shared
            # evidence pipeline evaluates the remembered grounded chunks.
            needs_new_evidence=(
                request_kind
                != ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER
            ),
        )

    @staticmethod
    def _parallel_question(
        query: str, session: ChatSession, explicit_people: tuple[str, ...]
    ) -> str:
        """Carry the previous predicate into a short explicit-entity comparison."""

        if not explicit_people or not session.turns:
            return ""
        if not re.fullmatch(
            r"(?:(?:그럼|그러면|그렇다면)\s*)?[가-힣]{2,4}(?:은|는|이|가)?[?.!]?",
            query.strip(),
        ):
            return ""
        previous = session.turns[-1].user
        replaced = re.sub(
            r"(?<![가-힣])[가-힣]{2,4}(?=(?:은|는|이|가))",
            explicit_people[0],
            previous,
            count=1,
        )
        return replaced if replaced != previous else f"{explicit_people[0]}에 관해 {previous}"

    @staticmethod
    def _referenced_person(session: ChatSession) -> str:
        """Prefer the most recently focused entity; assistant prose is context only."""

        if len(session.recent_people) == 1:
            return session.recent_people[0]
        if len(session.recent_people) > 1:
            return ""
        if not session.turns:
            return ""
        candidates = re.findall(
            r"(?<![가-힣])([가-힣]{2,4})(?=(?:은|는|이|가|과|와)\s*)",
            session.turns[-1].assistant,
        )
        return next((item for item in reversed(candidates) if item not in _PERSON_STOPWORDS), "")
