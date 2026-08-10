"""Resolve conversational references without treating chat history as evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    r"여기|이곳|거기|아까|방금|이후에는?|왜\s*그랬|"
    r"누가\s*(?:참여|관여|주도)|그래서|그건|"
    r"관련(?:된)?\s*(?:인물|사람)|"
    r"그\s*(?:이유|결과|영향|의미|배경)|"
    r"그럼|그러면|그렇다면|이어서|계속해서"
)
ELLIPTICAL_FOLLOWUP = re.compile(
    r"(?:(?:그럼|그러면|그렇다면)\s*)?(?:좀\s*)?(?:더\s*)?"
    r"(?:왜|언제|어디서|누가|누구|어떻게|무슨\s*이유(?:로)?|"
    r"이유|결과|영향|의미|배경|그다음|다음)"
    r"(?:은|는|이|가|부터|까지|였어|였어요|인가요|야|예요|지|요|"
    r"\s*(?:알려\s*줘|설명해\s*줘|말해\s*줘))?[?.!]*|"
    r"(?:좀\s*)?(?:더\s*)?(?:쉽게|자세히|짧게|간단히|다시)\s*"
    r"(?:설명해|알려|말해|요약해)(?:\s*줘|\s*주세요|요)?[?.!]*",
    re.IGNORECASE,
)
PLACE_REFERENCE = re.compile(r"여기|이곳|거기|그곳|그\s*장소|이\s*장소")
PERSON_REFERENCE = re.compile(r"그\s*사람|그\s*인물")
EXPLICIT_PLACE = re.compile(
    r"구\s*일본영사관|목포(?:역|항|부|진|해관|세관)|삼학도|유달산|"
    r"무안감리서|동양척식주식회사(?:\s*목포지점)?|호남은행|일본영사관|목포"
)
EXPLICIT_PERSON = re.compile(
    r"(?<![가-힣])([가-힣]{2,4})(?=(?:은|는|이|가)\s*(?:어떤\s*사람|누구))"
)
GENERIC_PEOPLE_FOLLOWUP = re.compile(
    r"관련(?:된)?\s*(?:인물|사람)(?:은|이)?\s*누구(?:인가요|예요|야)?[?.!]?$"
)
_PERSON_STOPWORDS = {"사람", "인물", "누구", "당시", "당시에", "그때"}
EVENT = re.compile(r"(?:학생운동|독립운동|노동운동|개항|시위|파업|사건)")
PERIOD = re.compile(r"(?:18|19|20)\d{2}년|일제강점기|대한제국|근대")


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
    search_query: str
    active_place: str
    active_piece: str
    active_topic: str
    recent_entities: tuple[str, ...]
    recent_event: str
    recent_period: str
    followup_resolved: bool


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
        explicit_entities = (*explicit_places, *explicit_people)
        explicit_followup = bool(FOLLOWUP.search(query))
        elliptical_followup = bool(
            session.turns
            and not explicit_entities
            and ELLIPTICAL_FOLLOWUP.fullmatch(query.strip())
        )
        is_followup = bool(session.turns and (explicit_followup or elliptical_followup))
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
        if elliptical_followup:
            # 짧은 재질문에는 주제가 없으므로 직전 사용자 질문만 결합한다.
            # 어시스턴트 답변은 검색 근거나 검색어로 재사용하지 않는다.
            terms.append(session.turns[-1].user)
        elif generic_people_followup:
            terms.append(session.turns[-1].user)
        elif explicit_entities:
            terms.extend(explicit_entities)
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
        if is_followup and not generic_people_followup and not elliptical_followup:
            if PERSON_REFERENCE.search(query):
                terms.extend(session.recent_entities[:1])
            if conversational_event:
                terms.append(conversational_event)
            if conversational_period:
                terms.append(conversational_period)
            if conversational_topic:
                terms.append(conversational_topic)
            if not terms:
                terms.append(session.turns[-1].user)
        ordered = tuple(dict.fromkeys(value for value in terms if value))
        search_query = " ".join((*ordered, query)) if ordered else query
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
        return ResolvedContext(
            search_query=search_query,
            active_place=active_place,
            active_piece=active_piece,
            active_topic=topic,
            recent_entities=recent_entities,
            recent_event=events[0] if events else conversational_event,
            recent_period=periods[0] if periods else conversational_period,
            followup_resolved=is_followup,
        )
