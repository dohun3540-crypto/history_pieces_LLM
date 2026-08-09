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
FOLLOWUP = re.compile(
    r"그\s*(?:때|당시|사람|사건|학교|건물|역|회사|뒤|과정|자료)|"
    r"여기|이곳|아까|왜\s*그랬|누가\s*(?:참여|관여|주도)|그래서|그건|"
    r"관련(?:된)?\s*(?:인물|사람)"
)
EXPLICIT_TARGET = re.compile(
    r"목포(?:역|항|부|진|해관|세관)?|삼학도|유달산|무안감리서|"
    r"동양척식주식회사|호남은행|일본영사관"
)
EVENT = re.compile(r"(?:학생운동|독립운동|노동운동|개항|시위|파업|사건)")
PERIOD = re.compile(r"(?:18|19|20)\d{2}년|일제강점기|대한제국|근대")


def context_label(value: str | None) -> str:
    if not value:
        return ""
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
        active_place = context_label(current_place_id) or session.active_place
        active_piece = context_label(current_piece_id) or session.active_piece
        explicit = EXPLICIT_TARGET.findall(query)
        is_followup = bool(session.turns and FOLLOWUP.search(query))
        terms: list[str] = []
        if explicit:
            terms.extend(explicit)
        elif active_place and (
            is_followup
            or re.search(r"여기|이곳|이\s*장소|왜\s*중요|무슨\s*일", query)
        ):
            terms.append(active_place)
        if active_piece and active_piece not in terms and (
            is_followup or re.search(r"이\s*조각|현재\s*조각|방금\s*본", query)
        ):
            terms.append(active_piece)
        if is_followup:
            if re.search(r"그\s*(?:사람|사건|학교|건물|역|회사)|그건|여기|이곳", query):
                terms.extend(session.recent_entities[:1])
            if session.recent_event:
                terms.append(session.recent_event)
            if session.recent_period:
                terms.append(session.recent_period)
            if session.active_topic:
                terms.append(session.active_topic)
            if not terms:
                terms.append(session.turns[-1].user)
        ordered = tuple(dict.fromkeys(value for value in terms if value))
        if explicit and ordered:
            search_query = " ".join(ordered)
        elif is_followup and ordered:
            search_query = " ".join(ordered)
            if len(ordered) == 1 and session.turns and ordered[0] == session.turns[-1].user:
                search_query += " " + query
        else:
            search_query = " ".join((*ordered, query)) if ordered else query
        events = EVENT.findall(query)
        periods = PERIOD.findall(query)
        topic = explicit[0] if explicit else (events[0] if events else session.active_topic)
        return ResolvedContext(
            search_query=search_query,
            active_place=active_place,
            active_piece=active_piece,
            active_topic=topic,
            recent_entities=tuple(dict.fromkeys((*explicit, *session.recent_entities)))[:8],
            recent_event=events[0] if events else session.recent_event,
            recent_period=periods[0] if periods else session.recent_period,
            followup_resolved=is_followup,
        )
