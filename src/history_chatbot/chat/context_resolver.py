"""Resolve conversational references without treating chat history as evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from history_chatbot.chat.session import ChatSession
from history_chatbot.retrieval.query_normalizer import explicit_subject_words


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
    r"그\s*(?:때|당시|사람|사건|학교|건물|역|노선|회사|곳|장소|뒤|과정|자료)|"
    r"여기|이곳|거기|아까|방금|이후에는?|왜\s*(?:그랬|왔|온)|"
    r"누가\s*(?:참여|관여|주도)|그래서|그건|그게|그\s*(?:일|내용|말)|그\s*사람\s*말고|"
    r"관련(?:된)?\s*(?:인물|사람)|"
    r"관련\s*(?:시기|날짜|장소)|첫(?:\s*번째)?\s*단체|두\s*번째\s*단체|둘(?:을|은|이)|"
    r"그\s*(?:이유|결과|영향|의미|배경)|"
    r"그럼|그러면|그렇다면|이어서|계속해서|그\s*다음|지금(?:은)?|또\s*있어|다른\s*건|"
    r"원래(?:는)?\s*(?:뭐|무엇|어떤)|확실하지\s*않은\s*부분|불확실한\s*부분|돌아가|돌아오"
)
ELLIPTICAL_FOLLOWUP = re.compile(
    r"(?:(?:그럼|그러면|그렇다면)\s*)?(?:좀\s*)?(?:더\s*)?"
    r"(?:왜|언제|어디서|누가|누구|어떻게|무슨\s*이유(?:로)?|"
    r"이유|결과|영향|의미|배경|그다음|다음|뭐\s*하는\s*(?:곳|기관|회사)(?:이었어|이야|인가요)?|"
    r"(?:건립|설립|개통|준공)\s*시기)"
    r"(?:은|는|이|가|부터|까지|였어|였어요|인가요|야|예요|지|요|"
    r"\s*(?:알려\s*줘|설명해\s*줘|말해\s*줘|지었어|만들었어|왔던\s*거야))?[?.!]*|"
    r"(?:아주\s*)?(?:좀\s*)?(?:더\s*)?(?:쉽게|자세히|짧게|간단히|간단하게|다시)\s*"
    r"(?:설명해|알려|말해|요약해|정리해|풀어)(?:\s*줘|\s*주세요|요)?[?.!]*|"
    r"(?:원래(?:는)?\s*(?:뭐|무엇)(?:였어|인가요|야)?|"
    r"또\s*있어|다른\s*건|관련\s*(?:인물|사람|시기|날짜|장소)(?:은|는)?)[?.!]*",
    re.IGNORECASE,
)
PLACE_REFERENCE = re.compile(r"여기|이곳|거기|그곳|그\s*장소|이\s*장소")
PERSON_REFERENCE = re.compile(r"(?:그|이|저)\s*(?:사람|인물)|그(?:는|가|를)(?=\s|[?.!,]|$)")
EXPLICIT_PLACE = re.compile(
    r"구\s*목포\s*일본영사관|구\s*일본영사관|목포근대역사관\s*[12]관|목포(?:역|항|부|진|해관|세관)|삼학도|유달산|"
    r"고하도|무안감리서|동양척식주식회사(?:\s*목포지점)?|호남은행|일본영사관|목포"
)
EXPLICIT_PERSON = re.compile(
    r"(?<![가-힣])([가-힣]{2,4})(?=(?:은|는|이|가)"
    r"[^?.!]{0,40}(?:어떤\s*사람|누구|인물|참석|도착|왔|"
    r"무슨\s*일|무엇을\s*했|뭘\s*했|온\s*(?:이유|사건)|갔|간\s*이유))"
)
GENERIC_PEOPLE_FOLLOWUP = re.compile(
    r"관련(?:된)?\s*(?:인물|사람)(?:은|이)?\s*누구(?:인가요|예요|야)?[?.!]?$"
)
_PERSON_STOPWORDS = {
    "사람", "인물", "누구", "당시", "당시에", "그때", "여기", "결과",
    "언제였어", "왜", "어떻게", "전시관", "건물", "장소", "여기서", "거기서", "이곳", "그곳",
}
EVENT = re.compile(r"(?:학생운동|독립운동|노동운동|개항|시위|파업|사건)")
PERIOD = re.compile(r"(?:18|19|20)\d{2}년|일제강점기|대한제국|근대")
TRANSFORMATION = re.compile(
    r"(?:쉽게|더\s*쉽게|짧게|간단히|간단하게|자세히|다시|한\s*줄로?|핵심만(?:\s*말해\s*줘)?|무슨\s*(?:말|뜻)이야)[?.!]*$|"
    r"그(?:거|게|건)(?:\s*(?:무슨|어떤)\s*(?:말|뜻)(?:이야|이야기야|인가요)?)?[?.!]*$|"
    r"그\s*말이\s*(?:뭐야|무슨\s*뜻이야)[?.!]*$|"
    r"(?:간단히\s*말하면|좀\s*풀어서\s*말해\s*줘|한\s*문장으로\s*말해\s*줘)[?.!]*$|"
    r"(?:아주\s*)?(?:좀\s*)?(?:더\s*)?(?:쉽게|짧게|간단히|간단하게|한\s*문장으로|다시)\s*"
    r"(?:설명해|알려|말해|요약해|정리해|풀어)(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"아까\s*답(?:을|변을)?\s*다시\s*(?:설명해|말해)(?:\s*줘|\s*주세요)?[?.!]*$|"
    r"(?:정리해서|요약해서)\s*(?:알려|말해)(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"(?:초등학생|아이)(?:도|가)?\s*이해할\s*수\s*있게\s*"
    r"(?:설명해|알려|말해)(?:\s*줘|\s*주세요|요)?[?.!]*$|"
    r"(?:초등학생|아이)(?:도|가)?\s*알게\s*(?:설명해|알려|말해)?(?:\s*줘|\s*주세요|요)?[?.!]*$",
    re.IGNORECASE,
)
DETAIL_EXPANSION = re.compile(
    r"(?:더|자세히)[?.!]*$|"
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
    active_subject: str
    active_person: str
    stable_evidence_anchor: str
    current_intent: str
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
        is_comparison = bool(re.search(r"차이|달라|비교", query))
        if is_comparison and session.active_subject:
            active_numbered = re.fullmatch(r"(.+?)\s*([12])관", session.active_subject)
            sibling = re.search(r"([12])관", query)
            if active_numbered and sibling:
                expanded = f"{active_numbered.group(1).strip()} {sibling.group(1)}관"
                explicit_places = tuple(dict.fromkeys(
                    (session.active_subject, expanded, *explicit_places)
                ))
        explicit_people = tuple(
            value
            for value in dict.fromkeys(EXPLICIT_PERSON.findall(query))
            if value not in _PERSON_STOPWORDS
        )
        explicit_people = tuple(
            person for person in explicit_people
            if not any(
                person in place or place in person
                for place in explicit_places
            )
        )
        if re.search(r"관련(?:된)?\s*(?:인물|사람)|인물이나\s*장소", query):
            # The noun before "관련 인물" is the historical subject, not a
            # person merely because a later token asks for people.
            explicit_people = ()
        if PERSON_REFERENCE.search(query):
            explicit_people = ()
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
        generic_subjects = explicit_subject_words(query)
        generic_subjects = tuple(
            value for value in generic_subjects
            if value not in {"하는", "핵심만", "말해줘", "곳이었어"}
        )
        if re.match(r"\s*원래(?:는)?\s*(?:뭐|무엇|어떤)", query):
            generic_subjects = ()
        if TRANSFORMATION.fullmatch(query.strip()) and re.search(r"그(?:거|게|건|\s*말)", query):
            generic_subjects = ()
        if (
            not explicit_people
            and not PLACE_REFERENCE.search(query)
            and not TRANSFORMATION.fullmatch(query.strip())
            and re.search(r"누구|사람|인물|그는|무슨\s*일|무엇을\s*했|뭘\s*했", query)
            and len(generic_subjects) == 1
            and re.fullmatch(r"[가-힣]{2,4}", generic_subjects[0])
        ):
            explicit_people = generic_subjects
        referenced_subjects = self._referenced_subjects(query, session)
        explicit_entities = tuple(dict.fromkeys(
            (*explicit_places, *explicit_people, *referenced_subjects, *generic_subjects)
        ))
        explicit_events = tuple(dict.fromkeys(EVENT.findall(query)))
        return_target = self._return_target(query, session)
        returns_to_named_topic = bool(
            return_target
        )
        returns_to_first_topic = bool(
            session.evidence_turns
            and re.search(r"다시\s*(?:첫|처음)\s*(?:사건|주제)", query)
        )
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
        anchored_place_relation = bool(
            session.stable_evidence_anchor
            and explicit_places
            and (
                re.search(r"그(?:게|것|일)|거기|그곳", query)
                or re.search(r"왜\s+.+(?:에|에서)\s*(?:있|갔|왔)", query)
            )
        )
        refers_to_place = bool(PLACE_REFERENCE.search(query))
        generic_people_followup = bool(
            session.turns and GENERIC_PEOPLE_FOLLOWUP.fullmatch(query.strip())
        )
        validated_topic = self._validated_topic(session, conversational_topic)
        validated_place = self._validated_place(session, conversational_place)

        if explicit_places:
            active_place = explicit_places[0]
        elif journey_place and refers_to_place:
            active_place = journey_place
        else:
            active_place = conversational_place or journey_place
        if explicit_entities and not is_followup and not explicit_places and not refers_to_place:
            active_place = ""
        active_piece = journey_piece or conversational_piece
        terms: list[str] = []
        parallel_question = self._parallel_question(query, session, explicit_people)
        referenced_person = self._referenced_person(session) if PERSON_REFERENCE.search(query) else ""
        if returns_to_first_topic:
            terms.append(session.evidence_turns[0].user)
        elif returns_to_named_topic:
            # An explicit return target replaces, rather than augments, the immediately
            # preceding topic.  Conversation text only resolves the target; retrieved
            # chunks remain the sole historical evidence.
            terms.append(return_target)
        elif is_transformation or is_expansion:
            # These requests operate on the immediately preceding answer. Its factual
            # claims remain non-evidence; the orchestrator reuses only retrieved chunks.
            terms.append(
                session.stable_evidence_anchor
                or session.active_subject
                or conversational_topic
                or active_place
            )
        elif parallel_question:
            terms.extend((explicit_people[0], conversational_event, active_place))
        elif elliptical_followup:
            # Chain follow-ups keep stable structured context instead of recursively
            # appending the immediately previous wording. Assistant prose is never used.
            evidence_anchor = session.stable_evidence_anchor or self._validated_evidence_user(
                session, validated_place, validated_topic
            )
            if evidence_anchor:
                terms.append(evidence_anchor)
            else:
                terms.extend((validated_place, conversational_event, validated_topic))
                if validated_place or conversational_event or validated_topic:
                    terms.append(conversational_period)
            if not any(terms) and self._last_turn_context_is_trusted(session):
                terms.append(session.turns[-1].user)
        elif generic_people_followup:
            evidence_anchor = session.stable_evidence_anchor or self._validated_evidence_user(
                session, validated_place, validated_topic
            )
            if evidence_anchor:
                terms.append(evidence_anchor)
            else:
                terms.extend((validated_place, conversational_event, validated_topic))
                if validated_place or conversational_event or validated_topic:
                    terms.append(conversational_period)
            if not any(terms) and self._last_turn_context_is_trusted(session):
                terms.append(session.turns[-1].user)
        elif anchored_place_relation:
            terms.extend((session.stable_evidence_anchor, *explicit_places))
        elif explicit_entities and is_followup:
            terms.extend((active_place if refers_to_place else "", *explicit_entities))
            if re.search(r"관련|와도|과도", query) and conversational_event:
                terms.append(conversational_event)
        elif (
            active_place
            and (is_followup or refers_to_place)
            and not session.active_person
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
            and not returns_to_named_topic and not returns_to_first_topic
        ):
            if PERSON_REFERENCE.search(query):
                terms.append(referenced_person)
            if validated_topic and not referenced_person:
                terms.append(validated_topic)
            if conversational_event and not referenced_person and self._last_turn_context_is_trusted(session):
                terms.append(conversational_event)
            if conversational_period:
                terms.append(conversational_period)
            if not terms and self._last_turn_context_is_trusted(session):
                terms.append(session.turns[-1].user)
        ordered = tuple(dict.fromkeys(value for value in terms if value))
        resolved_question = parallel_question or query
        if is_comparison and len(explicit_places) >= 2:
            resolved_question = " ".join((*explicit_places[:2], query))
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
        prefix_terms = tuple(
            value
            for value in ordered
            if value not in resolved_question
            or (returns_to_named_topic and value == return_target)
        )
        search_query = (
            " ".join((*prefix_terms, resolved_question))
            if prefix_terms else resolved_question
        )
        if returns_to_first_topic:
            detail = (
                "장소" if re.search(r"장소|어디", query)
                else "인물" if re.search(r"인물|사람|누구", query)
                else "시기" if re.search(r"언제|시기|연도|날짜", query)
                else ""
            )
            search_query = " ".join(
                value for value in (session.evidence_turns[0].user, detail) if value
            )
        events = list(explicit_events)
        periods = PERIOD.findall(query)
        active_person = (
            explicit_people[0] if explicit_people
            else referenced_person if referenced_person
            else session.active_person
        )
        specific_generic = generic_subjects[0] if generic_subjects else ""
        specific_place = explicit_places[0] if explicit_places else ""
        if (
            specific_generic
            and specific_place
            and specific_place in specific_generic
            and len(specific_generic) > len(specific_place)
        ):
            specific_place = ""
        specific_event = events[0] if events else ""
        if (
            specific_generic
            and specific_event
            and specific_event in specific_generic
            and len(specific_generic) > len(specific_event)
        ):
            specific_event = ""
        active_subject = (
            session.stable_evidence_anchor or session.active_subject or conversational_topic
            if is_transformation
            else session.stable_evidence_anchor
            if anchored_place_relation
            else active_person if active_person and (explicit_people or PERSON_REFERENCE.search(query))
            else specific_place
            if specific_place
            else specific_event
            if specific_event
            else specific_generic
            if specific_generic
            else session.stable_evidence_anchor or session.active_subject or conversational_topic
        )
        topic = (
            session.evidence_turns[0].active_topic
            if returns_to_first_topic
            else return_target
            if returns_to_named_topic
            else active_subject
            if active_subject
            else events[0]
            if events
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
            active_subject=active_subject,
            active_person=active_person,
            stable_evidence_anchor=session.stable_evidence_anchor,
            current_intent=self._intent(query, request_kind),
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
    def _validated_topic(session: ChatSession, topic: str) -> str:
        if not topic:
            return ""
        if any(
            turn.active_topic == topic and turn.chunk_ids
            for turn in session.evidence_turns
        ):
            return topic
        return topic if ConversationContextResolver._last_turn_context_is_trusted(session) else ""

    @staticmethod
    def _validated_place(session: ChatSession, place: str) -> str:
        if not place:
            return ""
        if any(
            turn.active_place == place and turn.chunk_ids
            for turn in session.evidence_turns
        ):
            return place
        return place if ConversationContextResolver._last_turn_context_is_trusted(session) else ""

    @staticmethod
    def _last_turn_context_is_trusted(session: ChatSession) -> bool:
        if not session.turns:
            return False
        answer = session.turns[-1].assistant
        return not bool(re.search(
            r"확인하지\s*못|확인할\s*수\s*없|근거를\s*확인하지\s*못|"
            r"추측하지\s*않|insufficient[_ ]evidence|llm[_ ]error",
            answer,
            re.IGNORECASE,
        ))

    @staticmethod
    def _return_target(query: str, session: ChatSession) -> str:
        if not re.search(r"다시|아까|돌아가|돌아오|복귀", query):
            return ""
        compact_query = re.sub(r"\s+", "", query)
        candidates: list[str] = []
        for turn in reversed(session.evidence_turns):
            candidates.extend((turn.active_topic, turn.active_place))
            candidates.extend(explicit_subject_words(turn.user))
        for candidate in dict.fromkeys(value for value in candidates if value):
            if re.sub(r"\s+", "", candidate) in compact_query:
                return candidate
        named = re.search(
            r"(?:다시|아까)\s+([0-9A-Za-z가-힣·]{2,30})(?:으?로|\s*이야기)",
            query,
        )
        if named:
            return named.group(1)
        return ""

    @staticmethod
    def _validated_evidence_user(
        session: ChatSession, place: str, topic: str
    ) -> str:
        fallback = ""
        for turn in reversed(session.evidence_turns):
            if (place and turn.active_place == place) or (topic and turn.active_topic == topic):
                fallback = fallback or turn.user
                if explicit_subject_words(turn.user):
                    return turn.user
        return fallback

    @staticmethod
    def _referenced_subjects(query: str, session: ChatSession) -> tuple[str, ...]:
        if not re.search(
            r"첫(?:\s*번째)?\s*단체|두\s*번째\s*단체|둘(?:을|은|이|\s*중)|두\s*사건|"
            r"확실하지\s*않은\s*부분|불확실한\s*부분",
            query,
        ):
            return ()
        for turn in reversed(session.evidence_turns):
            subjects = explicit_subject_words(turn.user)
            if len(subjects) < 2:
                continue
            if re.search(r"첫(?:\s*번째)?\s*단체", query):
                return (subjects[0],)
            if re.search(r"두\s*번째\s*단체", query):
                return (subjects[1],)
            return subjects[:2]
        return ()

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
        """Use only evidence-derived referents, never generated assistant prose."""

        if session.active_person:
            return session.active_person
        if session.active_topic and session.active_topic in session.recent_people:
            return session.active_topic
        if len(session.recent_people) == 1:
            return session.recent_people[0]
        if len(session.recent_people) > 1:
            return ""
        return ""

    @staticmethod
    def _intent(query: str, request_kind: ConversationRequestKind) -> str:
        if request_kind == ConversationRequestKind.TRANSFORM_PREVIOUS_ANSWER:
            return "transform"
        patterns = (
            ("current", r"지금|현재|오늘날"),
            ("time", r"언제|시기|연도|날짜|건립|설립|개통|준공"),
            ("cause", r"왜|이유|원인|배경|계기"),
            ("role", r"무슨\s*일|무엇을\s*했|뭘\s*했|어떤\s*활동|활동을\s*했|역할|어디에\s*쓰"),
            ("people", r"누구|누가|인물|사람"),
            ("place", r"어디|장소|지역"),
            ("result", r"그\s*뒤|이후|다음|결과|영향|변했|달라졌"),
        )
        return next((name for name, pattern in patterns if re.search(pattern, query)), "overview")
