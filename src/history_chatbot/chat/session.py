"""실행 모드별로 격리된 제한 길이 대화 세션."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from history_chatbot.runtime import RuntimeMode


@dataclass(frozen=True, slots=True)
class SessionTurn:
    user: str
    assistant: str


@dataclass(frozen=True, slots=True)
class EvidenceTurn:
    """A retrieval trace; assistant prose is deliberately excluded."""

    user: str
    active_place: str
    active_topic: str
    chunk_ids: tuple[str, ...]
    active_subject: str = ""
    active_person: str = ""
    answered_intent: str = ""


@dataclass(slots=True)
class ChatSession:
    session_id: str
    locale: str = "ko"
    turns: list[SessionTurn] = field(default_factory=list)
    summary: str = ""
    active_place: str = ""
    active_piece: str = ""
    active_topic: str = ""
    active_subject: str = ""
    active_person: str = ""
    stable_evidence_anchor: str = ""
    last_answered_intent: str = ""
    recent_entities: tuple[str, ...] = ()
    recent_people: tuple[str, ...] = ()
    recent_event: str = ""
    recent_period: str = ""
    evidence_turns: list[EvidenceTurn] = field(default_factory=list)


class SessionStore:
    def __init__(
        self,
        mode: RuntimeMode,
        *,
        path: Path | None = None,
        max_turns: int = 8,
        max_summary_chars: int = 800,
    ) -> None:
        self.mode = mode
        self.path = path
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars
        self._sessions: dict[str, ChatSession] = {}
        self._load()

    def create(self, locale: str = "ko") -> ChatSession:
        session = ChatSession(uuid.uuid4().hex, locale)
        self._sessions[session.session_id] = session
        self._save()
        return session

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None, locale: str = "ko") -> ChatSession:
        if session_id:
            found = self.get(session_id)
            if found is not None:
                return found
        return self.create(locale)

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        session = self._sessions[session_id]
        session.turns.append(SessionTurn(user, assistant))
        if len(session.turns) > self.max_turns:
            removed = session.turns.pop(0)
            addition = (
                f"[USER]\n{removed.user}\n"
                f"[ASSISTANT | 대화 문맥, 사실 근거 아님]\n{removed.assistant}"
            )
            entries = [
                item for item in (*session.summary.split("\n---\n"), addition) if item
            ]
            while entries and len("\n---\n".join(entries)) > self.max_summary_chars:
                entries.pop(0)
            session.summary = "\n---\n".join(entries)
        self._save()

    def add_evidence_turn(
        self,
        session_id: str,
        *,
        user: str,
        active_place: str,
        active_topic: str,
        chunk_ids: tuple[str, ...],
        active_subject: str = "",
        active_person: str = "",
        answered_intent: str = "",
    ) -> None:
        """Persist only identifiers of chunks actually returned by retrieval."""

        if not chunk_ids:
            return
        session = self._sessions[session_id]
        session.evidence_turns.append(
            EvidenceTurn(
                user, active_place, active_topic, chunk_ids,
                active_subject, active_person, answered_intent,
            )
        )
        session.evidence_turns = session.evidence_turns[-self.max_turns :]
        self._save()

    def update_context(self, session_id: str, **values: object) -> None:
        session = self._sessions[session_id]
        for name in (
            "active_place", "active_piece", "active_topic", "active_subject",
            "active_person", "stable_evidence_anchor", "last_answered_intent",
            "recent_entities", "recent_people", "recent_event", "recent_period",
        ):
            if name in values:
                setattr(session, name, values[name])
        self._save()

    def reset(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None) is not None
        if removed:
            self._save()
        return removed

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("runtime_mode") != self.mode.value:
            raise ValueError("다른 실행 모드의 세션 데이터는 불러올 수 없습니다.")
        for item in payload.get("sessions", []):
            session = ChatSession(
                session_id=item["session_id"],
                locale=item.get("locale", "ko"),
                turns=[SessionTurn(**turn) for turn in item.get("turns", [])],
                summary=item.get("summary", ""),
                active_place=item.get("active_place", ""),
                active_piece=item.get("active_piece", ""),
                active_topic=item.get("active_topic", ""),
                active_subject=item.get("active_subject", ""),
                active_person=item.get("active_person", ""),
                stable_evidence_anchor=item.get("stable_evidence_anchor", ""),
                last_answered_intent=item.get("last_answered_intent", ""),
                recent_entities=tuple(item.get("recent_entities", ())),
                recent_people=tuple(item.get("recent_people", ())),
                recent_event=item.get("recent_event", ""),
                recent_period=item.get("recent_period", ""),
                evidence_turns=[
                    EvidenceTurn(
                        user=value.get("user", ""),
                        active_place=value.get("active_place", ""),
                        active_topic=value.get("active_topic", ""),
                        chunk_ids=tuple(value.get("chunk_ids", ())),
                        active_subject=value.get("active_subject", ""),
                        active_person=value.get("active_person", ""),
                        answered_intent=value.get("answered_intent", ""),
                    )
                    for value in item.get("evidence_turns", [])
                ],
            )
            self._sessions[session.session_id] = session

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runtime_mode": self.mode.value,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "locale": session.locale,
                    "turns": [asdict(turn) for turn in session.turns],
                    "summary": session.summary,
                    "active_place": session.active_place,
                    "active_piece": session.active_piece,
                    "active_topic": session.active_topic,
                    "active_subject": session.active_subject,
                    "active_person": session.active_person,
                    "stable_evidence_anchor": session.stable_evidence_anchor,
                    "last_answered_intent": session.last_answered_intent,
                    "recent_entities": list(session.recent_entities),
                    "recent_people": list(session.recent_people),
                    "recent_event": session.recent_event,
                    "recent_period": session.recent_period,
                    "evidence_turns": [asdict(value) for value in session.evidence_turns],
                }
                for session in self._sessions.values()
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
