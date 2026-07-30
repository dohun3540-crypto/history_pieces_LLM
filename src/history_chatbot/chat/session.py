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


@dataclass(slots=True)
class ChatSession:
    session_id: str
    locale: str = "ko"
    turns: list[SessionTurn] = field(default_factory=list)
    summary: str = ""


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
            addition = f"사용자: {removed.user}\n응답: {removed.assistant}\n"
            session.summary = (session.summary + addition)[-self.max_summary_chars :]
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
                }
                for session in self._sessions.values()
            ],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
