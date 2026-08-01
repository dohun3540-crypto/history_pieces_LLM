"""Build a minimal, deterministic prompt for untrusted remote inference workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
) -> RemotePrompt:
    """Serialize only anonymous evidence labels and bounded text for a remote worker."""

    resolved = policy or RemotePromptPolicy()
    resolved.validate()
    clean_system = sanitize_remote_text(system_prompt, enabled=resolved.sanitize_enabled)
    clean_query = sanitize_remote_text(user_query, enabled=resolved.sanitize_enabled)
    if not clean_system or not clean_query:
        raise ValueError("remote-safe prompt의 시스템 지침 또는 질문이 비어 있습니다.")

    question_section = f"[사용자 질문]\n{clean_query}"
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
    for item in chunks:
        if len(evidence_blocks) >= resolved.max_evidence_items:
            break
        title = _truncate(
            sanitize_remote_text(item.chunk.title, enabled=resolved.sanitize_enabled), 160
        )
        publisher = _truncate(
            sanitize_remote_text(item.chunk.publisher, enabled=resolved.sanitize_enabled), 160
        )
        evidence = _truncate(
            sanitize_remote_text(item.chunk.text, enabled=resolved.sanitize_enabled),
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
    user_prompt = "[검색 근거]\n" + "\n\n".join(evidence_blocks) + "\n\n" + question_section

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


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _total_chars(system_prompt: str, user_prompt: str, messages: Sequence[LLMMessage]) -> int:
    return len(system_prompt) + len(user_prompt) + sum(len(item.content) for item in messages)
