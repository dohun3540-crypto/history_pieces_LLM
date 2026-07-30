"""Mock·로컬·원격 생성기가 공유하는 안정적인 LLM 계약."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator, Protocol


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("message role이 올바르지 않습니다.")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    messages: tuple[LLMMessage, ...] = ()
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False
    request_id: str = ""
    timeout: float = 60.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_prompt.strip() or not self.user_prompt.strip():
            raise ValueError("system_prompt와 user_prompt는 필수입니다.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature는 0~2여야 합니다.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p는 0 초과 1 이하여야 합니다.")
        if not 1 <= self.max_new_tokens <= 8192:
            raise ValueError("max_new_tokens는 1~8192여야 합니다.")
        if not 0.1 <= self.timeout <= 600:
            raise ValueError("timeout은 0.1~600초여야 합니다.")
        if len(self.stop_sequences) > 16 or any(len(item) > 128 for item in self.stop_sequences):
            raise ValueError("stop sequence의 개수 또는 길이가 제한을 초과했습니다.")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class LLMResponse:
    generated_text: str
    finish_reason: str
    usage: TokenUsage
    model: str
    model_revision: str
    request_id: str
    latency_ms: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(asdict(self.usage))
        value.pop("usage")
        return value


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    event: str
    data: dict[str, object]

    def __post_init__(self) -> None:
        if self.event not in {"start", "token", "delta", "completed", "error"}:
            raise ValueError("지원하지 않는 스트리밍 이벤트입니다.")


class ChatCompletionBackend(Protocol):
    backend_name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...
    def stream_complete(self, request: LLMRequest) -> Iterator[LLMStreamEvent]: ...
    def readiness(self) -> dict[str, object]: ...
