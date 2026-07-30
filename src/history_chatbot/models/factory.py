"""실제 모델 설치 전에도 안정적인 백엔드 경계를 제공하는 factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from history_chatbot.models.base import BaseLLM, GenerationRequest
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.models.remote import (
    RemoteLLMBackend,
    RemoteLLMConfig,
    UnconfiguredRemoteLLMBackend,
)
from history_chatbot.runtime import RuntimeMode


class StreamingLLMBackend(BaseLLM, ABC):
    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """취소 가능한 스트리밍 조각을 반환한다."""


class LocalLLMBackend(StreamingLLMBackend):
    def generate(self, request: GenerationRequest) -> str:
        raise RuntimeError("로컬 Llama 백엔드가 아직 설치되지 않았습니다.")

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        raise RuntimeError("로컬 Llama 백엔드가 아직 설치되지 않았습니다.")
        yield  # pragma: no cover


def build_llm_backend(
    backend: str,
    *,
    runtime_mode: RuntimeMode,
    fallback_message: str,
    remote_config: RemoteLLMConfig | None = None,
) -> object:
    if backend == "mock":
        if runtime_mode == RuntimeMode.PRODUCTION:
            raise ValueError("production 모드에서는 MockLLM을 사용할 수 없습니다.")
        return MockLLM(fallback_message)
    if backend == "local":
        return LocalLLMBackend()
    if backend == "remote":
        config = remote_config or RemoteLLMConfig.from_environment(runtime_mode)
        return RemoteLLMBackend(config, mode=runtime_mode)
    raise ValueError(f"지원하지 않는 LLM backend입니다: {backend}")


def build_llm_from_environment(
    runtime_mode: RuntimeMode,
    *,
    environ: dict[str, str] | None = None,
    fallback_message: str = "확인 가능한 자료가 부족합니다.",
) -> object:
    import os

    values = os.environ if environ is None else environ
    backend = values.get("LLM_BACKEND", "mock" if runtime_mode.allows_fixtures else "")
    if not backend:
        raise ValueError("production에는 LLM_BACKEND 설정이 필요합니다.")
    if backend == "remote" and (
        not values.get("LLM_BASE_URL", "").strip()
        or not values.get("LLM_MODEL", "").strip()
    ):
        return UnconfiguredRemoteLLMBackend()
    remote_config = RemoteLLMConfig.from_environment(runtime_mode, values) if backend == "remote" else None
    return build_llm_backend(
        backend,
        runtime_mode=runtime_mode,
        fallback_message=fallback_message,
        remote_config=remote_config,
    )
