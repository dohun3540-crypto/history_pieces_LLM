"""실제 모델 설치 전에도 안정적인 백엔드 경계를 제공하는 factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from history_chatbot.models.base import BaseLLM, GenerationRequest
from history_chatbot.models.mock_llm import MockLLM
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


class RemoteLLMBackend(StreamingLLMBackend):
    def generate(self, request: GenerationRequest) -> str:
        raise RuntimeError("원격 LLM 서버가 아직 구성되지 않았습니다.")

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        raise RuntimeError("원격 LLM 서버가 아직 구성되지 않았습니다.")
        yield  # pragma: no cover


def build_llm_backend(
    backend: str,
    *,
    runtime_mode: RuntimeMode,
    fallback_message: str,
) -> BaseLLM:
    if backend == "mock":
        if runtime_mode == RuntimeMode.PRODUCTION:
            raise ValueError("production 모드에서는 MockLLM을 사용할 수 없습니다.")
        return MockLLM(fallback_message)
    if backend == "local":
        return LocalLLMBackend()
    if backend == "remote":
        return RemoteLLMBackend()
    raise ValueError(f"지원하지 않는 LLM backend입니다: {backend}")
