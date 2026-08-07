"""실제 모델 설치 전에도 안정적인 백엔드 경계를 제공하는 factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping

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
    environ: Mapping[str, str] | None = None,
    fallback_message: str = "확인 가능한 자료가 부족합니다.",
) -> object:
    import os

    values = os.environ if environ is None else environ
    requested_backend = values.get(
        "HISTORY_LLM_BACKEND",
        values.get("LLM_BACKEND", "mock" if runtime_mode.allows_fixtures else ""),
    ).strip().lower()
    aliases = {
        "openai_compatible": "openai",
        "project_llama": "project",
    }
    api_format = aliases.get(requested_backend)
    backend = "remote" if api_format else requested_backend
    if not backend:
        raise ValueError("production에는 LLM_BACKEND 설정이 필요합니다.")
    remote_values = dict(values)
    aliases_to_legacy = {
        "HISTORY_LLM_BASE_URL": "LLM_BASE_URL",
        "HISTORY_LLM_MODEL_ID": "LLM_MODEL",
        "HISTORY_LLM_API_KEY": "LLM_API_KEY",
        "HISTORY_LLM_API_FORMAT": "LLM_API_FORMAT",
        "HISTORY_LLM_ALLOWED_HOSTS": "LLM_ALLOWED_HOSTS",
        "HISTORY_LLM_MODEL_REVISION": "LLM_MODEL_REVISION",
    }
    for alias, legacy in aliases_to_legacy.items():
        if remote_values.get(alias, "").strip():
            remote_values[legacy] = remote_values[alias]
    if api_format:
        remote_values["LLM_API_FORMAT"] = api_format
    if backend == "remote":
        missing_fields = [
            name
            for name in ("LLM_BASE_URL", "LLM_MODEL")
            if not remote_values.get(name, "").strip()
        ]
        if (
            _environment_flag(remote_values, "LLM_API_KEY_REQUIRED", False)
            and not remote_values.get("LLM_API_KEY", "").strip()
        ):
            missing_fields.append("LLM_API_KEY")
        if missing_fields:
            return UnconfiguredRemoteLLMBackend(tuple(missing_fields))
    remote_config = (
        RemoteLLMConfig.from_environment(runtime_mode, remote_values)
        if backend == "remote"
        else None
    )
    return build_llm_backend(
        backend,
        runtime_mode=runtime_mode,
        fallback_message=fallback_message,
        remote_config=remote_config,
    )


def _environment_flag(
    values: Mapping[str, str], name: str, default: bool
) -> bool:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}은 true 또는 false여야 합니다.")
