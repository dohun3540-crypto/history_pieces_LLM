"""외부 Llama 추론 서버용 보안·재시도·스키마 검증 백엔드."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from history_chatbot.models.contract import (
    ChatCompletionBackend,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    TokenUsage,
)
from history_chatbot.chat.remote_safe import RemotePromptPolicy
from history_chatbot.runtime import RuntimeMode


ERROR_CODES = {
    "connection_error",
    "timeout",
    "authentication_error",
    "rate_limited",
    "model_not_found",
    "invalid_response",
    "server_not_ready",
    "context_length_exceeded",
    "generation_failed",
}
class RemoteLLMError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"알 수 없는 Remote LLM 오류 코드: {code}")
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


class TransportConnectionError(RuntimeError):
    pass


class TransportTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes


@dataclass(frozen=True, slots=True)
class HttpStreamResponse:
    status: int
    lines: Iterable[bytes]


class HttpTransport(Protocol):
    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float
    ) -> HttpResponse: ...

    def stream(
        self, method: str, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> HttpStreamResponse: ...


class UrllibHttpTransport:
    def request(self, method, url, headers, body, timeout) -> HttpResponse:
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return HttpResponse(response.status, response.read())
        except HTTPError as error:
            return HttpResponse(error.code, error.read())
        except TimeoutError as error:
            raise TransportTimeoutError("원격 LLM 요청 시간이 초과되었습니다.") from error
        except (URLError, OSError) as error:
            raise TransportConnectionError("원격 LLM 서버에 연결할 수 없습니다.") from error

    def stream(self, method, url, headers, body, timeout) -> HttpStreamResponse:
        request = Request(url, data=body, headers=headers, method=method)
        try:
            response = urlopen(request, timeout=timeout)  # noqa: S310
            return HttpStreamResponse(response.status, response)
        except HTTPError as error:
            return HttpStreamResponse(error.code, ())
        except TimeoutError as error:
            raise TransportTimeoutError("원격 LLM 스트림 연결 시간이 초과되었습니다.") from error
        except (URLError, OSError) as error:
            raise TransportConnectionError("원격 LLM 스트림에 연결할 수 없습니다.") from error


@dataclass(frozen=True, slots=True)
class RemoteLLMConfig:
    api_format: str
    base_url: str
    model: str
    model_revision: str = ""
    api_key: str = field(default="", repr=False)
    api_key_required: bool = False
    readiness_probe_enabled: bool = True
    timeout_seconds: float = 60.0
    max_retries: int = 2
    backoff_seconds: float = 0.25
    allowed_hosts: tuple[str, ...] = ()
    context_window: int = 8192
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    remote_history_enabled: bool = False
    remote_history_max_turns: int = 1
    remote_context_max_chars: int = 12_000
    remote_chunk_max_chars: int = 1_600
    remote_max_evidence_items: int = 4
    remote_sanitize_enabled: bool = True

    def validate(self, mode: RuntimeMode) -> None:
        if self.api_format not in {"openai", "project"}:
            raise ValueError("LLM API 형식은 openai 또는 project여야 합니다.")
        if mode == RuntimeMode.PRODUCTION and (not self.base_url or not self.model):
            raise ValueError("production에는 LLM_BASE_URL과 LLM_MODEL이 필요합니다.")
        if self.api_key_required and not self.api_key:
            raise ValueError("LLM_API_KEY_REQUIRED=true이면 LLM_API_KEY가 필요합니다.")
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("LLM_BASE_URL은 유효한 http(s) URL이어야 합니다.")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("LLM_BASE_URL에 인증정보, query 또는 fragment를 넣을 수 없습니다.")
        host = parts.hostname.lower()
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if mode == RuntimeMode.PRODUCTION and host not in local_hosts and not self.allowed_hosts:
            raise ValueError("production 원격 LLM에는 LLM_ALLOWED_HOSTS 설정이 필요합니다.")
        if mode == RuntimeMode.PRODUCTION and self.allowed_hosts and host not in self.allowed_hosts:
            raise ValueError("production 허용 목록에 없는 LLM 서버입니다.")
        if mode == RuntimeMode.PRODUCTION and host not in local_hosts and parts.scheme != "https":
            raise ValueError("production 비로컬 LLM 서버는 https를 사용해야 합니다.")
        if not 0 <= self.max_retries <= 3:
            raise ValueError("max_retries는 0~3이어야 합니다.")
        if not 0.1 <= self.timeout_seconds <= 600:
            raise ValueError("timeout은 0.1~600초여야 합니다.")
        if not 256 <= self.context_window <= 1_000_000:
            raise ValueError("context_window 범위가 올바르지 않습니다.")
        LLMRequest(
            system_prompt="validation",
            user_prompt="validation",
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            timeout=self.timeout_seconds,
        )
        self.remote_prompt_policy().validate()

    def remote_prompt_policy(self) -> RemotePromptPolicy:
        return RemotePromptPolicy(
            history_enabled=self.remote_history_enabled,
            history_max_turns=self.remote_history_max_turns,
            context_max_chars=self.remote_context_max_chars,
            chunk_max_chars=self.remote_chunk_max_chars,
            max_evidence_items=self.remote_max_evidence_items,
            sanitize_enabled=self.remote_sanitize_enabled,
        )

    @classmethod
    def from_environment(
        cls, mode: RuntimeMode, environ: Mapping[str, str] | None = None
    ) -> "RemoteLLMConfig":
        source = os.environ if environ is None else environ
        values = dict(source)
        aliases_to_legacy = {
            "HISTORY_LLM_BASE_URL": "LLM_BASE_URL",
            "HISTORY_LLM_MODEL_ID": "LLM_MODEL",
            "HISTORY_LLM_API_KEY": "LLM_API_KEY",
            "HISTORY_LLM_API_FORMAT": "LLM_API_FORMAT",
            "HISTORY_LLM_ALLOWED_HOSTS": "LLM_ALLOWED_HOSTS",
            "HISTORY_LLM_MODEL_REVISION": "LLM_MODEL_REVISION",
        }
        for alias, legacy in aliases_to_legacy.items():
            if values.get(alias, "").strip():
                values[legacy] = values[alias]
        config = cls(
            api_format=values.get("LLM_API_FORMAT", "openai"),
            base_url=values.get("LLM_BASE_URL", ""),
            api_key=values.get("LLM_API_KEY", ""),
            api_key_required=_environment_flag(
                values, "LLM_API_KEY_REQUIRED", False
            ),
            readiness_probe_enabled=_environment_flag(
                values, "LLM_READINESS_PROBE", False
            ),
            model=values.get("LLM_MODEL", ""),
            model_revision=values.get("LLM_MODEL_REVISION", ""),
            timeout_seconds=float(values.get("LLM_TIMEOUT_SECONDS") or 60),
            max_new_tokens=int(values.get("LLM_MAX_NEW_TOKENS") or 512),
            temperature=float(values.get("LLM_TEMPERATURE") or 0.2),
            top_p=float(values.get("LLM_TOP_P") or 0.9),
            context_window=int(values.get("LLM_CONTEXT_WINDOW") or 8192),
            allowed_hosts=tuple(
                host.strip().lower()
                for host in values.get("LLM_ALLOWED_HOSTS", "").split(",")
                if host.strip()
            ),
            remote_history_enabled=_environment_flag(
                values, "LLM_REMOTE_HISTORY_ENABLED", False
            ),
            remote_history_max_turns=int(
                values.get("LLM_REMOTE_HISTORY_MAX_TURNS") or 1
            ),
            remote_context_max_chars=int(
                values.get("LLM_REMOTE_CONTEXT_MAX_CHARS") or 12_000
            ),
            remote_chunk_max_chars=int(
                values.get("LLM_REMOTE_CHUNK_MAX_CHARS") or 1_600
            ),
            remote_max_evidence_items=int(
                values.get("LLM_REMOTE_MAX_EVIDENCE_ITEMS") or 4
            ),
            remote_sanitize_enabled=_environment_flag(
                values, "LLM_REMOTE_SANITIZE_ENABLED", True
            ),
        )
        config.validate(mode)
        return config


class ServerAdapter(Protocol):
    generate_path: str
    stream_path: str
    health_path: str
    ready_path: str
    fallback_ready_path: str | None

    def payload(self, request: LLMRequest, config: RemoteLLMConfig) -> dict[str, object]: ...
    def parse_response(
        self, payload: dict[str, object], request: LLMRequest, config: RemoteLLMConfig, latency_ms: int
    ) -> LLMResponse: ...
    def parse_stream_line(self, line: str) -> tuple[str, str | dict[str, object]] | None: ...
    def is_ready(
        self, payload: dict[str, object], config: RemoteLLMConfig, path: str
    ) -> bool: ...


class OpenAICompatibleAdapter:
    generate_path = stream_path = "/v1/chat/completions"
    health_path = "/health"
    ready_path = "/v1/models"
    fallback_ready_path = "/ready"

    def payload(self, request: LLMRequest, config: RemoteLLMConfig) -> dict[str, object]:
        messages = [{"role": "system", "content": request.system_prompt}]
        messages.extend({"role": item.role, "content": item.content} for item in request.messages)
        messages.append({"role": "user", "content": request.user_prompt})
        return {
            "model": config.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_new_tokens,
            "stop": list(request.stop_sequences) or None,
            "stream": request.stream,
            "user": request.request_id,
        }

    def parse_response(self, payload, request, config, latency_ms) -> LLMResponse:
        try:
            choice = payload["choices"][0]  # type: ignore[index]
            text = choice["message"]["content"]  # type: ignore[index]
            usage = payload["usage"]  # type: ignore[index]
            return _response(
                text,
                choice.get("finish_reason", "stop"),  # type: ignore[union-attr]
                usage,
                str(payload.get("model") or config.model),
                config.model_revision,
                str(payload.get("id") or request.request_id),
                latency_ms,
            )
        except (KeyError, IndexError, TypeError) as error:
            raise RemoteLLMError("invalid_response", "원격 LLM 응답 필드가 올바르지 않습니다.") from error

    def parse_stream_line(self, line):
        value = line.removeprefix("data:").strip()
        if not value:
            return None
        if value == "[DONE]":
            return ("completed", {})
        try:
            payload = json.loads(value)
            choice = payload["choices"][0]
            delta = choice.get("delta", {}).get("content", "")
            if delta:
                return ("delta", str(delta))
            if choice.get("finish_reason"):
                return ("completed", {"finish_reason": choice["finish_reason"]})
            return None
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RemoteLLMError("invalid_response", "스트리밍 응답 형식이 올바르지 않습니다.") from error

    def is_ready(self, payload, config, path):
        if path == self.fallback_ready_path:
            reported_model = payload.get("model")
            return bool(
                payload.get("ready", payload.get("status") == "ready")
                and (not reported_model or reported_model == config.model)
            )
        models = payload.get("data")
        if not isinstance(models, list):
            raise RemoteLLMError(
                "invalid_response", "원격 LLM 모델 목록 형식이 올바르지 않습니다."
            )
        return any(
            isinstance(item, dict) and item.get("id") == config.model
            for item in models
        )


class ProjectLlamaAdapter:
    generate_path = "/generate"
    stream_path = "/generate/stream"
    health_path = "/health"
    ready_path = "/ready"
    fallback_ready_path = None

    def payload(self, request, config):
        return {
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "messages": [asdict(item) for item in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_new_tokens,
            "stop_sequences": list(request.stop_sequences),
            "stream": request.stream,
            "request_id": request.request_id,
            "model": config.model,
            "model_revision": config.model_revision,
        }

    def parse_response(self, payload, request, config, latency_ms):
        try:
            usage = payload["usage"]
            return _response(
                payload["generated_text"],
                payload["finish_reason"],
                usage,
                payload.get("model") or config.model,
                payload.get("model_revision") or config.model_revision,
                payload.get("request_id") or request.request_id,
                latency_ms,
            )
        except (KeyError, TypeError) as error:
            raise RemoteLLMError("invalid_response", "원격 LLM 응답 필드가 올바르지 않습니다.") from error

    def parse_stream_line(self, line):
        try:
            payload = json.loads(line)
            event = payload["event"]
            if event in {"token", "delta"}:
                return (event, str(payload.get("text") or payload.get("delta") or ""))
            if event == "completed":
                return ("completed", payload)
            if event == "error":
                return ("error", payload)
            if event == "start":
                return ("start", payload)
            raise KeyError(event)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RemoteLLMError("invalid_response", "스트리밍 응답 형식이 올바르지 않습니다.") from error

    def is_ready(self, payload, config, path):
        del config, path
        return bool(payload.get("ready", payload.get("status") == "ready"))


class RemoteLLMBackend(ChatCompletionBackend):
    backend_name = "remote"

    def __init__(
        self,
        config: RemoteLLMConfig,
        *,
        mode: RuntimeMode,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate(mode)
        self.config = config
        self.mode = mode
        self.transport = transport or UrllibHttpTransport()
        self.sleep = sleep
        self.monotonic = monotonic
        self.adapter: ServerAdapter = (
            OpenAICompatibleAdapter()
            if config.api_format == "openai"
            else ProjectLlamaAdapter()
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = self.monotonic()
        payload = self.adapter.payload(request, self.config)
        response = self._request_with_retry(
            self.adapter.generate_path,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            request.timeout,
            started,
        )
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteLLMError("invalid_response", "원격 LLM이 잘못된 JSON을 반환했습니다.") from error
        return self.adapter.parse_response(
            parsed,
            request,
            self.config,
            int((self.monotonic() - started) * 1000),
        )

    def stream_complete(self, request: LLMRequest):
        started = self.monotonic()
        yield LLMStreamEvent(
            "start",
            {"request_id": request.request_id, "model": self.config.model},
        )
        text_parts: list[str] = []
        finish_reason = "stop"
        try:
            payload = self.adapter.payload(request, self.config)
            response = self._open_stream_with_retry(
                self.adapter.stream_path,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                request.timeout,
                started,
            )
            for raw_line in response.lines:
                if self.monotonic() - started >= min(
                    request.timeout, self.config.timeout_seconds
                ):
                    raise RemoteLLMError(
                        "timeout", "원격 LLM 스트림 deadline을 초과했습니다."
                    )
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError as error:
                    raise RemoteLLMError(
                        "invalid_response", "스트리밍 응답이 UTF-8이 아닙니다."
                    ) from error
                if not line:
                    continue
                parsed = self.adapter.parse_stream_line(line)
                if parsed is None:
                    continue
                event, data = parsed
                if event in {"token", "delta"}:
                    value = str(data)
                    text_parts.append(value)
                    yield LLMStreamEvent(event, {"text": value})
                elif event == "completed":
                    if isinstance(data, dict):
                        finish_reason = str(data.get("finish_reason", finish_reason))
                    break
                elif event == "error":
                    raise RemoteLLMError("generation_failed", "원격 생성 스트림이 오류를 반환했습니다.")
            else:
                raise RemoteLLMError(
                    "generation_failed",
                    "원격 생성 스트림이 완료 이벤트 없이 종료되었습니다.",
                )
            if not "".join(text_parts).strip():
                raise RemoteLLMError(
                    "invalid_response", "원격 LLM이 빈 응답을 반환했습니다."
                )
            response_value = LLMResponse(
                "".join(text_parts),
                finish_reason,
                TokenUsage(0, 0, 0),
                self.config.model,
                self.config.model_revision,
                request.request_id,
                int((self.monotonic() - started) * 1000),
            )
            yield LLMStreamEvent("completed", response_value.to_dict())
        except (RemoteLLMError, TransportConnectionError, TransportTimeoutError) as error:
            normalized = self._normalize_transport_error(error)
            yield LLMStreamEvent("error", normalized.to_dict())

    def readiness(self) -> dict[str, object]:
        if not self.config.readiness_probe_enabled:
            return {
                "configured": True,
                "reachable": False,
                "model_ready": False,
                "status": "remote_unverified",
                "verification": "not_probed",
            }
        try:
            health = self.transport.request(
                "GET", self._url(self.adapter.health_path), self._headers(), None, 3.0
            )
            self._raise_for_status(health.status)
            ready_path = self.adapter.ready_path
            ready = self.transport.request(
                "GET", self._url(ready_path), self._headers(), None, 3.0
            )
            if ready.status == 404 and self.adapter.fallback_ready_path:
                ready_path = self.adapter.fallback_ready_path
                ready = self.transport.request(
                    "GET", self._url(ready_path), self._headers(), None, 3.0
                )
            self._raise_for_status(ready.status)
            payload = json.loads(ready.body.decode("utf-8") or "{}")
            model_ready = self.adapter.is_ready(payload, self.config, ready_path)
            return {
                "configured": True,
                "reachable": True,
                "model_ready": model_ready,
                "status": "ready" if model_ready else "model_not_ready",
            }
        except Exception as error:
            normalized = self._normalize_transport_error(error)
            return {
                "configured": True,
                "reachable": False,
                "model_ready": False,
                "status": "remote_llm_unreachable",
                "error_code": normalized.code,
            }

    def _request_with_retry(self, path, body, request_timeout, started):
        last: RemoteLLMError | None = None
        deadline = started + min(request_timeout, self.config.timeout_seconds)
        for attempt in range(self.config.max_retries + 1):
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise RemoteLLMError("timeout", "원격 LLM 요청 deadline을 초과했습니다.", retryable=True)
            try:
                response = self.transport.request(
                    "POST", self._url(path), self._headers(), body, remaining
                )
                self._raise_for_status(response.status)
                return response
            except (RemoteLLMError, TransportConnectionError, TransportTimeoutError) as error:
                last = self._normalize_transport_error(error)
                if not last.retryable or attempt >= self.config.max_retries:
                    raise last from None
                self.sleep(min(self.config.backoff_seconds * (2**attempt), remaining))
        raise last or RemoteLLMError("generation_failed", "원격 생성에 실패했습니다.")

    def _open_stream_with_retry(self, path, body, request_timeout, started):
        last: RemoteLLMError | None = None
        deadline = started + min(request_timeout, self.config.timeout_seconds)
        for attempt in range(self.config.max_retries + 1):
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise RemoteLLMError("timeout", "원격 LLM 스트림 deadline을 초과했습니다.")
            try:
                response = self.transport.stream(
                    "POST", self._url(path), self._headers(), body, remaining
                )
                self._raise_for_status(response.status)
                return response
            except (RemoteLLMError, TransportConnectionError, TransportTimeoutError) as error:
                last = self._normalize_transport_error(error)
                if not last.retryable or attempt >= self.config.max_retries:
                    raise last from None
                self.sleep(min(self.config.backoff_seconds * (2**attempt), remaining))
        raise last or RemoteLLMError("generation_failed", "원격 스트림 연결에 실패했습니다.")

    def _headers(self):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _url(self, path):
        return f"{self.config.base_url.rstrip('/')}{path}"

    @staticmethod
    def _raise_for_status(status):
        if 200 <= status < 300:
            return
        mapping = {
            400: ("generation_failed", False),
            401: ("authentication_error", False),
            403: ("authentication_error", False),
            404: ("model_not_found", False),
            408: ("timeout", False),
            413: ("context_length_exceeded", False),
            429: ("rate_limited", False),
            503: ("server_not_ready", True),
        }
        code, retryable = mapping.get(status, ("generation_failed", status >= 500))
        raise RemoteLLMError(code, f"원격 LLM 요청 실패({code})", retryable=retryable)

    @staticmethod
    def _normalize_transport_error(error) -> RemoteLLMError:
        if isinstance(error, RemoteLLMError):
            return error
        if isinstance(error, TransportTimeoutError):
            return RemoteLLMError("timeout", "원격 LLM 요청 시간이 초과되었습니다.")
        if isinstance(error, TransportConnectionError):
            return RemoteLLMError("connection_error", "원격 LLM 서버에 연결할 수 없습니다.", retryable=True)
        return RemoteLLMError("generation_failed", "원격 LLM 처리에 실패했습니다.")


def _response(text, finish_reason, usage, model, revision, request_id, latency):
    if not isinstance(text, str) or not text.strip() or not isinstance(usage, dict):
        raise RemoteLLMError("invalid_response", "원격 LLM 응답 타입이 올바르지 않습니다.")
    try:
        prompt = int(usage["prompt_tokens"])
        completion = int(usage["completion_tokens"])
        total = int(usage.get("total_tokens", prompt + completion))
    except (KeyError, TypeError, ValueError) as error:
        raise RemoteLLMError("invalid_response", "token usage가 올바르지 않습니다.") from error
    return LLMResponse(
        text,
        str(finish_reason),
        TokenUsage(prompt, completion, total),
        str(model),
        str(revision),
        str(request_id),
        latency,
    )


class UnconfiguredRemoteLLMBackend(ChatCompletionBackend):
    """설정 누락을 readiness로 보고하기 위한 비생성 backend."""

    backend_name = "remote"

    def __init__(self, missing_fields: tuple[str, ...] = ()) -> None:
        self.missing_fields = missing_fields

    def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        raise RemoteLLMError("server_not_ready", "원격 LLM 설정이 완료되지 않았습니다.")

    def stream_complete(self, request: LLMRequest):
        yield LLMStreamEvent(
            "error",
            {
                "code": "server_not_ready",
                "message": "원격 LLM 설정이 완료되지 않았습니다.",
                "retryable": False,
                "request_id": request.request_id,
            },
        )

    def readiness(self) -> dict[str, object]:
        return {
            "configured": False,
            "reachable": False,
            "model_ready": False,
            "status": "remote_llm_unconfigured",
            "configuration_status": "not_configured",
            "missing_fields": list(self.missing_fields),
        }


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
