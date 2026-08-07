import json

import pytest

from history_chatbot.models.context_budget import ContextBudgetManager
from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.factory import build_llm_from_environment
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.models.remote import (
    HttpResponse,
    HttpStreamResponse,
    OpenAICompatibleAdapter,
    ProjectLlamaAdapter,
    RemoteLLMBackend,
    RemoteLLMConfig,
    RemoteLLMError,
    TransportConnectionError,
    TransportTimeoutError,
    UnconfiguredRemoteLLMBackend,
)
from history_chatbot.runtime import RuntimeMode


class FakeTransport:
    def __init__(self, responses=None, stream_response=None) -> None:
        self.responses = list(responses or [])
        self.stream_response = stream_response
        self.calls = []

    def request(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def stream(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        if isinstance(self.stream_response, Exception):
            raise self.stream_response
        return self.stream_response


def config(**overrides):
    values = {
        "api_format": "openai",
        "base_url": "http://localhost:8001",
        "model": "meta-llama/test-instruct",
        "model_revision": "revision-1",
        "api_key": "top-secret-test-key",
        "max_retries": 2,
        "backoff_seconds": 0,
    }
    values.update(overrides)
    return RemoteLLMConfig(**values)


def request(**overrides):
    values = {
        "system_prompt": "근거 안에서만 답변",
        "user_prompt": "테스트 질문",
        "request_id": "request-1",
        "timeout": 10,
    }
    values.update(overrides)
    return LLMRequest(**values)


def openai_response(status=200):
    return HttpResponse(
        status,
        json.dumps(
            {
                "id": "request-1",
                "model": "meta-llama/test-instruct",
                "choices": [
                    {"message": {"content": "근거 기반 답변"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
        ).encode(),
    )


def backend(transport, **overrides):
    return RemoteLLMBackend(
        config(**overrides),
        mode=RuntimeMode.PRODUCTION,
        transport=transport,
        sleep=lambda _: None,
    )


def test_normal_non_streaming_response_and_secret_not_in_url() -> None:
    transport = FakeTransport([openai_response()])
    response = backend(transport).complete(request())
    assert response.generated_text == "근거 기반 답변"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 14
    assert response.model_revision == "revision-1"
    assert response.request_id == "request-1"
    _, url, headers, _, _ = transport.calls[0]
    assert "top-secret-test-key" not in url
    assert headers["Authorization"] == "Bearer top-secret-test-key"


def test_normal_streaming_and_completed_event() -> None:
    stream = HttpStreamResponse(
        200,
        [
            b'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ],
    )
    events = list(backend(FakeTransport(stream_response=stream)).stream_complete(request(stream=True)))
    assert events[0].event == "start"
    assert [event.data["text"] for event in events if event.event == "delta"] == [
        "hello ",
        "world",
    ]
    assert events[-1].event == "completed"
    assert events[-1].data["generated_text"] == "hello world"


def test_timeout_is_structured_and_limited_retry() -> None:
    transport = FakeTransport([TransportTimeoutError()] * 3)
    with pytest.raises(RemoteLLMError) as raised:
        backend(transport).complete(request())
    assert raised.value.code == "timeout"
    assert not raised.value.retryable
    assert len(transport.calls) == 1


def test_connection_error_is_structured_after_retries() -> None:
    transport = FakeTransport([TransportConnectionError()] * 3)
    with pytest.raises(RemoteLLMError) as raised:
        backend(transport).complete(request())
    assert raised.value.code == "connection_error"
    assert raised.value.retryable
    assert len(transport.calls) == 3


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "generation_failed"),
        (401, "authentication_error"),
        (404, "model_not_found"),
        (429, "rate_limited"),
    ],
)
def test_non_retryable_http_errors(status, code) -> None:
    transport = FakeTransport([HttpResponse(status, b"{}")])
    with pytest.raises(RemoteLLMError) as raised:
        backend(transport).complete(request())
    assert raised.value.code == code
    assert not raised.value.retryable
    assert len(transport.calls) == 1


def test_server_error_retries_then_succeeds() -> None:
    transport = FakeTransport([HttpResponse(500, b"{}"), openai_response()])
    response = backend(transport).complete(request())
    assert response.generated_text == "근거 기반 답변"
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        HttpResponse(200, b"not-json"),
        HttpResponse(200, b'{"choices": []}'),
    ],
)
def test_invalid_json_or_missing_fields(response) -> None:
    with pytest.raises(RemoteLLMError) as raised:
        backend(FakeTransport([response])).complete(request())
    assert raised.value.code == "invalid_response"


def test_empty_content_is_rejected() -> None:
    response = openai_response()
    payload = json.loads(response.body)
    payload["choices"][0]["message"]["content"] = "   "
    with pytest.raises(RemoteLLMError) as raised:
        backend(FakeTransport([HttpResponse(200, json.dumps(payload).encode())])).complete(
            request()
        )
    assert raised.value.code == "invalid_response"


def test_stream_disconnect_returns_error_event() -> None:
    stream = HttpStreamResponse(
        200,
        [b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}'],
    )
    events = list(backend(FakeTransport(stream_response=stream)).stream_complete(request(stream=True)))
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "generation_failed"
    assert events[-1].data["retryable"] is False


def test_api_key_is_masked_from_repr_and_error() -> None:
    value = config()
    assert "top-secret-test-key" not in repr(value)
    transport = FakeTransport([HttpResponse(401, b"top-secret-test-key")])
    with pytest.raises(RemoteLLMError) as raised:
        backend(transport).complete(request())
    assert "top-secret-test-key" not in str(raised.value)


def test_production_configuration_and_arbitrary_url_are_rejected() -> None:
    unconfigured = build_llm_from_environment(
        RuntimeMode.PRODUCTION,
        environ={"LLM_BACKEND": "remote", "LLM_MODEL": "model"},
    )
    assert unconfigured.readiness()["status"] == "remote_llm_unconfigured"
    with pytest.raises(ValueError, match="허용 목록"):
        RemoteLLMBackend(
            config(base_url="https://unapproved.example", allowed_hosts=("gpu.example",)),
            mode=RuntimeMode.PRODUCTION,
            transport=FakeTransport(),
        )


def test_environment_factory_keeps_mock_for_development_and_blocks_production() -> None:
    assert isinstance(
        build_llm_from_environment(RuntimeMode.DEVELOPMENT, environ={}), MockLLM
    )
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        build_llm_from_environment(RuntimeMode.PRODUCTION, environ={})


@pytest.mark.parametrize(
    ("backend_name", "adapter_type"),
    [
        ("openai_compatible", OpenAICompatibleAdapter),
        ("project_llama", ProjectLlamaAdapter),
    ],
)
def test_environment_factory_selects_remote_adapter_without_network(
    backend_name, adapter_type
) -> None:
    selected = build_llm_from_environment(
        RuntimeMode.DEVELOPMENT,
        environ={
            "LLM_BACKEND": backend_name,
            "LLM_BASE_URL": "http://localhost:8001",
            "LLM_MODEL": "test-model",
            "LLM_READINESS_PROBE": "false",
        },
    )
    assert isinstance(selected, RemoteLLMBackend)
    assert isinstance(selected.adapter, adapter_type)
    assert selected.readiness()["status"] == "remote_unverified"


def test_history_llm_environment_aliases_select_existing_remote_backend() -> None:
    selected = build_llm_from_environment(
        RuntimeMode.DEVELOPMENT,
        environ={
            "HISTORY_LLM_BACKEND": "openai_compatible",
            "HISTORY_LLM_BASE_URL": "http://localhost:8001",
            "HISTORY_LLM_MODEL_ID": "hackathon-llama",
            "HISTORY_LLM_API_FORMAT": "openai",
            "LLM_READINESS_PROBE": "false",
        },
    )

    assert isinstance(selected, RemoteLLMBackend)
    assert isinstance(selected.adapter, OpenAICompatibleAdapter)
    assert selected.config.base_url == "http://localhost:8001"
    assert selected.config.model == "hackathon-llama"
    assert selected.readiness()["status"] == "remote_unverified"


@pytest.mark.parametrize(
    "environment",
    [
        {"LLM_BACKEND": "remote", "LLM_MODEL": "test-model"},
        {"LLM_BACKEND": "remote", "LLM_BASE_URL": "http://localhost:8001"},
        {
            "LLM_BACKEND": "remote",
            "LLM_BASE_URL": "http://localhost:8001",
            "LLM_MODEL": "test-model",
            "LLM_API_KEY_REQUIRED": "true",
        },
    ],
)
def test_environment_factory_reports_incomplete_remote_configuration(environment) -> None:
    selected = build_llm_from_environment(RuntimeMode.DEVELOPMENT, environ=environment)
    assert isinstance(selected, UnconfiguredRemoteLLMBackend)
    readiness = selected.readiness()
    assert readiness["configuration_status"] == "not_configured"
    assert readiness["missing_fields"]


def test_environment_factory_rejects_invalid_timeout_and_boolean() -> None:
    base = {
        "LLM_BACKEND": "remote",
        "LLM_BASE_URL": "http://localhost:8001",
        "LLM_MODEL": "test-model",
    }
    with pytest.raises(ValueError):
        build_llm_from_environment(
            RuntimeMode.DEVELOPMENT,
            environ={**base, "LLM_TIMEOUT_SECONDS": "not-a-number"},
        )
    with pytest.raises(ValueError, match="LLM_READINESS_PROBE"):
        build_llm_from_environment(
            RuntimeMode.DEVELOPMENT,
            environ={**base, "LLM_READINESS_PROBE": "sometimes"},
        )


def test_development_service_factory_uses_environment_backend(monkeypatch) -> None:
    from history_chatbot.chat import service

    selected = MockLLM("selected")

    class FakeRetrieval:
        def __init__(self, config) -> None:
            self.config = config

        def validate_index(self) -> bool:
            return False

    class FakeSessions:
        def __init__(self, mode, path) -> None:
            self.mode = mode
            self.path = path

    monkeypatch.setattr(service, "HybridRetrievalService", FakeRetrieval)
    monkeypatch.setattr(service, "SessionStore", FakeSessions)
    monkeypatch.setattr(
        service,
        "build_llm_from_environment",
        lambda mode, environ: selected,
    )

    orchestrator = service.create_development_orchestrator(
        environ={"LLM_BACKEND": "openai_compatible"}
    )

    assert orchestrator.llm is selected


def test_context_budget_keeps_system_question_and_high_score_evidence() -> None:
    manager = ContextBudgetManager(256)
    result = manager.fit(
        system_prompt="S" * 40,
        user_prompt="Q" * 20,
        evidence=["E" * 100, "F" * 300],
        conversation=["old" * 100, "recent"],
        max_new_tokens=64,
    )
    assert result.system_prompt
    assert result.user_prompt
    assert result.evidence == ("E" * 100,)
    assert result.trimmed_evidence == 1
    assert result.trimmed_conversation >= 1


def test_openai_readiness_uses_health_and_models_without_generation() -> None:
    transport = FakeTransport(
        [
            HttpResponse(200, b'{"status":"ok"}'),
            HttpResponse(
                200,
                b'{"object":"list","data":[{"id":"meta-llama/test-instruct"}]}',
            ),
        ]
    )
    status = backend(transport).readiness()
    assert status["status"] == "ready"
    assert [call[0] for call in transport.calls] == ["GET", "GET"]
    assert transport.calls[0][1].endswith("/health")
    assert transport.calls[1][1].endswith("/v1/models")
    assert all("/generate" not in call[1] and "/chat/completions" not in call[1] for call in transport.calls)


def test_openai_readiness_falls_back_for_legacy_worker_without_models() -> None:
    transport = FakeTransport(
        [
            HttpResponse(200, b'{"status":"ok"}'),
            HttpResponse(404, b'{"error":"not found"}'),
            HttpResponse(
                200,
                b'{"ready":true,"status":"ready","model":"meta-llama/test-instruct"}',
            ),
        ]
    )

    status = backend(transport).readiness()

    assert status["status"] == "ready"
    assert [call[1].rsplit("/", 1)[-1] for call in transport.calls] == [
        "health",
        "models",
        "ready",
    ]


def test_readiness_distinguishes_model_not_ready_and_unreachable() -> None:
    not_ready = backend(
        FakeTransport(
            [
                HttpResponse(200, b'{"status":"ok"}'),
                HttpResponse(200, b'{"data":[{"id":"another-model"}]}'),
            ]
        )
    ).readiness()
    assert not_ready["status"] == "model_not_ready"
    assert not_ready["reachable"] is True
    assert not_ready["model_ready"] is False
    unreachable = backend(
        FakeTransport([TransportTimeoutError()])
    ).readiness()
    assert unreachable["status"] == "remote_llm_unreachable"
    assert unreachable["error_code"] == "timeout"


def test_project_readiness_keeps_health_and_ready_contract() -> None:
    transport = FakeTransport(
        [
            HttpResponse(200, b'{"status":"ok"}'),
            HttpResponse(200, b'{"ready":true,"status":"ready"}'),
        ]
    )

    status = backend(transport, api_format="project").readiness()

    assert status["status"] == "ready"
    assert transport.calls[0][1].endswith("/health")
    assert transport.calls[1][1].endswith("/ready")


def test_project_api_contract() -> None:
    response = HttpResponse(
        200,
        json.dumps(
            {
                "generated_text": "답변",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
                "model": "model",
                "model_revision": "rev",
                "request_id": "request-1",
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    transport = FakeTransport([response])
    result = backend(transport, api_format="project").complete(request())
    assert result.generated_text == "답변"
    assert transport.calls[0][1].endswith("/generate")
