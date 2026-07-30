import json

import pytest

from history_chatbot.models.context_budget import ContextBudgetManager
from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.factory import build_llm_from_environment
from history_chatbot.models.remote import (
    HttpResponse,
    HttpStreamResponse,
    RemoteLLMBackend,
    RemoteLLMConfig,
    RemoteLLMError,
    TransportTimeoutError,
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


@pytest.mark.parametrize(
    ("status", "code"),
    [
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


def test_readiness_uses_health_and_ready_without_generation() -> None:
    transport = FakeTransport(
        [
            HttpResponse(200, b'{"status":"ok"}'),
            HttpResponse(200, b'{"ready":true}'),
        ]
    )
    status = backend(transport).readiness()
    assert status["status"] == "ready"
    assert [call[0] for call in transport.calls] == ["GET", "GET"]
    assert all("/generate" not in call[1] and "/chat/completions" not in call[1] for call in transport.calls)


def test_readiness_distinguishes_model_not_ready_and_unreachable() -> None:
    not_ready = backend(
        FakeTransport(
            [
                HttpResponse(200, b'{"status":"ok"}'),
                HttpResponse(200, b'{"ready":false}'),
            ]
        )
    ).readiness()
    assert not_ready["status"] == "model_not_ready"
    unreachable = backend(
        FakeTransport([TransportTimeoutError()])
    ).readiness()
    assert unreachable["status"] == "remote_llm_unreachable"
    assert unreachable["error_code"] == "timeout"


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
