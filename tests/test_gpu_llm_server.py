import importlib.util
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from history_chatbot.models.contract import LLMRequest
from history_chatbot.models.remote import (
    OpenAICompatibleAdapter,
    RemoteLLMBackend,
    RemoteLLMConfig,
)
from history_chatbot.runtime import RuntimeMode


SERVER_PATH = Path(__file__).parents[1] / "scripts" / "gpu_llm_server.py"
SERVER_SPEC = importlib.util.spec_from_file_location("gpu_llm_server", SERVER_PATH)
assert SERVER_SPEC is not None and SERVER_SPEC.loader is not None
gpu_llm_server = importlib.util.module_from_spec(SERVER_SPEC)
sys.modules[SERVER_SPEC.name] = gpu_llm_server
SERVER_SPEC.loader.exec_module(gpu_llm_server)

from gpu_llm_server import (  # noqa: E402
    InferenceApplication,
    QuietHTTPServer,
    RequestError,
    ServerConfig,
    TransformersRuntime,
    create_handler,
)


class FakeRuntime:
    def __init__(self, *, error=None) -> None:
        self.calls = []
        self.error = error
        self.response_text = "generated answer"

    def generate(self, messages, options):
        self.calls.append((messages, options))
        if self.error:
            raise self.error
        return self.response_text, 12, 5


def config(**overrides):
    values = {"model_path": "unused-test-path", "model_id": "local-test-model"}
    values.update(overrides)
    return ServerConfig(**values)


@contextmanager
def running_server(application):
    server = QuietHTTPServer(("127.0.0.1", 0), create_handler(application))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%s" % server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request_json(url, *, method="GET", payload=None, token=""):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_environment_defaults_to_localhost_and_requires_model_path() -> None:
    with pytest.raises(ValueError, match="GPU_LLM_MODEL_PATH"):
        ServerConfig.from_environment({})
    selected = ServerConfig.from_environment({"GPU_LLM_MODEL_PATH": "cached-model"})
    assert selected.host == "127.0.0.1"
    assert selected.port == 8001
    protected = config(auth_token="fake-test-token-not-a-real-secret")
    assert "fake-test-token-not-a-real-secret" not in repr(protected)


def test_openai_adapter_payload_is_accepted_without_contract_change() -> None:
    adapter = OpenAICompatibleAdapter()
    app_config = RemoteLLMConfig(
        api_format="openai",
        base_url="http://127.0.0.1:8001",
        model="local-test-model",
    )
    payload = adapter.payload(
        LLMRequest(system_prompt="근거 안에서 답하세요", user_prompt="질문", max_new_tokens=32),
        app_config,
    )
    runtime = FakeRuntime()
    response = InferenceApplication(config(), runtime).complete(payload)
    assert response["choices"][0]["message"]["content"] == runtime.response_text
    assert response["usage"]["total_tokens"] == 17
    assert runtime.calls[0][0][-1] == {"role": "user", "content": "질문"}


def test_remote_backend_readiness_and_completion_reach_worker() -> None:
    secret = "fake-test-token-not-a-real-secret"
    runtime = FakeRuntime()
    application = InferenceApplication(config(auth_token=secret), runtime)
    with running_server(application) as base_url:
        backend = RemoteLLMBackend(
            RemoteLLMConfig(
                api_format="openai",
                base_url=base_url,
                model="local-test-model",
                api_key=secret,
                api_key_required=True,
                readiness_probe_enabled=True,
            ),
            mode=RuntimeMode.DEVELOPMENT,
        )
        readiness = backend.readiness()
        response = backend.complete(
            LLMRequest(system_prompt="system", user_prompt="question", max_new_tokens=32)
        )

    assert readiness == {
        "configured": True,
        "reachable": True,
        "model_ready": True,
        "status": "ready",
    }
    assert response.generated_text == runtime.response_text
    assert response.usage.total_tokens == 17
    assert runtime.calls[0][0] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]


def test_health_ready_optional_auth_and_no_secret_echo() -> None:
    secret = "fake-test-token-not-a-real-secret"
    application = InferenceApplication(config(auth_token=secret), FakeRuntime())
    with running_server(application) as base_url:
        status, error = request_json(base_url + "/health")
        assert status == 401
        assert secret not in json.dumps(error)
        assert request_json(base_url + "/health", token=secret) == (200, {"status": "ok"})
        ready_status, ready = request_json(base_url + "/ready", token=secret)
        assert ready_status == 200
        assert ready == {"ready": True, "status": "ready", "model": "local-test-model"}


def test_http_completion_has_openai_shape_and_hides_internal_exception() -> None:
    runtime = FakeRuntime()
    application = InferenceApplication(config(), runtime)
    payload = {
        "model": "local-test-model",
        "messages": [{"role": "user", "content": "짧은 질문"}],
        "max_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }
    with running_server(application) as base_url:
        status, response = request_json(base_url + "/v1/chat/completions", method="POST", payload=payload)
    assert status == 200
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["finish_reason"] == "stop"

    failing = InferenceApplication(config(), FakeRuntime(error=RuntimeError("/private/model/path")))
    with running_server(failing) as base_url:
        status, response = request_json(base_url + "/v1/chat/completions", method="POST", payload=payload)
    serialized = json.dumps(response)
    assert status == 500
    assert "/private/model/path" not in serialized
    assert "traceback" not in serialized.lower()


def test_request_limits_and_streaming_rejection_are_explicit() -> None:
    application = InferenceApplication(config(max_input_chars=256, max_new_tokens=64), FakeRuntime())
    base = {
        "model": "local-test-model",
        "messages": [{"role": "user", "content": "질문"}],
        "stream": False,
    }
    with pytest.raises(RequestError, match="max_tokens"):
        application.complete({**base, "max_tokens": 65})
    with pytest.raises(RequestError, match="max_tokens"):
        application.complete({**base, "max_tokens": 1.5})
    with pytest.raises(RequestError) as raised:
        application.complete({**base, "stream": True})
    assert raised.value.code == "streaming_not_supported"
    with pytest.raises(RequestError) as raised:
        application.complete({**base, "model": "wrong-model"})
    assert raised.value.code == "model_not_found"
    with pytest.raises(RequestError) as raised:
        application.complete(
            {
                "model": "local-test-model",
                "messages": [{"role": "user", "content": "가" * 257}],
            }
        )
    assert raised.value.code == "context_length_exceeded"


def test_transformers_loads_local_bfloat16_model_once(monkeypatch) -> None:
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return object()

    class FakeModel:
        def eval(self):
            calls.append(("eval",))

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("model", path, kwargs))
            return FakeModel()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    fake_transformers = type(
        "FakeTransformers",
        (),
        {"AutoTokenizer": FakeTokenizerFactory, "AutoModelForCausalLM": FakeModelFactory},
    )()
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    runtime = TransformersRuntime.load(config())
    assert runtime.model is not None
    assert calls[0] == ("tokenizer", "unused-test-path", {"local_files_only": True})
    assert calls[1][0:2] == ("model", "unused-test-path")
    assert calls[1][2]["local_files_only"] is True
    assert calls[1][2]["device_map"] == "auto"
    assert calls[1][2]["torch_dtype"] is FakeTorch.bfloat16
    assert calls.count(("eval",)) == 1
