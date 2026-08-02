"""Minimal OpenAI-compatible inference server for an isolated GPU worker.

This file intentionally depends only on the Python standard library at import
time.  PyTorch and Transformers are imported lazily when the model is loaded so
the HTTP contract can be tested without a GPU or model download.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    model_id: str = "local-llama"
    host: str = "127.0.0.1"
    port: int = 8001
    auth_token: str = field(default="", repr=False)
    max_request_bytes: int = 65_536
    max_messages: int = 8
    max_input_chars: int = 12_000
    max_input_tokens: int = 6_144
    max_new_tokens: int = 256

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "ServerConfig":
        values = os.environ if environ is None else environ
        config = cls(
            model_path=values.get("GPU_LLM_MODEL_PATH", "").strip(),
            model_id=values.get("GPU_LLM_MODEL_ID", "local-llama").strip(),
            host=values.get("GPU_LLM_HOST", "127.0.0.1").strip(),
            port=_integer(values, "GPU_LLM_PORT", 8001),
            auth_token=values.get("GPU_LLM_AUTH_TOKEN", ""),
            max_request_bytes=_integer(values, "GPU_LLM_MAX_REQUEST_BYTES", 65_536),
            max_messages=_integer(values, "GPU_LLM_MAX_MESSAGES", 8),
            max_input_chars=_integer(values, "GPU_LLM_MAX_INPUT_CHARS", 12_000),
            max_input_tokens=_integer(values, "GPU_LLM_MAX_INPUT_TOKENS", 6_144),
            max_new_tokens=_integer(values, "GPU_LLM_MAX_NEW_TOKENS", 256),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("GPU_LLM_MODEL_PATH is required")
        if not self.model_id:
            raise ValueError("GPU_LLM_MODEL_ID must not be empty")
        if not self.host:
            raise ValueError("GPU_LLM_HOST must not be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("GPU_LLM_PORT must be between 1 and 65535")
        limits = {
            "GPU_LLM_MAX_REQUEST_BYTES": (self.max_request_bytes, 1_024, 1_048_576),
            "GPU_LLM_MAX_MESSAGES": (self.max_messages, 1, 32),
            "GPU_LLM_MAX_INPUT_CHARS": (self.max_input_chars, 256, 100_000),
            "GPU_LLM_MAX_INPUT_TOKENS": (self.max_input_tokens, 128, 131_072),
            "GPU_LLM_MAX_NEW_TOKENS": (self.max_new_tokens, 1, 2_048),
        }
        for name, (value, minimum, maximum) in limits.items():
            if not minimum <= value <= maximum:
                raise ValueError("%s must be between %s and %s" % (name, minimum, maximum))


class RequestError(ValueError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class TransformersRuntime:
    """One model instance shared by every request handled by the process."""

    def __init__(self, tokenizer: Any, model: Any, torch_module: Any, config: ServerConfig) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.config = config
        self._generation_lock = threading.Lock()

    @classmethod
    def load(cls, config: ServerConfig) -> "TransformersRuntime":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the GPU inference worker")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return cls(tokenizer, model, torch, config)

    def generate(self, messages: List[Dict[str, str]], options: Mapping[str, Any]) -> Tuple[str, int, int]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(prompt) > self.config.max_input_chars:
            raise RequestError(400, "context_length_exceeded", "The prompt is too long")
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > self.config.max_input_tokens:
            raise RequestError(400, "context_length_exceeded", "The prompt is too long")

        model_device = next(self.model.parameters()).device
        encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}
        max_new_tokens = min(int(options["max_tokens"]), self.config.max_new_tokens)
        temperature = float(options["temperature"])
        generate_options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": float(options["top_p"]),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_options["temperature"] = temperature

        with self._generation_lock:
            with self.torch.inference_mode():
                output = self.model.generate(**encoded, **generate_options)
        generated = output[0][prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("model returned an empty response")
        return text, prompt_tokens, int(generated.shape[-1])


class InferenceApplication:
    def __init__(self, config: ServerConfig, runtime: Any) -> None:
        self.config = config
        self.runtime = runtime

    def authorize(self, authorization: str) -> bool:
        if not self.config.auth_token:
            return True
        expected = "Bearer " + self.config.auth_token
        return hmac.compare_digest(authorization, expected)

    def health(self) -> Dict[str, Any]:
        return {"status": "ok"}

    def ready(self) -> Dict[str, Any]:
        return {"ready": True, "status": "ready", "model": self.config.model_id}

    def complete(self, payload: Any) -> Dict[str, Any]:
        messages, options = _validate_payload(payload, self.config)
        text, prompt_tokens, completion_tokens = self.runtime.generate(messages, options)
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class QuietHTTPServer(HTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        # BaseServer prints tracebacks to stderr; suppress them on the shared worker.
        return


def create_handler(application: InferenceApplication) -> Any:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalLlamaHTTP/1.0"

        def log_message(self, format_string: str, *args: Any) -> None:
            # Do not persist request paths, prompts, responses, or auth headers.
            return

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path == "/health":
                self._json(200, application.health())
            elif self.path == "/ready":
                self._json(200, application.ready())
            else:
                self._error(404, "not_found", "Endpoint not found")

        def do_POST(self) -> None:
            if not self._authorized():
                return
            if self.path != "/v1/chat/completions":
                self._error(404, "not_found", "Endpoint not found")
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._error(400, "invalid_request", "Invalid Content-Length")
                return
            if length <= 0 or length > application.config.max_request_bytes:
                self._error(413, "request_too_large", "Request body size is not allowed")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(200, application.complete(payload))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(400, "invalid_json", "Request body must be valid UTF-8 JSON")
            except RequestError as error:
                self._error(error.status, error.code, str(error))
            except Exception:
                # Never expose local paths, model details, prompts, or tracebacks.
                self._error(500, "generation_failed", "Generation failed")

        def _authorized(self) -> bool:
            if application.authorize(self.headers.get("Authorization", "")):
                return True
            self._error(401, "authentication_error", "Authentication failed")
            return False

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _validate_payload(payload: Any, config: ServerConfig) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RequestError(400, "invalid_request", "Request body must be a JSON object")
    if payload.get("stream", False) is not False:
        raise RequestError(400, "streaming_not_supported", "This worker supports non-streaming requests only")
    if payload.get("model") != config.model_id:
        raise RequestError(404, "model_not_found", "Requested model is not available")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > config.max_messages:
        raise RequestError(400, "invalid_request", "messages must be a non-empty bounded list")
    cleaned = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError(400, "invalid_request", "Each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise RequestError(400, "invalid_request", "Each message requires a supported role and text content")
        total_chars += len(content)
        cleaned.append({"role": role, "content": content})
    if total_chars > config.max_input_chars:
        raise RequestError(400, "context_length_exceeded", "The prompt is too long")

    max_tokens = _bounded_number(payload.get("max_tokens", 128), "max_tokens", 1, config.max_new_tokens, int)
    temperature = _bounded_number(payload.get("temperature", 0.2), "temperature", 0.0, 2.0, float)
    top_p = _bounded_number(payload.get("top_p", 0.9), "top_p", 0.01, 1.0, float)
    return cleaned, {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}


def _bounded_number(value: Any, name: str, minimum: float, maximum: float, converter: Any) -> Any:
    if isinstance(value, bool):
        raise RequestError(400, "invalid_request", "%s is invalid" % name)
    if converter is int and not isinstance(value, int):
        raise RequestError(400, "invalid_request", "%s is invalid" % name)
    try:
        converted = converter(value)
    except (TypeError, ValueError):
        raise RequestError(400, "invalid_request", "%s is invalid" % name)
    if not minimum <= converted <= maximum:
        raise RequestError(400, "invalid_request", "%s is outside the allowed range" % name)
    return converted


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        raise ValueError("%s must be an integer" % name)


def main() -> int:
    try:
        config = ServerConfig.from_environment()
        runtime = TransformersRuntime.load(config)
        server = QuietHTTPServer(
            (config.host, config.port),
            create_handler(InferenceApplication(config, runtime)),
        )
    except Exception:
        print("GPU inference worker failed to start", file=sys.stderr, flush=True)
        return 1
    print("GPU inference worker ready on %s:%s" % (config.host, config.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
