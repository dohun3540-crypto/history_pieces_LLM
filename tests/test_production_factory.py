from pathlib import Path

import pytest

from history_chatbot.chat import service
from history_chatbot.chat.api import create_app
from history_chatbot.models.remote import RemoteLLMBackend
from history_chatbot.retrieval.service import RetrievalConfig
from history_chatbot.runtime import ProductionNotReadyError, RuntimeMode


class FakeStore:
    def chunks(self):
        return []

    def metadata(self):
        return {"mode": "production"}


class ReadyRetrieval:
    def __init__(self, config) -> None:
        self.config = config
        self.store = FakeStore()
        self.encoder = type(
            "Encoder", (), {"model_id": "hashing-v1", "revision": "builtin"}
        )()

    def validate_index(self):
        return []


class MissingRetrieval(ReadyRetrieval):
    def validate_index(self):
        return ["index artifact missing"]


def remote_environment() -> dict[str, str]:
    return {
        "LLM_BACKEND": "openai_compatible",
        "LLM_BASE_URL": "https://llm.example",
        "LLM_MODEL": "test-model",
        "LLM_ALLOWED_HOSTS": "llm.example",
        "LLM_READINESS_PROBE": "false",
    }


def test_production_factory_assembles_service_without_external_calls(monkeypatch) -> None:
    monkeypatch.setattr(service, "HybridRetrievalService", ReadyRetrieval)

    application_service = service.create_production_service(
        environ=remote_environment(), session_path=None
    )

    assert application_service.orchestrator.mode == RuntimeMode.PRODUCTION
    assert isinstance(application_service.orchestrator.llm, RemoteLLMBackend)
    assert application_service.orchestrator.llm.readiness()["status"] == "remote_unverified"


def test_production_app_assembles_when_api_dependencies_are_installed(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.setattr(service, "HybridRetrievalService", ReadyRetrieval)
    application_service = service.create_production_service(
        environ=remote_environment(), session_path=None
    )

    assert create_app(service=application_service).title == "History Pieces Reference Web Demo"


def test_production_factory_rejects_missing_index(monkeypatch) -> None:
    monkeypatch.setattr(service, "HybridRetrievalService", MissingRetrieval)

    with pytest.raises(ProductionNotReadyError, match="index artifact missing"):
        service.create_production_service(
            environ=remote_environment(), session_path=None
        )


def test_production_factory_rejects_non_production_retrieval_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        RetrievalConfig,
        "load",
        classmethod(
            lambda cls, path: RetrievalConfig(
                runtime_mode="development",
                fixture_chunks_path=Path("fixture.jsonl"),
            )
        ),
    )

    with pytest.raises(ValueError, match="production retrieval"):
        service.create_production_service(
            environ=remote_environment(), session_path=None
        )


def test_production_readiness_reports_unconfigured_llm(monkeypatch) -> None:
    monkeypatch.setattr(service, "HybridRetrievalService", ReadyRetrieval)

    application_service = service.create_production_service(
        environ={"LLM_BACKEND": "remote"}, session_path=None
    )

    readiness = application_service.readiness()
    assert readiness["status"] == "remote_llm_unconfigured"
    assert readiness["missing_llm_backend"] is True
