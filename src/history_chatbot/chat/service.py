"""CLI와 HTTP 어댑터가 공유하는 애플리케이션 서비스."""

from __future__ import annotations

from pathlib import Path

from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator
from history_chatbot.chat.session import SessionStore
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig
from history_chatbot.runtime import RuntimeMode


def create_development_orchestrator(
    *,
    runtime_dir: Path = Path(".runtime/development"),
    session_path: Path | None = None,
) -> ConversationalRagOrchestrator:
    mode = RuntimeMode.DEVELOPMENT
    config = RetrievalConfig(
        runtime_mode=mode.value,
        fixture_chunks_path=Path("tests/fixtures/rag/fictional_chunks.jsonl"),
        local_storage_path=runtime_dir / "retrieval",
        index_ready_path=Path("data/index_ready"),
        minimum_score=0.20,
        minimum_dense_score=0.72,
        final_top_k=10,
        max_chunks_per_document=2,
    )
    retrieval = HybridRetrievalService(config)
    if retrieval.validate_index():
        retrieval.build_index()
    sessions = SessionStore(
        mode,
        path=session_path or runtime_dir / "sessions.json",
    )
    return ConversationalRagOrchestrator(
        retrieval,
        MockLLM("확인 가능한 자료가 부족합니다."),
        sessions,
        mode=mode,
    )


class ChatApplicationService:
    def __init__(self, orchestrator: ConversationalRagOrchestrator) -> None:
        self.orchestrator = orchestrator

    def chat(self, payload: dict[str, object]) -> dict[str, object]:
        response = self.orchestrator.ask(
            str(payload.get("user_query", "")),
            session_id=str(payload["session_id"]) if payload.get("session_id") else None,
            locale=str(payload.get("locale", "ko")),
            top_k=int(payload.get("top_k", 3)),
        )
        return response.to_dict()

    def stream(self, payload: dict[str, object]):
        yield from self.orchestrator.stream(
            str(payload.get("user_query", "")),
            session_id=str(payload["session_id"]) if payload.get("session_id") else None,
            locale=str(payload.get("locale", "ko")),
            top_k=int(payload.get("top_k", 3)),
        )

    def reset(self, session_id: str) -> dict[str, object]:
        return {"session_id": session_id, "reset": self.orchestrator.reset(session_id)}

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def readiness(self) -> dict[str, object]:
        mode = self.orchestrator.mode
        errors = self.orchestrator.retrieval.validate_index()
        llm = self.orchestrator.llm.readiness()
        chunks = self.orchestrator.retrieval.store.chunks()
        real_documents = any(
            chunk.payload.get("data_classification") != "fictional_fixture"
            for chunk in chunks
        )
        if mode == RuntimeMode.DEVELOPMENT:
            status = "development_ready" if not errors else "missing_index"
        elif not llm.get("configured"):
            status = "remote_llm_unconfigured"
        elif not llm.get("reachable"):
            status = "remote_llm_unreachable"
        elif not llm.get("model_ready"):
            status = "model_not_ready"
        elif not real_documents:
            status = "missing_real_documents"
        elif errors:
            status = "missing_production_index"
        else:
            status = "production_ready"
        return {
            "status": status,
            "llm_configured": bool(llm.get("configured")),
            "remote_server_reachable": bool(llm.get("reachable")),
            "model_ready": bool(llm.get("model_ready")),
            "missing_real_documents": mode == RuntimeMode.PRODUCTION and not real_documents,
            "missing_llm_backend": mode == RuntimeMode.PRODUCTION and not llm.get("configured"),
            "missing_index": bool(errors),
        }
