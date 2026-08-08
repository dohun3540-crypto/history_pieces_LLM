"""CLI와 HTTP 어댑터가 공유하는 애플리케이션 서비스."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from history_chatbot.chat.orchestrator import ConversationalRagOrchestrator, StreamEvent
from history_chatbot.chat.session import SessionStore
from history_chatbot.models.contract import ChatCompletionBackend
from history_chatbot.models.factory import build_llm_from_environment
from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig
from history_chatbot.runtime import ProductionNotReadyError, RuntimeMode


def create_development_orchestrator(
    *,
    runtime_dir: Path = Path(".runtime/development"),
    session_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    llm: ChatCompletionBackend | None = None,
    in_memory_sessions: bool = False,
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
        path=(None if in_memory_sessions else session_path or runtime_dir / "sessions.json"),
    )
    return ConversationalRagOrchestrator(
        retrieval,
        llm or build_llm_from_environment(mode, environ=environ),
        sessions,
        mode=mode,
    )


def create_development_integration_service(
    *,
    runtime_dir: Path = Path(".runtime/development-integration"),
    environ: Mapping[str, str] | None = None,
    llm: ChatCompletionBackend | None = None,
) -> ChatApplicationService:
    """Build the explicit fictional-fixture integration service with memory sessions."""

    return ChatApplicationService(
        create_development_orchestrator(
            runtime_dir=runtime_dir,
            environ=environ,
            llm=llm,
            in_memory_sessions=True,
        )
    )


def create_development_real_service(
    *,
    retrieval_config_path: Path = Path("configs/retrieval.development-real.yaml"),
    llm: ChatCompletionBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> ChatApplicationService:
    """Build the isolated real-source development service with memory sessions."""

    mode = RuntimeMode.DEVELOPMENT
    config = RetrievalConfig.load(retrieval_config_path)
    if RuntimeMode.parse(config.runtime_mode) != mode:
        raise ValueError("development_real service에는 development 설정이 필요합니다.")
    if config.development_chunks_path is None:
        raise ValueError("development_chunks_path가 필요합니다.")
    retrieval = HybridRetrievalService(config)
    if retrieval.validate_index():
        retrieval.build_index()
    sessions = SessionStore(mode, path=None)
    return ChatApplicationService(
        ConversationalRagOrchestrator(
            retrieval,
            llm or build_llm_from_environment(mode, environ=environ),
            sessions,
            mode=mode,
            max_chunks_per_document=config.max_chunks_per_document,
        )
    )


def create_hackathon_orchestrator(
    *,
    runtime_dir: Path = Path(".runtime/hackathon"),
    chunks_path: Path = Path("data/provisional_hackathon/processed/chunks.jsonl"),
    session_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    llm: ChatCompletionBackend | None = None,
) -> ConversationalRagOrchestrator:
    mode = RuntimeMode.HACKATHON
    config = RetrievalConfig(
        runtime_mode=mode.value,
        provisional_chunks_path=chunks_path,
        local_storage_path=Path(".runtime/indexes/hackathon"),
        index_ready_path=Path("data/index_ready"),
        minimum_score=0.20,
        minimum_dense_score=0.72,
        final_top_k=10,
        max_chunks_per_document=2,
    )
    retrieval = HybridRetrievalService(config)
    if retrieval.validate_index() and chunks_path.is_file():
        retrieval.build_index()
    sessions = SessionStore(
        mode,
        path=session_path or runtime_dir / "sessions.json",
    )
    return ConversationalRagOrchestrator(
        retrieval,
        llm or build_llm_from_environment(mode, environ=environ),
        sessions,
        mode=mode,
    )


class HistoryChatService:
    def __init__(self, orchestrator: ConversationalRagOrchestrator) -> None:
        self.orchestrator = orchestrator

    def chat(self, payload: dict[str, object]) -> dict[str, object]:
        query = self._query(payload)
        response = self.orchestrator.ask(
            query,
            session_id=str(payload["session_id"]) if payload.get("session_id") else None,
            locale=str(payload.get("locale", "ko")),
            top_k=int(payload.get("top_k", 3)),
            conversation_mode=str(payload.get("conversation_mode", "free_chat")),
            screen_type=str(payload["screen_type"]) if payload.get("screen_type") else None,
            current_piece_id=str(payload["current_piece_id"]) if payload.get("current_piece_id") else None,
            current_place_id=str(payload["current_place_id"]) if payload.get("current_place_id") else None,
            completed_place_ids=tuple(str(x) for x in payload.get("completed_place_ids", ())),
            visited_piece_ids=tuple(str(x) for x in payload.get("visited_piece_ids", ())),
            existing_style_preferences=tuple(str(x) for x in payload.get("existing_style_preferences", ())),
            current_journey_step=str(payload["current_journey_step"]) if payload.get("current_journey_step") else None,
            piece_follow_up_count=(
                int(payload["piece_follow_up_count"])
                if payload.get("piece_follow_up_count") is not None else None
            ),
            return_target=str(payload.get("return_target", "game")),
            available_capabilities=tuple(str(x) for x in payload.get("available_capabilities", ())),
            storage_capability=payload.get("storage_capability") is True,
            user_consent=payload.get("user_consent") is True,
        )
        return response.to_dict()

    def search(self, query: str, *, top_k: int = 5) -> dict[str, object]:
        """LLM 호출 없이 기존 hybrid retriever 결과를 반환한다."""

        value = query.strip()
        if not value:
            raise ValueError("질문을 입력하세요.")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k는 1~10이어야 합니다.")
        results = self.orchestrator.retrieval.search(value)[:top_k]
        return {
            "query": value,
            "results": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.document_id,
                    "title": item.chunk.title,
                    "text": item.chunk.text,
                    "score": round(item.score, 6),
                    "source_name": item.chunk.publisher,
                    "source_url": item.chunk.source_url,
                }
                for item in results
            ],
        }

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        session_id: str | None = None,
        locale: str = "ko",
        current_place_id: str | None = None,
        current_piece_id: str | None = None,
        completed_place_ids: tuple[str, ...] = (),
        completed_piece_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """대화 문맥은 해석에만, 검색 chunk는 사실 근거로만 사용한다."""

        persistent = session_id is not None
        if persistent:
            session = self.orchestrator.sessions.get(session_id)
            if session is None:
                raise ValueError("존재하지 않는 session_id입니다.")
        else:
            session = self.orchestrator.sessions.create(locale)
        pending_user: str | None = None
        history_to_import = (chat_history or []) if not session.turns else []
        for message in history_to_import:
            role = str(message.get("role", ""))
            content = str(message.get("content", "")).strip()
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                self.orchestrator.sessions.add_turn(
                    session.session_id, pending_user, content
                )
                pending_user = None
        try:
            response = self.chat(
                {
                    "user_query": question,
                    "session_id": session.session_id,
                    "locale": locale,
                    "conversation_mode": "free_chat",
                    "screen_type": "free_chat",
                    "current_place_id": current_place_id,
                    "current_piece_id": current_piece_id,
                    "completed_place_ids": completed_place_ids,
                    "visited_piece_ids": completed_piece_ids,
                }
            )
        finally:
            if not persistent:
                self.orchestrator.sessions.reset(session.session_id)
        grounded = response.get("grounded") is True
        answer = str(response.get("answer", ""))
        if not grounded and response.get("status") == "insufficient_evidence":
            answer = "제공된 역사 자료에서 충분한 근거를 찾지 못했습니다."
        sources = (
            [
                {
                    "document_id": str(source.get("document_id", "")),
                    "chunk_id": str(source.get("chunk_id", "")),
                    "title": str(source.get("title", "")),
                    "source_name": str(source.get("institution", "")),
                    "source_url": str(source.get("source_url", "")),
                    "score": float(source.get("retrieval_score", 0.0)),
                }
                for source in response.get("sources", [])
                if isinstance(source, dict)
            ]
            if grounded
            else []
        )
        return {
            "answer": answer,
            "sources": sources,
            "grounded": grounded,
            "status": str(response.get("status", "")),
        }

    def readiness_v1(self) -> dict[str, object]:
        retrieval = self.orchestrator.retrieval.status()
        llm = self.orchestrator.llm.readiness()
        retriever_ready = bool(retrieval.get("ready"))
        llm_ready = bool(llm.get("model_ready"))
        return {
            "ready": retriever_ready and llm_ready,
            "index_loaded": retriever_ready and int(retrieval.get("chunks", 0)) > 0,
            "retriever": retriever_ready,
            "llm": llm_ready,
            "backend": self.orchestrator.llm.backend_name,
            "llm_status": str(llm.get("status", "unknown")),
        }

    @staticmethod
    def _query(payload: dict[str, object]) -> str:
        direct = str(payload.get("user_query", "")).strip()
        if direct:
            return direct
        transition = payload.get("mode_transition")
        if type(transition) is dict:
            pending = transition.get("pending_user_question")
            if type(pending) is str and pending.strip():
                return pending.strip()
        return ""

    def stream(self, payload: dict[str, object]):
        if "conversation_mode" in payload:
            # Track-aware requests use the fully guarded common response contract.
            yield StreamEvent("completed", self.chat(payload))
            return
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
        elif mode == RuntimeMode.HACKATHON:
            provisional = [
                chunk
                for chunk in chunks
                if chunk.payload.get("usage_status") == "provisional_hackathon"
            ]
            if not self.orchestrator.retrieval.config.provisional_chunks_path.is_file():
                status = "hackathon_data_missing"
            elif errors:
                status = "hackathon_index_missing"
            elif not provisional:
                status = "hackathon_data_partial"
            elif int(
                self.orchestrator.retrieval.store.metadata().get(
                    "provisional_document_count", 0
                )
            ) < 48:
                status = "hackathon_data_partial"
            elif any(
                str(chunk.payload.get("expires_or_review_after", "")) < "2026-07-30"
                for chunk in provisional
            ):
                status = "hackathon_expired"
            else:
                status = "hackathon_index_ready"
        elif not llm.get("configured"):
            status = "remote_llm_unconfigured"
        elif llm.get("status") == "remote_unverified":
            status = "remote_unverified"
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
            "llm_status": str(llm.get("status", "unknown")),
            "llm_configured": bool(llm.get("configured")),
            "remote_server_reachable": bool(llm.get("reachable")),
            "model_ready": bool(llm.get("model_ready")),
            "missing_real_documents": mode == RuntimeMode.PRODUCTION and not real_documents,
            "missing_llm_backend": mode == RuntimeMode.PRODUCTION and not llm.get("configured"),
            "missing_index": bool(errors),
            "rights_warning": (
                "hackathon_rights_unconfirmed"
                if mode == RuntimeMode.HACKATHON and chunks
                else ""
            ),
        }


# 기존 import와 frontend 연동을 유지하는 호환 이름.
ChatApplicationService = HistoryChatService


def create_production_service(
    *,
    retrieval_config_path: Path = Path("configs/retrieval.yaml"),
    session_path: Path | None = Path(".runtime/production/sessions.json"),
    environ: Mapping[str, str] | None = None,
) -> ChatApplicationService:
    """Assemble production dependencies without building indexes or probing the LLM."""

    mode = RuntimeMode.PRODUCTION
    config = RetrievalConfig.load(retrieval_config_path)
    if RuntimeMode.parse(config.runtime_mode) != mode:
        raise ValueError("production service에는 production retrieval 설정이 필요합니다.")
    retrieval = HybridRetrievalService(config)
    index_errors = retrieval.validate_index()
    if index_errors:
        raise ProductionNotReadyError(
            "production retrieval index가 준비되지 않았습니다: "
            + "; ".join(index_errors)
        )
    llm = build_llm_from_environment(mode, environ=environ)
    sessions = SessionStore(mode, path=session_path)
    orchestrator = ConversationalRagOrchestrator(
        retrieval,
        llm,
        sessions,
        mode=mode,
        max_chunks_per_document=config.max_chunks_per_document,
    )
    return ChatApplicationService(orchestrator)
