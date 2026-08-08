"""FastAPI adapter and offline reference UI for the integrated chat demo."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from history_chatbot.chat.api_models import (
    ChatRequest,
    ChatResponse,
    FreeChatRequest,
    GenericChatRequest,
    HealthResponse,
    JourneyActionRequest,
    PieceChatRequest,
    ReadyResponse,
    SearchRequest,
    SearchResponse,
    SessionCreateRequest,
    TrackChatRequest,
    TransitionRequest,
    validate_session_id,
)

from history_chatbot.chat.demo_journey import (
    InMemoryDemoJourneyProvider, JourneyProvider, JourneyProviderError,
)
from history_chatbot.chat.service import (
    ChatApplicationService, create_development_orchestrator,
    create_hackathon_orchestrator,
    create_production_service,
)
from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.track_models import FreeChatUiState, PieceChatUiState
from history_chatbot.runtime import RuntimeMode


STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
ASSET_DIR = Path(__file__).resolve().parent / "static" / "assets"


def create_app(
    service: ChatApplicationService | None = None,
    journey_provider: JourneyProvider | None = None,
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("HTTP API 실행에는 선택 의존성 fastapi와 ASGI 서버가 필요합니다.") from error

    resolved = service or _default_service()
    journeys = journey_provider or InMemoryDemoJourneyProvider()
    app = FastAPI(title="History Pieces Reference Web Demo", debug=False)

    @app.exception_handler(JourneyProviderError)
    async def journey_error(_request: Request, error: JourneyProviderError):
        return JSONResponse(
            status_code=error.status_code,
            content=_error(error.error_code, error.message, retryable=error.retryable),
        )

    @app.exception_handler(ValueError)
    async def invalid_request(_request: Request, error: ValueError):
        return JSONResponse(
            status_code=400,
            content=_error("invalid_request", str(error), retryable=False),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error_value: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error("invalid_payload", "요청 payload 형식이 올바르지 않습니다.", retryable=False),
        )

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _error_value: Exception):
        return JSONResponse(
            status_code=500,
            content=_error("internal_error", "요청을 처리하는 중 내부 오류가 발생했습니다.", retryable=True),
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")

    @app.get("/health")
    @app.get("/api/health")
    def health():
        return {
            "status": "ok", "service": "history-pieces",
            "chat_modes": [mode.value for mode in ConversationMode],
        }

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health_v1():
        return resolved.health()

    @app.get("/ready", response_model=ReadyResponse)
    @app.get("/api/v1/ready", response_model=ReadyResponse)
    def ready_v1():
        return resolved.readiness_v1()

    @app.post("/api/v1/search", response_model=SearchResponse)
    def search_v1(payload: SearchRequest):
        return resolved.search(payload.query, top_k=payload.top_k)

    @app.post("/api/v1/chat", response_model=ChatResponse)
    def chat_v1(payload: ChatRequest):
        result = resolved.answer(
            payload.message,
            [item.model_dump() for item in payload.history],
            session_id=payload.resolved_session_id(),
            locale=payload.resolved_locale(),
            current_place_id=payload.current_place_id,
            current_piece_id=payload.current_piece_id,
            completed_place_ids=payload.completed_place_ids,
            completed_piece_ids=payload.completed_piece_ids,
        )
        return ChatResponse(answer=str(result["answer"]))

    @app.post("/api/session")
    def create_session(payload: SessionCreateRequest | None = None):
        locale = (payload or SessionCreateRequest()).resolved_locale()
        session = resolved.orchestrator.sessions.create(locale)
        return journeys.create(session.session_id, locale).to_dict()

    @app.get("/api/session/{session_id}")
    def get_session(session_id: str):
        _session_id(session_id)
        return journeys.get(session_id).to_dict()

    @app.post("/api/chat/piece")
    def piece_chat(payload: PieceChatRequest):
        return _chat(resolved, journeys, payload, ConversationMode.PIECE_CHAT)

    @app.post("/api/chat/free")
    def free_chat(payload: FreeChatRequest):
        return _chat(resolved, journeys, payload, ConversationMode.FREE_CHAT)

    @app.post("/api/chat/transition")
    def transition(payload: TransitionRequest):
        session_id = payload.resolved_session_id()
        state = journeys.get(session_id)
        from_mode = ConversationMode(payload.from_mode or state.chat_mode)
        to_value = payload.to_mode or ""
        if to_value == "game":
            to_mode = ConversationMode.PIECE_CHAT
        else:
            to_mode = ConversationMode(to_value)
        if from_mode == ConversationMode.PIECE_CHAT and to_mode == ConversationMode.FREE_CHAT:
            action = "OPEN_FREE_CHAT"
        elif from_mode == ConversationMode.FREE_CHAT and to_mode == ConversationMode.PIECE_CHAT:
            action = "RETURN_TO_GAME"
        else:
            raise JourneyProviderError("invalid_transition", "지원하지 않는 mode transition입니다.", status_code=409)
        updated = journeys.apply_action(session_id, action, payload.model_dump())
        return {
            "request_state": "success", "action_code": action,
            "game_state_mutation": False, "transition": payload.mode_transition,
            "session": updated.to_dict(),
        }

    @app.post("/api/journey/action")
    def journey_action(payload: JourneyActionRequest):
        session_id = payload.resolved_session_id()
        action_code = payload.action_code or ""
        if not action_code:
            raise ValueError("action_code가 필요합니다.")
        before = journeys.get(session_id).to_dict()
        state = journeys.apply_action(session_id, action_code, payload.model_dump())
        changed = (
            before["current_piece_id"] != state.current_piece_id
            or before["completed_piece_ids"] != tuple(state.completed_piece_ids)
        )
        return {
            "request_state": "success", "action_code": action_code,
            "game_state_mutation": changed, "session": state.to_dict(),
        }

    # Backward-compatible generic API.
    @app.post("/api/chat")
    def chat(payload: GenericChatRequest):
        return resolved.chat(payload.service_payload())

    @app.post("/api/chat/stream")
    def stream(payload: GenericChatRequest):
        def events():
            try:
                value = payload.service_payload()
                for event in resolved.stream(value):
                    yield f"event: {event.event}\ndata: {json.dumps(asdict(event)['data'], ensure_ascii=False)}\n\n"
            except ValueError as error:
                yield f"event: error\ndata: {json.dumps(_error('invalid_request', str(error), retryable=False), ensure_ascii=False)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.delete("/api/sessions/{session_id}")
    def reset(session_id: str):
        _session_id(session_id)
        return resolved.reset(session_id)

    @app.get("/api/readiness")
    def readiness():
        return resolved.readiness()

    return app


def _default_service() -> ChatApplicationService:
    mode = RuntimeMode.parse(os.getenv("APP_MODE", "development"))
    if mode == RuntimeMode.HACKATHON:
        return ChatApplicationService(create_hackathon_orchestrator())
    if mode in {RuntimeMode.DEVELOPMENT, RuntimeMode.TEST}:
        return ChatApplicationService(create_development_orchestrator())
    return create_production_service()


def _chat(
    service: ChatApplicationService, journeys: JourneyProvider,
    payload: TrackChatRequest, mode: ConversationMode,
) -> dict[str, object]:
    session_id = payload.resolved_session_id()
    state = journeys.get(session_id)
    message = payload.resolved_message()
    ui_state = payload.ui_state
    if ui_state is not None:
        (PieceChatUiState if mode == ConversationMode.PIECE_CHAT else FreeChatUiState)(str(ui_state))
    response = service.chat({
        "user_query": message, "session_id": session_id,
        "locale": payload.resolved_locale(state.locale),
        "conversation_mode": mode.value, "screen_type": mode.value,
        "current_place_id": state.current_place_id,
        "current_piece_id": state.current_piece_id,
        "visited_piece_ids": tuple(state.completed_piece_ids),
        "current_journey_step": state.current_journey_step,
        "available_capabilities": state.available_capabilities,
        "return_target": payload.return_target,
    })
    state.temporary_context_state = list(dict.fromkeys(
        state.temporary_context_state + list(response.get("context_state", ()))
    ))
    response["situation_id"] = response["primary_situation_id"]
    response["piece_ui_state" if mode == ConversationMode.PIECE_CHAT else "free_ui_state"] = response["ui_state"]
    return response


def _session_id(value: str) -> str:
    return validate_session_id(value)


def _error(error_code: str, message: str, *, retryable: bool) -> dict[str, object]:
    return {
        "error_code": error_code, "message": message,
        "request_state": "error", "retryable": retryable, "details": {},
    }
