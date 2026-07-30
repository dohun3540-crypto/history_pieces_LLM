"""선택적 FastAPI 어댑터. 핵심 서비스는 FastAPI에 의존하지 않는다."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from history_chatbot.chat.service import (
    ChatApplicationService,
    create_development_orchestrator,
    create_hackathon_orchestrator,
)
from history_chatbot.runtime import RuntimeMode


def create_app(service: ChatApplicationService | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as error:  # pragma: no cover - 선택 의존성
        raise RuntimeError(
            "HTTP API 실행에는 선택 의존성 fastapi와 ASGI 서버가 필요합니다."
        ) from error

    if service is not None:
        resolved = service
    else:
        mode = RuntimeMode.parse(os.getenv("APP_MODE", "development"))
        if mode == RuntimeMode.HACKATHON:
            resolved = ChatApplicationService(create_hackathon_orchestrator())
        elif mode in {RuntimeMode.DEVELOPMENT, RuntimeMode.TEST}:
            resolved = ChatApplicationService(create_development_orchestrator())
        else:
            raise RuntimeError(
                "production API는 실제 원격 LLM과 approved_for_rag 인덱스를 명시적으로 주입해야 합니다."
            )
    app = FastAPI(title="Mokpo History Development RAG")

    @app.post("/api/chat")
    def chat(payload: dict[str, object]):
        try:
            return resolved.chat(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/chat/stream")
    def stream(payload: dict[str, object]):
        def events():
            try:
                for event in resolved.stream(payload):
                    yield (
                        f"event: {event.event}\n"
                        f"data: {json.dumps(asdict(event)['data'], ensure_ascii=False)}\n\n"
                    )
            except ValueError as error:
                yield f"event: error\ndata: {json.dumps({'message': str(error)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.delete("/api/sessions/{session_id}")
    def reset(session_id: str):
        return resolved.reset(session_id)

    @app.get("/api/health")
    def health():
        return resolved.health()

    @app.get("/api/readiness")
    def readiness():
        return resolved.readiness()

    return app
