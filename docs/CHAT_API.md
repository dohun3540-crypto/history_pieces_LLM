# Chat API

핵심 `ChatApplicationService`는 웹 프레임워크에 의존하지 않는다. FastAPI와
uvicorn은 선택 의존성이다.

```powershell
python -m pip install -e ".[api]"
uvicorn history_chatbot.chat.api:create_app --factory
```

Endpoint:

- `POST /api/chat`: JSON 질문을 처리하고 최종 답변과 출처를 반환
- `POST /api/chat/stream`: SSE `token` 이벤트 뒤 `complete` 이벤트 반환
- `DELETE /api/sessions/{session_id}`: 세션 초기화
- `GET /api/health`: 프로세스 생존 상태
- `GET /api/readiness`: 실행 모드와 인덱스·자료·LLM 준비 상태

요청 예:

```json
{
  "user_query": "붉은 등대 전시관을 알려줘",
  "session_id": null,
  "locale": "ko",
  "top_k": 3
}
```

출처에는 `source_id`, `document_id`, `title`, `institution`, `source_url`,
`chunk_id`, `excerpt`, `retrieval_score`, `license_status`, `is_fixture`가
포함된다. development 응답의 fixture 출처는 항상 `is_fixture=true`다.

readiness는 `development_ready`, `production_not_ready`를 주 상태로 구분하고
`missing_real_documents`, `missing_llm_backend`, `missing_index`를 별도
불리언 필드로 제공한다.
