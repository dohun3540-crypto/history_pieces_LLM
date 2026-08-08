# Chat API

대화 모드별 상세 계약과 reference UI state는 [CHAT_MODES.md](CHAT_MODES.md)를 따른다.
`conversation_mode`는 `piece_chat` 또는 `free_chat`만 허용하며, 지정된 `screen_type`은
같은 값을 가져야 한다. 선택 입력으로 `current_place_id`, `current_piece_id`,
`visited_piece_ids`(실제 완료 조각), `current_journey_step`, `piece_follow_up_count`,
`return_target`, `available_capabilities`, `storage_capability`, `user_consent`를 받는다.

응답은 기존 필드와 함께 `chat_mode`, `next_action_code`, `required_context`,
`missing_context`, `capability_supported`, `fallback_used`, `current_place_id`,
`current_piece_id`, `completed_piece_ids`, `game_state_mutation`, `mode_transition`,
`rag_used`, `storage_requested`, `storage_permitted`, `request_state`, `ui_state`,
`suggested_questions`를 반환한다. 실제 provider가 없는 action은 실행 완료가 아니라
capability 미지원 fallback이다.

Persona metadata는 하위 호환 필드에 추가해 `output_domain`, `speech_level`,
`persona_id`, `language`, `culture`, `conversation_stage`, `source_sufficiency`,
`translation_status`를 반환한다. `character_dialogue`는 한국어 반말,
`system_ui`는 기능 중심 존댓말이며 최종 기준은
[GIROKSAE_CHARACTER_PRINCIPLES_V11.md](GIROKSAE_CHARACTER_PRINCIPLES_V11.md)다.

## Reference web demo endpoints

- `GET /health`: 서비스 상태와 허용 chat mode
- `POST /api/session`: ephemeral demo journey 생성
- `GET /api/session/{session_id}`: 현재 demo 상태 조회
- `POST /api/chat/piece`: provider 문맥으로 piece-chat 호출
- `POST /api/chat/free`: provider 문맥으로 free-chat 호출
- `POST /api/chat/transition`: `piece_chat → free_chat`, `free_chat → game` 전환
- `POST /api/journey/action`: 명시적인 demo 여정 action 적용
- `GET /`: offline reference UI
- `GET /static/styles.css`, `GET /static/app.js`: package 정적 자산

```json
{"session_id":"<SESSION_ID>","user_message":"조금 피곤해요.","ui_state":"awaiting_reflection"}
```

```json
{"session_id":"<SESSION_ID>","action_code":"GO_NEXT_PIECE"}
```

Client가 보낸 place/piece/completed 목록은 chat 근거로 신뢰하지 않고 demo provider의
현재 상태를 service DTO로 전달한다. Chat 응답만으로는 상태를 변경하지 않는다.
오류는 `error_code`, `message`, `request_state`, `retryable`, `details`를 반환하며,
근거 부족은 HTTP 오류가 아닌 정상 `insufficient_evidence` 응답이다. 자세한 실행과 UI
제약은 [WEB_UI.md](WEB_UI.md)를 참고한다.

핵심 `ChatApplicationService`는 웹 프레임워크에 의존하지 않는다. FastAPI와
uvicorn은 선택 의존성이다.

```powershell
python -m pip install -e ".[api]"
uvicorn history_chatbot.chat.api:create_app --factory
```

Endpoint:

- `POST /api/v1/search`: LLM 호출 없이 hybrid 검색 결과 반환
- `POST /api/v1/chat`: 단순 `{answer}` 응답 계약의 근거 기반 채팅
- `GET /ready`, `GET /api/v1/ready`: retriever와 LLM readiness 반환
- `POST /api/chat`: JSON 질문을 처리하고 최종 답변과 출처를 반환
- `POST /api/chat/stream`: SSE `start`, `token`/`delta` 이벤트 뒤 `completed` 이벤트 반환
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

### `/api/v1/chat` 지속 대화

기존의 일회성 `{message, history}` 요청은 그대로 지원한다. 서버가 관리하는 지속
대화가 필요하면 먼저 `POST /api/session`으로 `session_id`를 발급받고 같은 ID를
후속 `/api/v1/chat` 요청에 전달한다. 응답은 기존 브라우저 계약을 유지하기 위해
계속 `{answer}` 하나만 반환한다.

```json
{
  "message": "그 건물은 당시 어떤 역할을 했나요?",
  "session_id": "<32자리 SESSION_ID>",
  "locale": "ko",
  "current_place_id": "mokpo-station-1932",
  "current_piece_id": "station-piece-1",
  "completed_place_ids": ["mokpo-music-hall"],
  "completed_piece_ids": ["music-piece-1"]
}
```

장소·조각 ID와 완료 목록은 대화 및 관광 여정 문맥일 뿐 역사적 사실의 근거가
아니다. 역사 설명은 항상 검색된 문서 chunk를 근거로 생성하며, 알 수 없는
`session_id`는 새 세션으로 암묵적으로 바꾸지 않고 `400 invalid_request`로 거절한다.
