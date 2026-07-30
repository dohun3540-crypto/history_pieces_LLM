# Remote LLM Backend

현재 실제 Llama 모델이나 외부 추론 서버는 연결되어 있지 않다. 이 구현은
MockLLM과 같은 `ChatCompletionBackend` 계약을 사용하는 원격 연결 경계이며,
테스트는 가짜 transport만 사용한다.

지원 형식:

- OpenAI-compatible: `POST /v1/chat/completions`
- 프로젝트 전용 FastAPI: `POST /generate`, `POST /generate/stream`
- 경량 상태 점검: `GET /health`, `GET /ready`

환경변수의 `LLM_BACKEND=remote`, `LLM_BASE_URL`, `LLM_ALLOWED_HOSTS`,
`LLM_MODEL`을 설정하면 코드
변경 없이 backend를 만들 수 있다. 실제 키는 Git에서 제외된 `.env` 또는 프로세스
환경변수에만 둔다. URL에는 키를 넣지 않으며 오류 메시지와 객체 repr에도 키를
표시하지 않는다.

production에서는 MockLLM을 거부한다. URL과 모델이 없거나 허용 host 정책에 맞지
않으면 시작 또는 readiness가 실패한다. 연결 오류와 5xx만 제한적으로 재시도하고,
인증·404·429·잘못된 요청은 재시도하지 않는다.

오류 코드는 `connection_error`, `timeout`, `authentication_error`,
`rate_limited`, `model_not_found`, `invalid_response`, `server_not_ready`,
`context_length_exceeded`, `generation_failed`로 구조화된다. 실패 시 역사 답변을
대체 생성하지 않는다.
