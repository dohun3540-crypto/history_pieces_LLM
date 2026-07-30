# LLM Server Contract

## 공통 요청

- `system_prompt`
- `user_prompt`
- `messages`: role/content 배열
- `temperature`, `top_p`, `max_new_tokens`
- `stop_sequences`
- `stream`
- `request_id`
- `timeout`은 클라이언트 deadline에 사용

요청별 값은 temperature 0~2, top_p 0 초과 1 이하, 출력 1~8192 token,
timeout 0.1~600초 범위로 검증한다.

## 공통 완료 응답

- `generated_text`
- `finish_reason`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `model`, `model_revision`
- `request_id`
- `latency_ms`

필수 필드나 타입이 다르면 `invalid_response`로 처리한다. HTML이나 script가
포함되어도 실행하거나 지시로 해석하지 않고 일반 문자열로 취급한다.

## 스트리밍

이벤트 순서는 `start` → `token` 또는 `delta` → `completed`다. 오류 시 `error`로
종료한다. `completed`에는 전체 답변, finish reason, token usage, model 정보가
포함된다. 완료 이벤트 없이 연결이 끝나면 retryable `generation_failed`다.

OpenAI-compatible 형식은 SSE `data:`와 `[DONE]` 또는 finish_reason을 해석한다.
프로젝트 전용 형식은 JSONL `start`, `token`/`delta`, `completed`, `error`
이벤트를 사용한다.
