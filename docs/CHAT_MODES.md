# 기록새 대화 모드 계약

> 이 문서는 piece/free state와 전환 계약만 담당한다. Persona와 말투는
> [GIROKSAE_CHARACTER_PRINCIPLES_V11.md](GIROKSAE_CHARACTER_PRINCIPLES_V11.md)를
> 유일한 기준으로 사용한다.

기록새 backend는 같은 persona와 상황 classifier를 사용하면서 `piece_chat`과
`free_chat`을 별도 정책 트랙으로 처리한다. 이 문서는 프런트엔드 구현 설명이 아니라
서비스 DTO와 reference UI state 계약이다.

## 공통 문맥과 불변 조건

`SharedSessionContext`는 session/locale, 현재 장소·조각, 실제 완료 조각,
현재 여정 단계, 임시 길이·피로·불만 상태, persona와 사용 가능한 capability를 담는다.
완료 조각은 중복과 빈 ID를 거부하며, 관계 설명에 전달되는 목록은
`completed_piece_ids`와 교집합으로 제한한다.

모든 응답은 `chat_mode`, `game_state_mutation`, `storage_permitted`, `rag_used`,
`request_state`, `ui_state`를 포함한다. `game_state_mutation` 기본값은 항상 false다.
게임 진행 변경은 backend 대화 응답이 아니라 실제 게임 서비스가 승인한 별도 action의
책임이다. 저장은 `storage_capability`, `user_consent`, `SAVE_SHORT_REFLECTION`
capability가 모두 있을 때만 허용한다.

## piece_chat

조각 체험 안의 짧은 대화다. 감정·피로·불만·낮은 참여에는 RAG를 사용하지 않는다.
낮은 참여에는 `SKIP_REFLECTION`, 피로에는 `PAUSE_JOURNEY`, 짧은 답변 요청에는
`CONTINUE_WITH_SHORT_MODE`를 제안한다. 후속 질문은 `piece_follow_up_count=0`일 때만
한 번 제공한다.

짧은 사실 질문은 현재 화면에서 RAG로 답할 수 있다. 출처·자세한 배경·비교·관계나
복잡한 인물 질문은 검색을 실행하기 전에 `OPEN_FREE_CHAT` transition을 제안한다.
실제 free-chat UI capability가 없으면 질문과 게임 문맥을 보존한 채 fallback 상태를
반환할 뿐, 창이 열렸다고 표현하지 않는다.

reference UI state는 `hidden`, `showing_prompt`, `awaiting_reflection`, `responding`,
`ready_for_next_piece`, `skipped`, `paused`다. 실제 버튼이나 화면은 이 저장소에서
구현하지 않는다.

## free_chat

게임 진행과 독립적인 대화 트랙이다. 역사 사실은 RAG와 citation을 요구하고, 인사와
가벼운 대화는 검색하지 않는다. 검색 근거가 없으면 `insufficient_evidence`와 빈
citations를 반환한다. 기술·길·안전·접근성 요청에는 V03 no-RAG fallback을 그대로
적용한다. 응답에는 추천 질문과 `RETURN_TO_GAME` reference action 계약을 제공한다.

reference UI state는 `closed`, `opening`, `active`, `loading`, `showing_citations`,
`insufficient_evidence`, `error`, `returning_to_game`이다. free-chat 사용이나 종료는
미션 완료, 다음 조각 이동, 감상 저장을 수행하지 않는다.

## 모드 전환

`ModeTransition`은 transition ID, 출발·도착 모드, 이유, 원 질문, 복귀 target,
source session, 생성 시각, 장소·현재 조각·완료 조각을 보존한다.
`preserve_game_state`는 항상 true다. `open_free_chat`은 원 질문을 pending 상태로 넘기고,
`return_to_game`은 기존 place/piece와 return target을 유지한다. 전환 provider가 실패해도
이 DTO를 재사용해 질문과 위치를 복구할 수 있다.

## 범위 밖 기능

현재 저장소에는 실제 프런트엔드, 게임 상태 provider, 저장 provider, 지도·위치 provider,
검증된 시설 provider가 없다. 따라서 아이콘·버튼 표시, 창 열기·닫기, 미션 완료,
경로 계산, 접근성 시설 확인, 영구 저장은 구현됐다고 간주하지 않는다. 이 문서의 UI와
action은 연결을 위한 backend/reference 계약이다. 역사 corpus, 20개 상황·63개 seed,
retrieval 설정, hashing/E5 인덱스, production/provisional lane은 변경하지 않는다.
