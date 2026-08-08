# History Pieces reference web UI

## 통합 시각 디자인

웹 데모는 `docs/design/GAME_UI_UX_DESIGN_SPEC.md`의 종이 기록물, 갈색·금색,
어두운 촬영 화면 분위기를 참고한다. 제공된 PNG는 완성된 세로 화면 시안이므로
그 안의 버튼이나 문구를 실제 기능으로 취급하지 않는다. 대신 가독성 오버레이 아래
여정 분위기 배경으로만 사용한다.

배경 매핑은 역사적 장소나 사실을 뜻하지 않는 순수한 데모 상태 매핑이다.

- 로딩/초기 상태: `background_06.png`
- 조각 1: `background_02.png`
- 조각 2: `background_07.png`
- 조각 3: `background_01.png`
- 감상 건너뛰기: `background_03.png`
- 일시정지: `background_04.png`
- 데모 여정 완료: `background_05.png`

`giroksae_character.png`도 투명 캐릭터 단독 컷이 아니라 배경과 문구가 포함된
승인 시안이다. 원본을 수정하거나 늘리지 않고 `object-fit: cover`로 crop해 piece_chat,
free_chat, 플로팅 진입점에서 같은 기록새 정체성을 보여준다.

정적 자산은 `/assets/...` 경로로 FastAPI가 제공한다. 실제 게임 장면, 촬영 기능,
지도, 위치, 시설 또는 저장 provider가 연결된 것으로 해석하면 안 된다.

이 UI는 `piece_chat`과 `free_chat` backend 계약을 브라우저에서 확인하기 위한 최소
통합 데모다. 상용 디자인이나 실제 게임 프런트엔드가 아니며, 외부 CDN·Node 빌드·외부
이미지 없이 FastAPI가 정적 HTML/CSS/vanilla JavaScript를 제공한다.

## 설치와 실행

Windows PowerShell에서 저장소 루트를 연 뒤 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[api]"
$env:APP_MODE = "development"
python -m uvicorn history_chatbot.chat.api:create_app --factory --host 127.0.0.1 --port 8000
```

실제 임시 corpus를 사용하는 비상업적 해커톤 smoke에서는 로컬 처리 파일을 먼저
준비하고 다음처럼 명시한다. `mock`은 browser/RAG 연결 확인용이며 실제 Llama 추론
성공으로 간주하지 않는다.

```powershell
$env:APP_MODE = "hackathon"
$env:HISTORY_LLM_BACKEND = "mock"
python -m uvicorn history_chatbot.chat.api:create_app --factory --host 127.0.0.1 --port 8000
```

실제 remote Llama 연결은 `docs/VLLM_E2E_SMOKE.md`의 환경변수와 smoke 절차를
사용한다.

브라우저 주소는 `http://127.0.0.1:8000/`이다. API 문서는
`http://127.0.0.1:8000/docs`에서 확인한다. 기본 bind는 로컬 loopback이며 debug
mode를 사용하지 않는다.

## 데모 흐름

페이지는 ephemeral session을 만들고 역사 사실이 아닌 중립적인 `조각 1`~`조각 3`
label을 사용한다.

1. Piece-chat에 `조금 피곤해요`를 보내면 no-RAG pause 제안을 확인한다.
2. `이 장소의 역사를 출처와 함께 자세히 알려주세요`를 보내면 구조화된
   `OPEN_FREE_CHAT` transition으로 panel이 열린다.
3. 원 질문, 현재 장소·조각, 완료 조각과 복귀 target이 보존되고 질문이 복원된다.
4. Free-chat 역사 질문은 RAG를 사용하며 citation이 있을 때만 출처 버튼이 나타난다.
   근거가 없으면 citation은 비고 `insufficient_evidence`가 표시된다.
5. 닫기를 누르면 `RETURN_TO_GAME` 후 같은 조각과 완료 목록으로 돌아온다.
6. 다음 조각은 대화 응답이 아닌 명시적인 `GO_NEXT_PIECE` action으로만 이동한다.

Refresh 시 `sessionStorage`의 session ID로 같은 server process의 상태를 다시 읽는다.
서버를 재시작하면 in-memory demo session은 사라진다.

## UI와 접근성

- 구조화된 piece/free/request state를 사용하며 응답 문자열을 action 판단에 쓰지 않는다.
- 요청 중 입력과 버튼을 비활성화하고 같은 메시지의 중복 요청을 막는다.
- 사용자 메시지와 citation은 `textContent`와 DOM API로만 렌더링한다.
- 실제 URL이 있을 때만 citation link를 만들고 긴 제목은 줄바꿈한다.
- Dialog semantics, Escape 닫기, focus 순환과 원래 focus 복귀를 지원한다.
- 연결·로딩·오류·근거 부족 상태는 `aria-live`로 알린다.
- 키보드 focus, 모바일 단일 열, reduced-motion 설정을 지원한다.

## 기록새 이미지 계약

현재 package에는 승인 시안 `chat/static/assets/giroksae/giroksae_character.png`가
포함되어 `/assets/giroksae/giroksae_character.png`로 제공된다. 투명 단독 컷이
아니므로 원본을 수정하거나 왜곡하지 않고 기존 crop과 대체 텍스트 계약을 유지한다.

## 한계와 provider 경계

`InMemoryDemoJourneyProvider`는 `JourneyProvider` protocol의 reference 구현이다.
현재 실제 게임 DB, 계정 간 영속 상태, 감상 저장, 지도·위치·검증 시설 provider,
사진 overlay, 기록새 이미지와 production LLM은 연결하지 않았다.
`SAVE_SHORT_REFLECTION`은 capability unavailable을 반환한다. V03 기술·길·접근성
문의는 기존 안전 fallback을 사용한다. 기록새 말풍선은
`character_dialogue=banmal`, 기능·오류 안내는 `system_ui=polite_ui` metadata를
따르며 최종 기준은 [기록새 최종 캐릭터 원칙](GIROKSAE_CHARACTER_PRINCIPLES_V11.md)이다.
