# 기록새 상황 대화 연결

> **문서 역할:** 이 문서는 상황·seed·RAG·safety/action 계약을 설명한다. Persona와
> 런타임 말투에 관한 기존 내용은 superseded이며 최종 기준이 아니다. 정체성·말투·출력
> 영역·생성 prompt의 유일한 기준은
> [GIROKSAE_CHARACTER_PRINCIPLES_V11.md](GIROKSAE_CHARACTER_PRINCIPLES_V11.md)다.

## 기준 데이터

사람이 작성한 원본은 `docs/giroksae_situation_seed.md`이며 실제 구성은 17개 상황과
54개 사례다. 초안 하단의 55개 표기는 집계 오류였으므로 원본 사례를 추가하거나
본문을 고치지 않고 54개로 정정했다. `scripts/export_giroksae_seed.py`는 Markdown 표를
`configs/giroksae_situations.json`으로 기계적으로 변환한다.

JSON의 `source_fields`는 화면, 사용자 입력, 대응 목표, 응답 초안, 다음 행동, 태그의
Markdown 원문을 보존한다. 런타임 필드인 `requires_rag`, `requires_clarification`,
`response_length_mode` 등은 별도로 파생한다. loader는 17/54 개수, ID 중복,
`human_authored_seed`/`reviewed_seed`, 원본 문구 일치를 검증한다. 생성형 증강 데이터는
이 loader로 읽을 수 없다.

## 말투 결정

이 절의 과거 “기본 존댓말” 결정은 superseded됐다. 현재 runtime은 최종 승인 문서에
따라 `character_dialogue=banmal`, `system_ui=polite_ui`,
`historical_docent=formal_docent`, `journey_film_caption=neutral_caption`으로 분리한다.
원본 `response_draft`와 `response_draft_original`은 말투 source가 아니라 provenance
보존 데이터이며 수정하지 않는다.

`zh-CN`은 locale로 허용하며 `configs/giroksae_zh_cn_terms.json`에 검수된 용어 사전의
연결 지점만 마련했다. 전체 중국어 번역은 생성하지 않았다.

## 분류와 RAG 정책

분류 입력은 발화, 대화 모드, 화면, locale, 현재 조각/장소, 최근 턴, 완료 조각,
현재 세션의 스타일 설정을 포함한다. 결과에는 primary/secondary 상황, confidence,
의도 코드, RAG·명확화 여부, 길이 모드, 개인화 후보, 다음 행동과 내부 reason code가
포함된다. reason code는 고정 코드이며 chain-of-thought를 저장하지 않는다.

- 사실·인물·도시생활·근거 검증·교차문화 비교는 검색한다.
- 인사, 감상 공감, 낮은 참여, 피로, 불만 대응은 검색과 LLM을 호출하지 않는다.
- 건축, 여정 관계, 비교, 부정 감정, 개인 대화, 스타일 요청은 사실 요청이 있을 때만 검색한다.
- 출처 요청은 `RESPONSE_STYLE_REQUEST`를 primary로, `EVIDENCE_AND_CORRECTION`을 secondary로 분류하고 검색한다.
- 욕설·강한 불만은 다른 감정 분류보다 우선하며 맞대응하거나 훈계하지 않는다.
- 지시 대상이나 비교 지역이 불명확하면 검색 전에 명확화 질문을 한다.
- 검색 결과가 없으면 기존과 동일하게 LLM 호출을 차단한다.

`piece_chat`은 현재 조각/장소 및 실제 완료 조각 ID를 전달한다. 여정 질문의 prompt에는
완료 ID만 게임 메타데이터로 넣고 역사 근거와 구분하도록 제한한다. 낮은 참여나 강한
불만에는 대화를 늘리지 않고 종료·건너뛰기 선택을 제공한다. `free_chat`은 전체 허용
corpus에서 사실 질문, 출처, 재검증과 비교를 처리한다. production과
`provisional_hackathon` 경계는 기존 retrieval 계층을 그대로 통과하므로 변경되지 않는다.

## 개인화

태그는 `session_observation`, `preference_candidate`, `journey_interest`로 분류한다.
각 관찰에는 confidence, evidence turn ID, 관찰 시각과 원문 발화를 별도로 둔다. 한 번의
스타일 요청은 현재 응답에 즉시 반영하지만 장기 profile로 확정하지 않는다. 반복 관찰이
명시된 경우에만 profile 후보가 될 수 있다. 피로·슬픔·불만·낮은 참여는 세션 관찰이며
영구 성향이 아니다. 허용 목록 밖 태그와 민감정보 추론 태그는 저장하지 않는다.

## 모델 backend

기본 `hashing-v1`은 유지한다. 선택형 `SentenceTransformerEncoder`는
`intfloat/multilingual-e5-small`, query/passage prefix, 정규화, CPU를 지원한다.
모델은 `local_files_only=True`로 열기 때문에 캐시에 없으면 자동 다운로드나 hashing
fallback 없이 실패한다. 인덱스 metadata에는 모델, revision, dimension, normalization,
두 prefix를 기록하며 불일치하면 검색을 거부한다.

실제 LLM은 기존 `RemoteLLMBackend`의 OpenAI-compatible 계약을 사용한다. base URL,
모델, API 키, timeout, 제한 retry, streaming, 빈 응답, malformed 응답 및 서버 장애를
지원하며 fake transport 테스트로 외부 호출 없이 검증한다. 응답에는 내부 prompt,
API 키, chain-of-thought를 포함하지 않는다.

## API 입력과 결과

기존 `user_query`, `session_id`, `locale`, `top_k` 외에 `conversation_mode`,
`screen_type`, `current_piece_id`, `current_place_id`, `visited_piece_ids`,
`existing_style_preferences`를 받을 수 있다. 결과에는 기존 필드를 유지하면서 상황,
다음 행동, follow-up, 개인화 후보, citation/evidence, grounded, confidence, refusal,
길이 모드, 검색 ID, model/embedding backend, latency와 warnings를 추가한다.

## V03 안전·도움 흐름

V03은 기존 17개 상황과 54개 reviewed human-authored seed를 변경하지 않고
`TECHNICAL_HELP`, `NAVIGATION_HELP`, `SAFETY_ACCESSIBILITY` 및 9개
`human_proposed_seed`를 별도 bundle로 추가한다. 제안 seed는 `review_pending`이며,
loader가 런타임에서 두 bundle을 합쳐 20개 상황·63개 사례로 제공한다. 구형 bundle만
읽어야 할 때는 `SituationSeedLoader(additions_path=None)`을 사용한다.

기존 `response_draft`는 `response_draft_original`로 그대로 노출하고 제거하지 않는다.
신규 bundle의 기존 `response_template`은 제안 당시 문구로 보존하며 최종 persona
source로 사용하지 않는다. 서비스는 원문을 직접 출력하지 않고 output domain별
canonical persona와 response policy를 사용한다.

세 신규 상황은 `piece_chat`과 `free_chat` 모두에서 허용되며 역사 RAG와 citation을
사용하지 않는다. action code, required context, missing context, provider capability,
fallback 사용 여부를 구조적으로 반환한다. 현재 실제 UI·지도·검증 시설 provider가
연결되지 않았으므로 구체적인 버튼·아이콘, 다음 조각, 거리·시간·방향, 접근성 시설을
만들지 않고 일반 점검, 공식 안내, 현장 직원 확인으로 fallback한다. 저장 완료 표현은
저장 capability와 사용자 동의가 모두 확인될 때만 허용한다.

개인화 태그는 관심·선호 후보만 유지하고, 피로·더위·불만·기술·길찾기·접근성 요청은
현재 세션의 `context_state`로 분리한다. `policy_flags`는 처리 규칙이며 사용자 프로필로
저장하지 않는다. 역사 corpus, production/provisional lane, hashing/E5 인덱스와 검색
설정에는 V03 반영으로 인한 변경이 없다.
