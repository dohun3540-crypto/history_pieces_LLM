# 기록새 상황 대화 연결

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

서비스 기본값은 자연스러운 한국어 존댓말(`polite`)이다. 기존 UI 설정의 기본 언어가
한국어이고 기존 챗봇 답변도 존댓말인 점, 조각 대화와 자유대화 사이에서 한 세션의
말투가 섞이지 않아야 한다는 점을 근거로 선택했다. 원본 `response_draft`의 반말은
수정하지 않고 기준 데이터로 보존한다. 향후 세션 설정으로 부드러운 반말 모드를
추가할 수 있지만 현재 런타임 응답은 존댓말로 일관된다.

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
