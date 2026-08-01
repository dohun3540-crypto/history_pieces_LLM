# 기록새 상황별 대화 데이터셋 v0.3 — 프로젝트 반영용

> **문서 역할:** 이 문서는 20개 상황, 63개 seed, safety fallback, action/context
> 계약만 담당한다. 이 문서의 “기본 존댓말”, persona, runtime template 관련 규칙은
> superseded됐으며 런타임 기준으로 사용하지 않는다. 최종 persona와 말투의 유일한
> 기준은 [GIROKSAE_CHARACTER_PRINCIPLES_V11.md](GIROKSAE_CHARACTER_PRINCIPLES_V11.md)다.

## 0. 반영 평가

이 데이터셋은 `history_pieces_LLM`의 상황 분류기, 응답 정책, RAG 라우팅, 개인화 후보 태그, 게임 액션 계약에 반영할 수 있다.

다만 v0.2 초안에는 다음 문제가 있어 그대로 반영하지 않고 v0.3에서 보완했다.

1. 기존 실제 seed는 54개이므로 신규 9개를 더하면 총 63개다. `62개` 표기를 `63개`로 정정했다.
2. 길 안내·접근성·기술 안내에 실제 UI·지도·시설 상태가 확인되지 않은 단정적 문장이 있었다.
3. `기록에 반영할게` 같은 표현은 저장 기능과 동의 상태가 확인된 경우에만 사용해야 한다.
4. `execute_rag`만으로는 외부 비교 자료, 실시간 위치, 시설 접근성 자료를 구분할 수 없다.
5. 사람 작성 원본과 프로젝트용 파생 정책 필드를 분리해야 한다.
6. 당시 원본 반말 초안과 과거 프로젝트의 기본 존댓말 persona가 충돌해 런타임 말투 변환이 필요하다고 판단했다. 이 판단은 최종 캐릭터 정책에 의해 superseded됐다.

### 프로젝트 반영 판정

- **상황 분류/라우팅 기준:** 반영 가능
- **human-authored seed:** 반영 가능
- **최종 답변 문구:** 템플릿으로 반영 가능하나 사실 슬롯 검증 필수
- **기술·길 안내·접근성 답변:** 실제 앱 상태·지도·시설 데이터가 있을 때만 실행
- **장기 개인화:** 즉시 확정 금지, 후보 관찰값으로만 반영
- **중국어:** locale 및 용어 연결 구조만 반영, 번역 데이터는 별도 검수 후 추가

---

## 1. 데이터셋 규모

- 상황 분류: **20개**
- 사용자 발화 예시: **63개**
- 기존 human-authored seed: **54개**
- 신규 제안 seed: **9개**
- 언어: 한국어
- 기본 서비스 말투: 존댓말
- 원본 초안 말투: 보존
- 중국어 병렬 데이터: 미작성

---

## 2. 권장 스키마

```json
{
  "example_id": "ARCH_02",
  "situation_id": "INTEREST_ARCHITECTURE",
  "screen_type": ["piece_chat", "free_chat"],
  "user_input": "건물 구조가 왜 이렇게 생겼어요?",
  "user_intent": "건축 구조의 역사적 이유 확인",
  "response_goal": "검증된 건립 시기와 용도에 근거해 설명",
  "response_draft_original": "건물의 구조는 건립 시기와 사용 목적을 함께 봐야 정확히 설명할 수 있어.",
  "response_template": "확인된 자료에 따르면 이 건물은 {verified_period}에 {verified_purpose}로 사용되었고, 그 영향이 {verified_feature}에 나타납니다.",
  "next_action_code": "ANSWER_WITH_CITATIONS",
  "personalization_tags": ["interest_architecture"],
  "context_state": ["prefers_detail_candidate"],
  "policy_flags": ["requires_rag", "requires_evidence"],
  "required_context": ["current_place_id"],
  "fallback_behavior": "INSUFFICIENT_EVIDENCE",
  "locale": "ko-KR",
  "source_type": "human_authored_seed",
  "review_status": "reviewed_seed"
}
```

### 필드 역할

- `personalization_tags`: 반복 시 장기 관심 후보가 될 수 있는 관심·선호
- `context_state`: 현재 발화·세션에서만 유효한 감정·피로·불만·임시 스타일
- `policy_flags`: RAG, 명확화, 출처, 앱 상태, 위치, 안전 처리 규칙
- `required_context`: 답변 전 반드시 필요한 게임·지도·시설 데이터
- `next_action_code`: UI와 백엔드가 실행할 구조화된 동작
- `response_draft_original`: 사람이 작성한 원문 보존
- `response_template`: 프로젝트 런타임에서 사용할 검증 가능한 답변 틀

---

## 3. 공통 정책

### 3.1 원본과 파생 데이터

- 기존 54개 문구와 ID는 변경하지 않는다.
- 신규 9개는 `human_proposed_seed` 또는 별도 승인 후 `human_authored_seed`로 구분한다.
- `requires_rag`, `required_context`, `next_action_code`는 파생 필드로 저장한다.
- 원본 문구와 서비스 출력 문구를 동일 필드에 덮어쓰지 않는다.

### 3.2 RAG 정책

- 인사·감상·피로·불만·진행 속도 조절은 기본적으로 RAG를 호출하지 않는다.
- 인물·건축·사건·도시 변화·출처·재검증은 RAG가 필요하다.
- 검색 결과가 없거나 신뢰 기준을 넘지 못하면 사실 슬롯을 채우지 않는다.
- `[건립 연도]`, `[인물명]` 같은 자리에 추정값을 넣지 않는다.
- “찾아볼게”로 종료하지 않고 검색 후 답변하거나 근거 부족을 알린다.

### 3.3 저장·기억 표현

다음 조건을 모두 만족할 때만 `기록에 반영한다`고 말한다.

- 실제 저장 기능이 활성화됨
- 저장 대상과 기간이 명확함
- 사용자 동의 또는 서비스 정책상 허용됨

그 외에는 다음처럼 표현한다.

- “이번 대화에서 참고할게요.”
- “이번 여정의 답변 방식에 반영할 수 있어요.”
- “저장 기능을 사용 중이라면 여정 기록에 반영할 수 있어요.”

### 3.4 말투

- 기본 출력은 자연스러운 존댓말이다.
- 원본 반말 문장은 `response_draft_original`로 보존한다.
- 한 세션 안에서 존댓말과 반말을 임의로 섞지 않는다.
- 욕설을 따라 하지 않고, 훈계하거나 과도하게 사과하지 않는다.

### 3.5 개인화

- 한 번의 발화로 장기 선호를 확정하지 않는다.
- 피로·불만·감정은 `context_state`로만 처리한다.
- 명시적인 “짧게 말해 주세요”는 현재 세션에 즉시 적용한다.
- 반복 관찰 또는 명시적 저장 요청이 있을 때만 profile 후보로 승격한다.
- 민감정보, 국적, 정치성향, 성격을 추론하지 않는다.

### 3.6 외부 비교·위치·안전

- 중국 도시 비교는 목포 자료만으로 답하지 않는다.
- 비교 대상 자료가 없으면 한계를 명시한다.
- 거리·소요시간·현재 위치는 지도/위치 서비스 데이터가 있을 때만 말한다.
- 경사로, 휠체어 접근성, 쉼터 위치는 검증된 시설 데이터가 있을 때만 단정한다.
- 안전·접근성 정보가 없으면 추측하지 말고 공식 안내 또는 현장 직원 확인을 권한다.

---

## 4. 상황 목록

| 번호 | situation_id | 사례 수 | 기본 처리 |
|---:|---|---:|---|
| 1 | `INTRO_GIROKSAE` | 3 | 비-RAG |
| 2 | `FREE_CHAT_GREETING` | 3 | 비-RAG |
| 3 | `REFLECTION_POSITIVE_GENERAL` | 3 | 비-RAG |
| 4 | `INTEREST_ARCHITECTURE` | 3 | 혼합/RAG |
| 5 | `INTEREST_PEOPLE` | 3 | RAG |
| 6 | `INTEREST_DAILY_CITY` | 3 | RAG |
| 7 | `COMPARISON_CONTEXT` | 3 | 혼합 |
| 8 | `EMOTION_POSITIVE` | 3 | 비-RAG |
| 9 | `EMOTION_NEGATIVE_HISTORY` | 3 | 혼합 |
| 10 | `LOW_ENGAGEMENT` | 3 | 비-RAG |
| 11 | `HISTORY_FACT_QUESTION` | 3 | RAG |
| 12 | `JOURNEY_CONTEXT_QUESTION` | 3 | 게임 문맥+혼합 |
| 13 | `RESPONSE_STYLE_REQUEST` | 3 | 스타일/출처 |
| 14 | `EVIDENCE_AND_CORRECTION` | 4 | 명확화/RAG |
| 15 | `STRONG_DISSATISFACTION` | 4 | 비-RAG |
| 16 | `CROSS_CULTURAL_COMPARISON` | 3 | 외부 근거 필요 |
| 17 | `PERSONAL_AND_LIGHT_CHAT` | 4 | 비-RAG |
| 18 | `TECHNICAL_HELP` | 3 | 앱 상태 필요 |
| 19 | `NAVIGATION_HELP` | 3 | 위치·지도 필요 |
| 20 | `SAFETY_ACCESSIBILITY` | 3 | 검증된 시설·안전 정보 필요 |

합계: **63개**

---

## 5. 신규 3개 상황의 프로젝트 반영안

기존 54개 seed는 현재 저장소의 `configs/giroksae_situations.json`을 원본 기준으로 유지한다. 아래 신규 9개만 추가 검토 후 병합한다.

### 5.1 `TECHNICAL_HELP`

> 실제 앱 UI와 기능이 확인된 경우에만 구체적인 버튼·아이콘을 안내한다.

| ID | 사용자 입력 | 응답 목표 | 서비스용 응답 템플릿 | 다음 행동 | 상태 | 정책 |
|---|---|---|---|---|---|---|
| TECH_01 | 사진이 안 겹쳐져요. | 오버레이 문제 진단 | 사진 비교 기능이 작동하지 않는군요. 현재 화면 상태와 권한 정보를 확인한 뒤 가능한 조치를 안내할게요. | `OPEN_TECH_DIAGNOSTIC_OVERLAY` | `technical_issue` | `requires_app_state`, `no_rag` |
| TECH_02 | 다음 버튼이 안 눌려요. | 진행 조건·오류 확인 | 다음 단계로 진행되지 않는군요. 완료 조건과 화면 오류 상태를 확인해 볼게요. | `CHECK_MISSION_COMPLETION_STATE` | `technical_issue` | `requires_app_state`, `no_rag` |
| TECH_03 | 소리가 안 나요. | 오디오 상태 확인 | 오디오가 재생되지 않는군요. 앱의 오디오 상태와 기기 음량 설정을 순서대로 확인해 주세요. | `OPEN_AUDIO_TROUBLESHOOTING` | `technical_issue` | `requires_app_state`, `no_rag` |

### 5.2 `NAVIGATION_HELP`

> 위치 권한과 지도 데이터가 없으면 거리·시간·방향을 생성하지 않는다.

| ID | 사용자 입력 | 응답 목표 | 서비스용 응답 템플릿 | 다음 행동 | 상태 | 정책 |
|---|---|---|---|---|---|---|
| NAV_01 | 다음 조각은 어디예요? | 다음 스팟 안내 | 다음 조각은 {verified_next_spot_name}에 있습니다. 위치 정보가 허용된 경우 지도에서 경로를 표시할게요. | `OPEN_ROUTE_TO_NEXT_PIECE` | `navigation_issue` | `requires_journey_state`, `requires_map_data` |
| NAV_02 | 여기서 얼마나 걸려요? | 검증된 이동 시간 제공 | 현재 위치와 목적지가 확인되면 예상 이동 시간을 계산해 드릴게요. | `CALCULATE_ROUTE_ETA` | `navigation_issue` | `requires_location`, `requires_map_data` |
| NAV_03 | 길을 잃었어요. | 현재 위치 확인·안전한 복귀 | 현재 위치를 확인할 수 있다면 경로를 다시 안내할게요. 위치 확인이 어렵거나 위험한 곳이라면 가까운 안내소나 직원에게 도움을 요청해 주세요. | `RECALCULATE_ROUTE_OR_SHOW_HELP` | `navigation_issue` | `requires_location`, `safety_first` |

### 5.3 `SAFETY_ACCESSIBILITY`

> 시설 접근성·우회로·쉼터는 검증된 현장 데이터 없이는 단정하지 않는다.

| ID | 사용자 입력 | 응답 목표 | 서비스용 응답 템플릿 | 다음 행동 | 상태 | 정책 |
|---|---|---|---|---|---|---|
| SAFE_01 | 계단 말고 다른 길 있어요? | 검증된 무장애 동선 제공 | 검증된 우회 동선이 있는지 확인해 볼게요. 확인되지 않으면 임의로 길을 안내하지 않고 공식 시설 안내를 보여드릴게요. | `CHECK_ACCESSIBLE_ROUTE` | `accessibility_request` | `requires_verified_facility_data`, `safety_first` |
| SAFE_02 | 휠체어로 갈 수 있어요? | 접근 가능 여부 확인 | 휠체어 접근 가능 여부와 경사로·엘리베이터 정보를 공식 시설 데이터에서 확인해 볼게요. | `CHECK_WHEELCHAIR_ACCESS` | `accessibility_request` | `requires_verified_facility_data`, `safety_first` |
| SAFE_03 | 너무 더워서 쉬고 싶어요. | 휴식 우선·검증된 쉼터 안내 | 우선 무리하지 말고 쉬어 주세요. 검증된 쉼터 정보가 있으면 가까운 장소를 안내하고, 없으면 현장 안내 표지나 직원에게 확인하도록 도와드릴게요. | `SHOW_VERIFIED_REST_AREAS_OR_HELP` | `current_fatigue`, `heat_discomfort` | `requires_verified_facility_data`, `safety_first` |

---

## 6. 기존 54개에 적용할 업데이트 규칙

1. 기존 ID와 `response_draft` 원문은 변경하지 않는다.
2. 정책성 태그를 `personalization_tags`에서 `policy_flags`로 파생 분리한다.
3. 일회성 감정·피로·불만·임시 스타일을 `context_state`로 분리한다.
4. 역사 사실 질문에는 `requires_rag`와 `requires_evidence`를 적용한다.
5. 출처 요청에는 `requires_citations`와 `OPEN_CITATION_PANEL`을 적용한다.
6. 모호한 질문은 검색 전에 `ASK_CLARIFICATION`으로 처리한다.
7. 기록·기억 표현은 저장 기능과 동의 상태가 확인된 경우에만 사용한다.
8. 서비스 출력은 존댓말 persona를 적용하되 원본 문장은 별도 보존한다.
9. 중국 비교는 `requires_external_comparison_source`가 충족되지 않으면 제한을 알린다.
10. `piece_chat`은 완료한 조각만, `free_chat`은 전체 승인 corpus만 사용한다.

---

## 7. 프로젝트 적용 매핑

### 7.1 권장 파일

```text
configs/giroksae_situations.schema.json
configs/giroksae_situations.json
docs/giroksae_situation_seed.md
docs/GIROKSAE_DIALOGUE_V03.md
src/history_chatbot/dialogue/situation_classifier.py
src/history_chatbot/dialogue/response_policy.py
src/history_chatbot/dialogue/personalization_tags.py
src/history_chatbot/dialogue/action_codes.py
src/history_chatbot/dialogue/context_requirements.py
tests/test_situation_schema.py
tests/test_response_policy.py
tests/test_dialogue_action_codes.py
tests/test_context_requirements.py
```

### 7.2 병합 원칙

- 기존 54개 JSON 레코드는 유지한다.
- 신규 9개는 `review_pending`으로 추가한다.
- 기존 ID는 변경하지 않는다.
- 신규 ID는 `TECH_01~03`, `NAV_01~03`, `SAFE_01~03`을 사용한다.
- 총 레코드 수 assertion은 `63`으로 갱신한다.
- 앱 기능이 아직 없다면 신규 액션은 안전한 fallback을 반환한다.

### 7.3 정책 플래그 최소 허용 목록

```text
no_rag
requires_rag
requires_evidence
requires_citations
requires_clarification
requires_time_scope
requires_visual_context
requires_completed_piece_context
requires_external_comparison_source
requires_app_state
requires_journey_state
requires_map_data
requires_location
requires_verified_facility_data
requires_storage_capability
requires_consent
source_insufficient
incorrect_premise
ambiguous_question
compare_prior_citations
safety_first
do_not_press
do_not_mirror_abuse
```

---

## 8. 필수 테스트

1. 상황 20개와 사례 63개를 정확히 로드한다.
2. 기존 54개 ID와 원문이 변경되지 않았음을 검증한다.
3. 신규 9개 ID가 중복되지 않는다.
4. 인사·감상·불만에서 불필요한 RAG를 호출하지 않는다.
5. 사실 질문에 `requires_rag`와 `requires_evidence`가 적용된다.
6. 빈 검색 결과에서 사실 슬롯을 생성하지 않는다.
7. 잘못된 전제는 그대로 수용하지 않는다.
8. 재검증 요청은 이전 citation과 새 결과를 비교한다.
9. 피로·불만은 장기 프로필로 저장하지 않는다.
10. 스타일 요청은 현재 세션에 즉시 적용한다.
11. 저장 기능이 없거나 동의가 없으면 “기록했다”고 말하지 않는다.
12. 중국 비교 자료가 없으면 비교 사실을 생성하지 않는다.
13. 앱 상태 없이 구체적인 UI 버튼을 지어내지 않는다.
14. 지도 데이터 없이 거리·시간·방향을 생성하지 않는다.
15. 시설 데이터 없이 경사로·휠체어 접근성을 단정하지 않는다.
16. 안전 상황에서 역사 RAG보다 안전 정책을 우선한다.
17. 과거 기본 존댓말의 세션 일관성 검증 항목은 superseded됐다. 현재 말투 검증은 최종 캐릭터 정책의 output domain별 speech level 계약을 따른다.
18. 평가 데이터는 검색 corpus에 들어가지 않는다.
19. production/provisional 데이터 격리가 유지된다.
20. 기존 전체 회귀 테스트가 모두 통과한다.

---

## 9. 반영 순서

1. 이 문서를 `docs/GIROKSAE_DIALOGUE_V03.md`에 추가한다.
2. 기존 54개 구조화 JSON은 그대로 둔다.
3. 스키마에 `context_state`, `policy_flags`, `required_context`, `next_action_code`, `fallback_behavior`를 추가한다.
4. 마이그레이션 스크립트로 기존 태그를 파생 필드에 복사하되 원본은 보존한다.
5. 신규 9개를 `review_pending` 상태로 추가한다.
6. 상황 분류기에 신규 3개 상황을 추가한다.
7. UI/게임 서비스가 제공하는 실제 context capability를 확인한다.
8. capability가 없는 액션은 안전한 fallback으로 처리한다.
9. 신규 단위 테스트와 전체 회귀 테스트를 실행한다.
10. diff와 데이터 수를 검토한 뒤 commit한다.

---

## 10. 최종 판정

v0.3은 프로젝트에 반영할 수 있다. 다만 “20개 상황/63개 사례가 존재한다”는 것과 “모든 답변을 실제로 즉시 수행할 수 있다”는 것은 다르다.

- 역사 대화 17개 상황은 현재 RAG·대화 정책에 바로 연결 가능하다.
- 신규 기술·위치·안전 상황 3개는 앱 상태, 지도, 시설 데이터 계약을 먼저 연결해야 한다.
- 실제 기능이 없는 상태에서는 안전한 안내와 fallback만 반환해야 한다.
- 총 서비스 투입 준비도는 대화 정책 기준 약 **90%**, 실제 앱 통합 기준 약 **75~80%**로 평가한다.
