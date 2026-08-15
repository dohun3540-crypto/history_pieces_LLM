# 목포 역사 RAG 대화 품질 평가

평가일: 2026-08-11 (Asia/Seoul)

## 실행 조건

- corpus: local `verified_hackathon` 112 documents / 624 chunks
- retrieval: 실제 `HybridRetrievalService` 경로(hashing dense + BM25 + RRF + threshold + dedup)
- generation: `MockLLM` (원격 Llama endpoint 미설정)
- 비교 기준: `git archive HEAD`의 수정 전 소스와 현재 작업 트리에 동일한
  10 scenarios / 26 turns를 실행
- 제한: mock 답변은 생성 품질이 아니다. claim-level groundedness, directness,
  fallback 자연스러움, overall conversational quality는 `unavailable`이다.

## BEFORE / AFTER / DELTA

| 자동 검사 | BEFORE | AFTER | DELTA |
|---|---:|---:|---:|
| scenarios | 10 | 10 | 0 |
| user turns | 26 | 26 | 0 |
| evidence/source 계약 | 26/26 | 26/26 | 동일 |
| 답변 완결성 | 25/26 | 26/26 | +1 |
| 지정 문맥 유지 | 3/7 | 7/7 | +4 |
| out-of-scope 거절 | 1/1 | 1/1 | 동일 |
| 동일 fallback 반복 | 2 | 0 | -2 |

| 실패 유형 | BEFORE | AFTER | 변화 |
|---|---:|---:|---:|
| RETRIEVAL_FAILURE | 11 | 0 | -11 |
| CONTEXTUALIZATION_FAILURE | 4 | 0 | -4 |
| CONTEXT_MEMORY_FAILURE | review에서 관찰 | 0 deterministic regressions | 개선 |
| FALLBACK_FAILURE | 2 | 0 | -2 |
| GENERATION_FAILURE | unavailable | unavailable | 평가 불가 |
| TRUNCATION_FAILURE | 1 | 0 | -1 |
| HALLUCINATION | 0 contract violations | 0 contract violations | 동일 |
| TOPIC_SWITCH_FAILURE | contextualization에 포함 | 0 deterministic regressions | 개선 |
| EVIDENCE_GAP | scenario에 의도적으로 존재 | 동일 | 안전 처리 |
| OUT_OF_SCOPE_FAILURE | 0 | 0 | 동일 |
| OTHER | 0 | 0 | 동일 |

`HALLUCINATION=0`은 evidence/source 구조 계약만 뜻하며 모델 답변 claim의 사실 검수가
끝났다는 뜻이 아니다. mock backend에서는 역사 claim 점수를 만들지 않았다.

## 실패 원인과 수정

- data: corpus에 천장 색, 목포진 책임자, 폐진 이후처럼 실제 gap이 있다. 이 경우
  answerability를 partial/unanswerable로 표시했다.
- retrieval: 구어체 filler가 hashing coverage 분모를 키웠고, active place 결과는
  정렬만 되어 다른 지역 문서가 섞였다. threshold는 내리지 않고 filler 정규화와
  명시 entity exact filtering을 추가했다.
- contextualization: “왜 왔던 거야”, “언제 처음 생긴 거야”가 누락됐고 chain
  follow-up이 직전 질문 문자열을 재귀적으로 이어 entity를 잃었다. stable structured
  topic/place/event를 사용하도록 바꿨다.
- context memory: 과거 assistant 문장은 계속 근거로 금지한다. 대신 이전 turn에서
  실제 검색된 chunk ID만 별도 `evidence_turns`로 저장하고 같은 topic의 후속 질문에서만
  partial evidence로 복원한다.
- prompt: 첫 문장 직접 답변, 잘못된 전제 비동조, 부분 답변, 완결 문장 규칙을 추가했다.
- fallback: 의도별 문구와 한 번의 관련 질문 방향으로 줄여 동일 hard fallback 반복을
  낮췄다.
- generation: `length/max_tokens`뿐 아니라 열린 괄호가 남은 stop 응답도 완성 문장
  prefix만 사용하거나 생성 실패로 처리한다. temperature/top-p/token 값은 실제 모델
  평가 없이 변경하지 않았다.

## split 및 권리

- train/dev: 4 scenarios / 11 turns
- validation: 3 scenarios / 7 turns
- holdout test: 3 scenarios / 8 turns

split은 scenario/topic/document 단위이며 holdout은 prompt나 few-shot에 사용하지
않는다. 현재 모든 근거는 `allowed_for_training=false`이므로 이 데이터는 평가 및 prompt
개발 전용이고 실제 SFT 데이터가 아니다.

## Fine-tuning 조사

- NVIDIA RTX 4060 Laptop GPU 8 GiB는 시스템에 보이지만 설치된 PyTorch는 CPU build이며
  `torch.cuda.is_available()`은 false다.
- Transformers는 설치되어 있으나 PEFT, datasets, TRL은 없다.
- local Llama checkpoint와 tokenizer 경로가 없다.
- 원격 inference endpoint/model 환경변수도 설정되어 있지 않다.
- corpus license/metadata가 training을 허용하지 않는다.

따라서 실제 SFT/LoRA는 수행하지 않았다. `configs/lora_sft.example.yaml`과
`scripts/train_lora.py`는 승인된 별도 데이터·checkpoint·CUDA 서버에서만 fail-closed로
동작하도록 준비했다.

---

# 실제 원격 Llama 최종 품질 감사 (2026-08-15)

이 절은 위 mock 기반 사전 평가와 분리된 실제 모델 평가다. 실제 모델은
`beomi/Llama-3-Open-Ko-8B-Instruct-preview`, backend는 DIS03의 RTX A6000에서 실행한
vLLM 0.7.3이며, 품질 평가에 사용한 mock generation은 0건이다. 원시 JSONL은
`.runtime/final-quality/`에 보존했고 요약은
`evaluation/conversation_quality/final_quality_audit_2026-08-15.json`에 기록했다.

## 평가량과 결과

| 구간 | turns | 실제 generation | fallback | backend error | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Round 0 frozen baseline | 146 | 103 | 32 | 1 | 14,008 ms | 14,198 ms |
| Round 1 targeted check | 26 | 19 | 6 | 0 | - | - |
| Round 2 frozen suite | 146 | 32 | 52 | 52 | 7,628 ms | 7,935 ms |
| Round 3 frozen suite, interrupted | 146 | 72 | 52 | 12 | 7,157 ms | 7,396 ms |

동결 suite는 single-turn 92개와 multi-turn 13 scenarios/54 turns, 총 146 turns다.
전체 round에서 성공한 실제 품질 generation은 226건이다. Round 3 결과는 학교 SSH와
tunnel 단절 후 12건이 `connection_error`가 되었고, 이후 regression rollback을 실제
모델로 재평가하지 못했으므로 **final metric으로 사용할 수 없다**.

## 발견 문제와 원인

- **CRITICAL / retrieval**: baseline의 no-evidence 8건 중 6건이 hashing n-gram의 우연한
  겹침으로 무관 문서를 통과시켰다. 고하도 질문에 광주학생운동, 송내호 질문에 다른
  인물을 답하는 식의 잘못된 entity/time/place 단정이 실제로 재현됐다.
- **CRITICAL / grounding**: baseline은 low evidence-overlap signal 28건, unsupported
  numeric signal 31건이었다. 이는 claim 전체 확정 수가 아니라 deterministic review
  signal이며, 세부 claim은 수동 검수가 필요하다.
- **MAJOR / context resolution**: 생략형 “건립 시기는?”와 명시적 topic return이
  불안정했다. Round 3에서도 “다시 목포역으로”가 직전 광주학생운동 evidence를 이어
  쓰는 오류가 재현됐다.
- **MAJOR / output handling**: baseline generation 87건이 length 제한에 닿았고 모델이
  prompt marker나 긴 연도열을 출력했다. 사용자에게 미완성 꼬리를 노출하지 않는
  stabilization을 추가했다.
- **CRITICAL / operations**: 실제 launcher는 job 16899/DIS03에서 정상 준비됐지만 약
  10분의 연속 평가 뒤 SSH connection reset이 발생했다. 이어 로그인 노드
  `155.230.135.209:10000` 자체가 connect timeout이 되어 holdout과 최종 cleanup 확인을
  진행할 수 없었다. 로그에는 CUDA OOM/traceback이 없고, 마지막 성공 요청까지 vLLM
  HTTP 200이었다.

## 개선 round

- Round 0: 수정 전 실제 baseline을 동결했다. no-evidence restriction은 2/8이었다.
- Round 1: hashing 후보/boilerplate/title 우선순위, query filler 제거, evidence-only
  prompt를 최소 수정했다. targeted no-evidence 4/4가 안전 fallback으로 바뀌었다.
- Round 2: context 생략형과 출력 stabilization, 256-token 운영 상한을 적용했다. 기존
  완료문장 검사가 punctuation 없는 length 응답을 예외로 바꿔 52건 `llm_error` 회귀가
  발생했다.
- Round 3: 해당 length 응답을 raw로 노출하지 않고 안전한 grounded limitation으로
  처리했다. no-evidence restriction은 8/8, low-overlap signal은 9, unsupported numeric
  signal은 4로 감소했으나 후반 SSH 단절로 12건이 실패했다. 이후 unit regression에서
  evidence title의 active-topic 덮어쓰기와 활용형 longest-anchor 조건을 발견해 두
  변경은 제거했다. 네트워크가 복구되지 않아 이 최종 코드의 actual 재평가는 없다.

## fallback/error-path 및 회귀

no evidence는 final attempt에서 8/8 안전 제한을 보였다. partial/conflicting evidence는
일부 actual 응답에서 부분 답변 또는 차이 명시를 확인했지만 SSH 단절 때문에 전체
scenario를 완주하지 못했다. connection refused, timeout, HTTP non-200, malformed JSON,
empty choices/content, retrieval exception 등 fault-injection 경로는 별도 테스트 79개가
통과했다. 전체 회귀 결과는 590 collected, 585 passed, 5 known skipped, 0 failed다.

## 운영 검증과 중단 사유

재시작에서 SSH key, Slurm allocation, DIS03 탐지, RTX A6000, vLLM 모델 load,
`/v1/models` expected ID, 실제 completion, tunnel, FastAPI, `/ready`의
`retriever=true`/`llm=true`, browser-ready까지 성공했다. 첫 `/v1/models`는 launcher
시작 후 약 4분 26초에 확인됐다. 장시간 평가 중 tunnel/SSH가 reset된 뒤 login node가
계속 timeout이어서 holdout 24 single + 5 multi scenarios/20 turns, 최종 15~20-turn
실사용 flow, 최종 Ctrl+C 및 Slurm cleanup 검증은 실행하지 못했다. 따라서 최종 판정은
`[NOT READY TO COMMIT]`이다.

---

# Canonical final verdict after Blocker Fix Final Pass

바로 앞 `SSH 복구 후 최종 재개 검증` 절은 blocker 수정 전 상태의 역사 기록이다.
이 문서의 `Blocker Fix Final Pass (2026-08-15)` 절에 기록한 후속 actual holdout,
targeted, fresh sanity와 613-test regression 결과가 현재 코드의 canonical final 결과이며,
그 결과가 이전 `[NOT READY TO COMMIT]` 판정을 대체한다. 현재 최종 판정은
`[READY TO COMMIT]`이다.

---

# Blocker Fix Final Pass (2026-08-15)

이 절은 위 재개 검증에서 발견된 여섯 blocker를 대상으로 한 최종 corrective pass다.
과거 baseline이나 frozen holdout 문항은 변경하지 않았고, launcher/backend 구조도
수정하지 않았다. 실제 품질 평가는 동일한
`beomi/Llama-3-Open-Ko-8B-Instruct-preview`만 사용했으며 mock generation은 0건이다.

## 원인과 최소 수정

- **unrelated retrieval**: hashing dense 결과가 짧은 공통 n-gram과 본문 navigation을
  subject 일치로 오인했다. 한국어 조사로 표시된 명시 subject를 일반적으로 추출하고,
  title 또는 factual opening과 일치하지 않는 후보를 fail-closed로 처리했다. 철도와 항만
  같은 명시적 병렬 facet은 각각의 보조 근거를 보존한다.
- **context contamination / topic return**: resolver가 최근 topic을 현재 메시지보다 먼저
  사용했고 알려진 place 목록 밖의 복귀 대상을 찾지 못했다. 현재 명시 subject, 명시
  return target, 검증된 active topic 순서를 적용하고 evidence turn의 user text만 anchor로
  사용했다. Assistant 생성 문장은 검색 근거나 person anchor로 사용하지 않는다.
- **false-premise drift**: insufficient/fallback 직후의 낮은 신뢰 context가 다음 짧은
  질문에 강한 anchor로 재사용됐다. fallback marker가 있는 turn은 상속하지 않고,
  검증된 evidence turn이 없으면 새 subject를 추측하지 않는다.
- **question echo**: sentence 단위 normalized exact/near-exact echo를 제거하며 무한 retry는
  하지 않는다. 짧은 질문도 6자 이상이면 exact echo guard가 적용된다.
- **truncation warnings**: 256-token 제한에서 `finish_reason=length`가 빈번하지만 raw
  unfinished output은 stabilization 뒤 노출되지 않았다. 320/384로 올리면 장문 근거 복제
  위험이 있어 256을 유지했다. warning flag 자체는 숨기지 않았다.

## Targeted 및 frozen holdout

최종 targeted acceptance set은 40 turns(31 actual generations, 9 safe fallbacks)로,
unrelated retrieval, four generic return chains, entity/pronoun follow-up, false-premise chain,
question echo와 truncation-prone factual 응답을 확인했다. 최종 40/40에서 unrelated
evidence, context drift, explicit return failure, echo, prompt leak, unfinished/raw fragment,
backend error는 0건이었다.

| frozen holdout 지표 | 이전 | blocker-fix |
|---|---:|---:|
| turns | 44 | 44 |
| actual generations | 28 | 30 |
| fallback | 16 | 14 |
| backend errors | 0 | 0 |
| critical unrelated retrieval | 4 | 0 |
| explicit topic-return failure | 1 | 0 |
| exact question echo | 1 | 0 |
| prompt leak | 0 | 0 |
| unfinished exposed | 0 | 0 |
| truncation warnings | 23 | 26 |
| p50 latency | 7,281 ms | 7,176 ms |
| p95 latency | 7,367 ms | 7,343 ms |

Frozen holdout의 organization chain 첫 turn은 당시 classifier가 RAG로 보내지 않아 뒤의
ordinal follow-up 한 건이 무관 evidence로 이동했다. holdout은 지시대로 한 번만 실행하고
문항/결과를 바꾸지 않았다. 이후 generic factual comparison routing, first/second group
reference, subject별 evidence 보존을 수정한 뒤 별도 actual 4-turn chain에서 송죽회와
신간회 source가 유지되고 question echo가 0임을 확인했다.

Fresh sanity는 holdout과 다른 14 turns(9 actual, 5 safe fallback)였다. 두 follow-up chain,
두 generic topic-return, no-evidence 2건과 false-premise 2건에서 critical grounding/context
failure는 0건이었다. 최신 regression 보정 뒤에도 별도 actual 8 turns로 false-premise
3-turn fail-closed, 양림동 선교사 evidence 유지, 달리도→나불도→달리도 복귀를 재확인했다
(5 actual, 3 safe fallback, backend error 0).

## 회귀, 운영, 남은 제한

최종 전체 suite는 **613 collected, 608 passed, 5 known skipped, 0 failed**다. 이전
598개보다 15개 증가한 것은 generic subject relevance, breadcrumb rejection, explicit
return, false-premise trust, ordinal group reference, relationship ellipsis, multi-subject
evidence 및 question echo 회귀 테스트를 추가했기 때문이다. 기존 fault-path 79개도 전체
suite에 포함되어 통과했다.

실제 tunnel의 `/v1/models`는 expected model ID를 반환했고 최신-code app `/ready`는
`ready/retriever/llm=true`, `backend=remote`, `llm_status=ready`였다. 이번 pass에서 launcher
구조는 변경하지 않았다. 검증 후 launcher-owned job 16910과 local process를 정리했고
ports 8000/8002/18001 listener는 0이었다. unrelated jobs 16891/16892는 보존했다. 남은
제한은 256-token generation의 높은 truncation warning,
간헐적인 보수적 fallback/부분 답변, 원문 corpus의 `닫기` 같은 scrape 표식이 답변에
남는 경미한 UX 문제다. raw unfinished answer, prompt marker, severe repetition, exact
question echo 및 재현 가능한 critical unrelated grounding은 최종 검증에서 0건이다.

핵심 acceptance blocker가 generic fix, actual frozen holdout, post-holdout targeted와 fresh
sanity에서 해소됐고 전체 회귀가 통과했으므로 최종 판정은 **`[READY TO COMMIT]`**이다.

---

# SSH 복구 후 최종 재개 검증 (2026-08-15)

이 절이 위의 SSH 장애 시점 interim 결론을 대체하는 **FINAL CURRENT CODE** 결과다.
학교 SSH는 `SSH_OK`/`ABRM02`로 복구됐고, rollback 이후 working tree를 최종 후보로
동결한 뒤 실제 원격 Llama 평가, frozen holdout, 20-turn 실사용, 전체 regression,
재시작과 두 차례 cleanup을 완료했다. 품질 평가에 사용한 mock generation은 0건이다.

## 최종 평가량

| 구간 | turns | 실제 generation | fallback | backend error | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Round 0 frozen baseline | 146 | 103 | 32 | 1 | 14,008 ms | 14,198 ms |
| FINAL current frozen suite | 146 | 106 | 37 | 0 | 7,214 ms | 7,412 ms |
| frozen holdout | 44 | 25 | 16 | 0 | 7,281 ms | 7,367 ms |
| targeted topic-return | 21 | 21 | 0 | 0 | - | - |
| 1-session real-use flow | 20 | 16 | 4 | 0 | 7,065 ms | 7,364 ms |

누적 성공 actual quality generation은 이전 226건과 재개 과정의 intermediate/final
381건을 합쳐 607건이다. FINAL frozen suite는 single-turn 92개와 13 scenarios/54
multi-turns이고, holdout은 사전에 동결된 24 single-turn + 5 scenarios/20 turns를
변경 없이 사용했다.

## BASELINE 대 FINAL

| 지표 | BASELINE | FINAL |
|---|---:|---:|
| actual generations | 103 | 106 |
| fallback | 32 | 37 |
| backend errors | 1 | 0 |
| low evidence-overlap signals | 28 | 20 |
| unsupported numeric signals | 31 | 4 |
| no-evidence safe restriction | 2/8 | 8/8 |
| false-premise correction/limitation signals | 2/10 | 9/10 |
| truncation warnings | 87 | 98 |
| unfinished answers exposed by harness | 0 | 0 |
| severe repetition failures | 0 | 0 |
| p50 latency | 14,008 ms | 7,214 ms |
| p95 latency | 14,198 ms | 7,412 ms |

FINAL에서 no-evidence 안전성, 숫자 폭주, backend error 및 지연은 개선됐다. 다만
`truncation_warnings`는 256-token 상한 때문에 98건으로 높으며, 후처리가 raw 미완성
꼬리를 노출하지는 않았어도 모델 출력 여유가 작다는 신호로 남는다.

## Holdout 및 실제 대화에서 발견된 미해결 문제

- **CRITICAL / retrieval-grounding**: `목포진의 역할과 시기`가 김대중노벨평화상기념관과
  국립해양유산연구소를 검색하고 김대중 관련 답변을 생성했다.
- **CRITICAL / context resolution**: 선교사·단체 follow-up 및 false-place correction
  chain에서 기존 entity를 잃고 철도·학생운동 등 무관 evidence로 이동했다.
- **MAJOR / topic return**: targeted 5 scenarios/21 turns와 frozen final의 두 return,
  20-turn flow의 `다시 목포역`은 성공했지만, holdout의 `나불도 → 다시 달리도`는
  달리도가 explicit place로 인식되지 않아 직전 나불도 문맥에 머물렀다. 즉 재현 가능한
  topic-return 실패가 1건 남았다.
- **MAJOR / output UX**: 231개 final/holdout/targeted/flow 응답의 deterministic scan에서
  prompt marker leak은 0건, raw unfinished는 0건이었으나 20-turn flow에서 사용자
  질문을 그대로 반복한 exact question echo 1건이 노출됐다. 연도 과다 신호는 frozen
  final에 2건 있었다.
- **MAJOR / false-premise continuity**: frozen suite 첫 전제 확인은 안전했지만 다음
  `관련 인물은?`에서 무관한 해양권역/철도 evidence가 사용됐다.

따라서 단위 테스트와 대표 targeted topic return은 통과했지만, 새로운 frozen holdout이
동일 계열의 일반화 실패를 재현했다. Round 3은 허용된 마지막 improvement round이므로
추가 entity별 mapping이나 architecture 확장은 수행하지 않았다.

## Fallback과 오류 경로

- no evidence: FINAL 8/8 안전 제한. 존재하지 않는 왕조 등에서 사실을 만들지 않았다.
- partial evidence: 일부는 근거 범위의 답변을 제공했지만, multi-turn에서 전체 fallback
  또는 무관 evidence로 이동하는 실패가 남았다.
- conflicting evidence: prompt에 충돌 상태를 전달하고 하나의 사실로 합치지 않는 경로를
  정적/실제 suite에서 확인했다.
- backend unavailable, connection refused, timeout, retrieval exception, empty output,
  malformed output: fault-injection subset 79/79 passed. 실제 품질 generation과 섞지 않았다.
- output scan: `[검색 근거]`, `[자료1]`, `[사용자 질문]`, 내부 system marker 노출 0건;
  exact question echo 1건; harness 기준 unfinished 0건, severe repetition 0건이다.

## 회귀 및 운영

전체 suite는 598 collected, 593 passed, 5 known skipped, 0 failed다. 이전 590개보다
8개 증가한 이유는 prompt leak, false-premise RAG routing, hashing guard, explicit topic
return 회귀 테스트를 추가했기 때문이다. 테스트 삭제는 없다.

운영 재검증에서는 job 16900과 restart job 16901이 모두 DIS03에 동적으로 할당됐다.
SSH key, p02/RTX A6000, vLLM 0.7.3, 정확한 model ID,
`/v1/chat/completions` non-empty 실제 생성, tunnel, FastAPI, `/ready`의
`ready/retriever/llm=true`, `backend=remote`, `llm_status=ready`를 확인했다. restart ready는
약 244초였다. 두 번의 Ctrl+C 이후 ports 8000/8002/18001과 launcher PID가 모두
사라졌고 jobs 16900/16901도 제거됐다. unrelated pending jobs 16891/16892는 보존했다.
이전 학교 SSH transient 장애는 이번 재개 동안 재현되지 않았다.

## 최종 판정

원격 Llama 인프라와 launcher 수명주기는 acceptance를 통과했고 regression/fault path도
모두 통과했다. 그러나 frozen holdout과 20-turn 실사용에서 재현 가능한 무관 evidence,
context 오염, explicit topic-return 실패, question echo가 남아 있다. 이는 단순 문체 제한이
아니라 grounding/multi-turn acceptance 위반이므로 최종 판정은
`[NOT READY TO COMMIT]`이다.

---

# Canonical final status

위 `SSH 복구 후 최종 재개 검증`의 `[NOT READY TO COMMIT]`은 blocker 수정 전 결과다.
그 뒤 수행한 `Blocker Fix Final Pass (2026-08-15)`의 실제 frozen holdout, targeted,
fresh sanity, 최신-code 확인 및 613-test regression이 현재 코드의 최종 근거다.
따라서 현재 canonical 판정은 `[READY TO COMMIT]`이다.
