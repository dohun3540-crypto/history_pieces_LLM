# 다국어 검색 평가

## 평가 범위

2026-08-01의 `provisional_hackathon` 48문서·133청크를 대상으로 수동 검증한
30개 사례를 사용했다. 한국어 사실 질문 10개, 같은 사실의 중국어 간체 질문 10개,
무관 질문 5개, corpus에 답이 없는 근거 부족 질문 5개다. 개항, 철도, 독립운동가,
근대 건축, 세관/항만, 상업과 도시 변화를 포함한다.

정답 source/chunk ID는 corpus 본문을 직접 확인해
`tests/fixtures/retrieval/multilingual_e5_evaluation.json`에 기록했다. 평가 데이터는
corpus와 분리되어 있다. 표본이 작고 현재 corpus에는 웹 페이지 탐색 UI 잡음이 많으므로
아래 수치를 일반적인 성능으로 확정해서는 안 된다.

## Backend 비교

| backend | R@1 | R@3 | R@5 | MRR | 한국어 R@3 | 중국어 R@3 | 근거 없음 거절 | 평균 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.35 | 0.50 | 0.55 | 0.4267 | 0.90 | 0.10 | 0.30 | 10.951 | 22.183 |
| hashing dense | 0.05 | 0.05 | 0.05 | 0.0500 | 0.10 | 0.00 | 1.00 | 37.985 | 50.386 |
| E5 dense | 0.45 | 0.50 | 0.60 | 0.4892 | 0.90 | 0.10 | 0.00 | 69.285 | 76.942 |
| BM25 + hashing | 0.30 | 0.40 | 0.50 | 0.3617 | 0.70 | 0.10 | 0.30 | 43.082 | 60.266 |
| BM25 + E5 | 0.45 | 0.55 | 0.70 | 0.5242 | 1.00 | 0.10 | 0.00 | 75.519 | 93.511 |

기본 0.72 threshold에서 BM25+E5는 한국어 10건을 모두 top 3에서 찾았지만 중국어는
1건만 찾았다. E5가 hashing보다 전체 recall과 MRR은 높았으나, 무관·근거 부족 질문을
거절하지 못했다. 특히 E5 cosine 점수의 높은 baseline과 긴 UI 잡음 청크 때문에 단일
절대 threshold가 답 존재 여부를 안정적으로 구분하지 못했다.

## 튜닝 후보

RRF k 10/30/60, dense/sparse top-k 8/12/16, final top-k 3/5, 문서당 1/2개와
dense threshold 0.72/0.76/0.80/0.82/0.84/0.86을 비교했다.

- RRF k와 candidate top-k 변화는 이 작은 표본에서 의미 있는 차이가 없었다.
- final top-k 3은 R@5를 낮췄다.
- 문서당 1개는 chunk recall을 낮췄다.
- threshold 0.80은 R@3 0.60, 중국어 R@3 0.20, 거절 0.20이었다.
- threshold 0.82는 R@3 0.55, 한국어 R@3 1.00, 중국어 R@3 0.10,
  근거 없음 거절 0.30이었다.
- 0.84/0.86은 거절을 더 개선하지 않고 R@5만 낮췄다.

따라서 실험용 공통 후보는 dense/sparse top-k 12, final top-k 5, RRF k 10,
minimum dense score 0.82, 문서당 2개다. 이는 production 기본값이 아니다.
한국어는 E5 hybrid가 유망하지만, 중국어는 별도 설정을 추천할 정도의 성능이 확인되지
않았다. 중국어는 현재 근거 부족 처리 또는 검수된 query translation/용어 사전 실험이
우선이며, 이번 작업에서는 중국어 대사나 번역 corpus를 만들지 않았다.

## 결론과 제한

기존 hashing 기본값과 production 설정은 유지한다. E5를 hackathon lane의 실험 backend로
명시 선택할 수 있지만, 답 존재 판정에는 retrieval score만 사용하지 말고 lexical gate,
점수 margin, reranker 또는 evidence entailment 검사를 추가 평가해야 한다. 특히 근거 부족
질문은 관련 장소 문서가 검색되더라도 질문의 세부 답이 없을 수 있으므로 검색 성공과
답변 가능성을 구분해야 한다.

실제 orchestrator smoke에서는 한국어 개항·철도·인물 질문, citation 반환,
`piece_chat`과 `free_chat` 연결이 정상 동작했다. 그러나 중국어 `朴爱顺`/`金玉实`
질의의 top 3가 정답 인물을 포함하지 않았고, 양자컴퓨터 질문과 가옥 설계자 질문도
청크를 반환했다. 이 관찰은 `reports/e5_smoke_test.json`에 그대로 기록했다.

원시 case별 순위·점수·latency와 모든 튜닝 후보는
`reports/embedding_benchmark.json`에 보존한다.
