# Multilingual search evaluation run

- 실행일: 2026-08-01
- corpus: provisional_hackathon 48문서 / 133청크
- model: intfloat/multilingual-e5-small
- revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
- device: CPU
- dimension: 384
- query/passage prefix: 적용
- normalized embeddings: true

## 핵심 결과

BM25+E5는 전체 Recall@1/3/5가 0.45/0.55/0.70, MRR 0.5242였다.
한국어 Recall@3는 1.00이었지만 중국어 Recall@3는 0.10이었다. 0.72 threshold의
무관·근거 부족 거절률은 0.00이었다. Threshold 0.82 후보는 한국어 Recall@3 1.00을
유지하고 전체 거절률을 0.30으로 높였으나 중국어 Recall@3는 0.10이고 근거 부족
거절률은 0.20에 그쳤다.

이 결과는 E5가 hashing보다 검색 recall은 개선하지만 현재 corpus 상태에서 안전한
다국어 기본 backend로 즉시 전환할 수 없음을 보여준다. 상세 수치와 case별 결과는
`embedding_benchmark.json`, 해석과 권장안은
`docs/MULTILINGUAL_SEARCH_EVALUATION.md`를 참조한다.

통합 smoke에서 한국어와 두 대화 모드는 정상적으로 근거와 citation을 반환했다.
중국어 인물 질의의 오검색과 무관·근거 부족 질문의 false positive도 재현됐으므로
E5 후보를 production 기본값으로 적용하지 않았다.
