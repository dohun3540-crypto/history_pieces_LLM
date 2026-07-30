# 아키텍처

## 현재 요청 흐름

1. CLI가 사용자의 `original_query`를 받습니다.
2. 전처리 계층이 원문을 보존하며 NFC와 공백을 정리한 `normalized_query`를 만듭니다.
3. `BaseRetriever` 구현이 문서를 검색합니다. 현재 구현은 메모리 키워드 검색입니다.
4. `BaseLLM` 구현이 검색 결과로 답합니다. 현재 `MockLLM`은 외부 호출을 하지 않습니다.
5. 답변과 질의를 `ConversationMemory`에 세션 기록으로 보관합니다.

검색 결과에는 `title`, `source`, `content`가 함께 이동하므로 답변에서 근거를 표시할
수 있습니다. 근거가 없으면 공통 fallback을 반환합니다.

## 교체 지점

- 검색: `KeywordRetriever` → FAISS, Chroma 또는 별도 벡터 검색 서비스
- 생성: `MockLLM` → 외부 GPU 서버의 Llama Instruct 어댑터
- 메모리: 프로세스 메모리 → 동의·보존 정책이 적용된 별도 저장소
- 언어: 질의 언어 감지, 다국어 임베딩, 번역/언어별 프롬프트 계층 추가

대화 메모리는 생성 모델과 독립적이며 가중치를 변경하거나 대화를 자동으로 학습
데이터에 넣지 않습니다.
