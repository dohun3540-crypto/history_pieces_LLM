# Development End-to-End RAG

현재 end-to-end 경로는 실제 역사 서비스가 아니라 소프트웨어 개발 검증용이다.
`tests/fixtures/rag`의 모든 내용은 **테스트용 가상 자료이며 실제 역사 사실이
아니다**. 실제 Llama, GPU, Hugging Face 로그인 및 외부 네트워크는 사용하지 않는다.

처리 흐름:

1. 질문·locale·top_k 검증
2. session_id 조회 또는 새 세션 생성
3. 지시어가 있는 후속 질문에 직전 질문 문맥 결합
4. fixture 하이브리드 검색
5. 검색 임계값, 중복 chunk, 문서별 chunk 제한 적용
6. 시스템 지침·대화 요약·검색 근거·질문을 구분한 프롬프트 생성
7. 근거가 있을 때만 MockLLM 호출
8. 실제 사용한 chunk에서만 출처 생성
9. 세션 기록 갱신

```powershell
python -m history_chatbot.chat.cli ask "붉은 등대 전시관을 알려줘"
python -m history_chatbot.chat.cli ask "그 건물은 어떤 설정이야?" --session-id "<SESSION_ID>"
python -m history_chatbot.chat.cli reset --session-id "<SESSION_ID>"
```

근거가 없으면 다음 의미의 응답을 반환한다.

```json
{
  "answer": "확인 가능한 자료가 부족합니다.",
  "status": "insufficient_evidence",
  "sources": [],
  "used_chunks": 0
}
```

fixture 검색 성공은 역사 정확도 검증이 아니다. production은 fixture와 MockLLM을
모두 차단한다.
