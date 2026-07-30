# Production Readiness

`/api/health`는 애플리케이션 프로세스의 생존만 확인한다. `/api/readiness`는 생성
요청을 실행하지 않고 원격 `/health`와 `/ready`만 조회하며 다음을 구분한다.

- `development_ready`
- `hackathon_data_missing`
- `hackathon_data_partial`
- `hackathon_index_missing`
- `hackathon_index_ready`
- `hackathon_expired`
- `hackathon_rights_unconfirmed`(경고)
- `remote_llm_unconfigured`
- `remote_llm_unreachable`
- `model_not_ready`
- `missing_real_documents`
- `missing_production_index`
- `production_ready`
- `provisional_data_detected`(차단)
- `provisional_index_detected`(차단)

production 준비 조건:

1. `LLM_BACKEND=remote`이며 URL·모델·허용 host 설정이 유효함
2. 원격 health와 ready 응답 성공
3. 모델 준비 완료
4. fixture가 아닌 실제 eligible 문서 존재
5. 운영 인덱스와 embedding model ID/revision 및 자료 snapshot 일치

컨텍스트 예산은 시스템 지침과 현재 질문을 먼저 보존하고, 높은 점수 근거와 최근
대화를 순서대로 유지한다. 오래된 대화와 낮은 점수 근거부터 제거하며 제거 수를
응답 metadata에 기록한다. 실제 tokenizer가 없으므로 현재는 보수적인 문자 기반
추정기를 사용한다.

실제 배포 전에는 선택한 Meta Llama 모델의 라이선스, gated 조건, 정확한 context
window와 tokenizer, RAM/VRAM 및 서버 동시성 제한을 다시 확인해야 한다.
MockLLM 테스트 통과는 실제 모델 품질 검증이 아니다.

해커톤 임시 자료 48건은 정식 승인 자료가 아니다. production에서는
`provisional_hackathon` 청크 경로 설정 자체를 거부하며, `.runtime/indexes/hackathon`
인덱스를 운영 인덱스로 재사용할 수 없다. 운영 준비 판정에는 계속
`reviewed + allowed_for_rag=true` 실제 자료만 반영한다.
