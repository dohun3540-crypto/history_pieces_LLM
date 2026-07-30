# 개발 모드와 가상 fixture

현재 실제 승인 자료는 0건이다. 개발 및 테스트는
`tests/fixtures/rag`의 소형 가상 자료로만 수행하며, 모든 fixture 문서와 문장에는
`테스트용 가상 자료이며 실제 역사 사실이 아님` 표시가 있다.

- `development`: 가상 fixture와 MockLLM을 사용할 수 있다.
- `test`: 자동 테스트에서 가상 fixture와 테스트 encoder를 사용할 수 있다.
- `production`: fixture와 MockLLM을 금지한다. `reviewed`,
  `allowed_for_rag=true`, 권리·출처 검증을 통과한 실제 자료가 없으면 준비 미완료
  오류를 낸다.

fixture 경로는 `data/index_ready`와 분리된다. fixture 레코드를
`data/index_ready`에 넣으면 실행 모드와 관계없이 거부한다. fixture 테스트 통과는
검색·격리 로직 검증일 뿐 목포 역사 사실의 정확성 검증이 아니다.
