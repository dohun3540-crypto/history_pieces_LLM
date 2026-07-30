# 증분 자료 반영 가이드

공식 자료 추가 흐름은 다음과 같다.

`collect → validate → review → prepare-index → incremental-index → index-version 갱신`

1. 수집 결과는 `draft`, RAG·학습 불허로 시작한다.
2. 권리와 역사 내용을 사람이 검수한다.
3. 8단계 prepare가 승인 자료 스냅샷과 tombstone을 만든다.
4. build-index가 `chunk_id`와 콘텐츠 해시가 같은 벡터를 재사용한다.
5. 새 청크만 임베딩하고 수정 청크는 재검수 후 다시 임베딩한다.
6. rejected 또는 삭제 문서는 다음 build에서 제거한다.
7. model ID, revision, 자료 snapshot hash를 함께 기록한다.

모델 재학습은 필요하지 않다. 공식 API 자료는 검수 후 인덱스만 갱신한다.
LoRA·QLoRA는 답변 방식 개선이 별도로 필요할 때만 검토한다.

교체 전 인덱스는 `snapshots`에 보존된다.
`HybridRetrievalService.rollback(source_snapshot)`에 정확한 snapshot hash를
전달하고, 복원 후 model ID와 revision 일치를 다시 검증한다.
