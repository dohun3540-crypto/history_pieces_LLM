# 검수 완료 자료의 RAG 인덱스 준비 가이드

이 계층은 사람 검수가 끝나고 RAG 사용이 허용된 문서의 기존 JSONL 청크를
검색 시스템 입력 형식으로 준비합니다. 임베딩 모델, 벡터 데이터베이스와 Llama
모델은 호출하지 않습니다.

## 허용 정책

다음 조건을 모두 충족해야 합니다.

- `review_status=reviewed`
- `allowed_for_rag=true`
- `source_reliability`가 `A` 또는 `B`
- `document_id`, `title`, `publisher`, `source_url` 존재
- `copyright_status`가 `unknown` 또는 `restricted`가 아님
- 출처표시가 필요하면 `attribution_text` 존재
- `reviewed_by`, `reviewed_at`, `verification_notes` 존재
- `local_path` 원본이 `data/raw` 아래에 존재
- `data/processed/<document_id>.jsonl`이 존재
- 모든 청크의 `document_id`와 `chunk_id`가 원 문서를 추적함

조건을 충족하지 못한 자료는 이유와 함께 제외됩니다. 승인 여부와 별개로
`allowed_for_rag=false`이면 절대 통과하지 않습니다.

## PowerShell 실행

저장소 루트에서 실행합니다.

```powershell
$env:PYTHONPATH = "src"

python -m history_chatbot.indexing.cli status
python -m history_chatbot.indexing.cli list-eligible
python -m history_chatbot.indexing.cli list-rejected
python -m history_chatbot.indexing.cli prepare
python -m history_chatbot.indexing.cli validate
```

기본 경로는 다음과 같습니다.

- 출처 manifest: `data/manifests/sources.jsonl`
- 원본 루트: `data/raw`
- 처리 청크: `data/processed`
- 인덱스 준비 결과: `data/index_ready`

각 명령에 `--manifest`, `--raw-root`, `--processed-dir`, `--output-dir`를
지정하여 테스트 fixture나 별도 환경을 사용할 수 있습니다.

검수 완료 문서가 없으면 오류로 취급하지 않고 다음 메시지를 출력합니다.

```text
현재 인덱싱 가능한 검수 완료 문서가 없습니다
```

## 출력

`prepare`는 다음 파일을 생성합니다.

- `data/index_ready/chunks.jsonl`
- `data/index_ready/index_manifest.json`

청크에는 다음 정보를 보존합니다.

- `document_id`, `chunk_id`, `chunk_index`, 본문과 문자 위치
- 제목, 발행기관, 원본 URL, 언어
- 저작권 상태, 라이선스, 출처표시 문구
- 검수자와 검수 시각, 신뢰도
- 인물, 장소, 기관, 사건, 키워드
- 시작·종료 연도와 역사 시기
- 페이지·section 정보
- 정규화한 청크 본문의 SHA-256

동일 문서 안에서 공백 차이만 있는 동일 본문 청크는 한 번만 기록합니다.
동일 `chunk_id`에 서로 다른 본문이 있으면 안전하지 않은 상태로 보고 중단합니다.

## 스냅샷·증분 처리·삭제 상태

`index_manifest.json`은 원본, 처리 청크와 문서 메타데이터의 SHA-256을
문서별로 기록합니다.

- 처음 보거나 해시가 바뀐 문서: `changed_document_ids`
- 이전 스냅샷과 같은 문서: `unchanged_document_ids`
- 이전에는 활성 상태였지만 삭제되거나 더 이상 허용되지 않는 문서:
  `tombstones`

이 정보는 향후 임베딩·벡터 DB 연결 시 변경 문서만 다시 처리하고 tombstone
문서를 실제 인덱스에서 제거하기 위한 기반입니다. 현재 단계에서는 벡터 인덱스를
직접 변경하지 않습니다.

## 운영 안전 원칙

- `data/index_ready`의 생성 결과는 Git에 커밋하지 않습니다.
- 테스트 fixture는 pytest 임시 디렉터리에서만 만듭니다.
- `prepare`는 원본, 처리 청크와 검수 manifest를 수정하지 않습니다.
- `validate`는 현재 eligibility 결과, 청크 필수 필드, 중복, 스냅샷 해시,
  tombstone 혼입을 다시 검사합니다.
- draft·rejected 또는 권리 미확인 문서가 하나라도 `chunks.jsonl`에 있으면
  검증 실패입니다.
