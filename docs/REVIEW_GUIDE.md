# 역사 자료 검수 가이드

검수자는 자동 정제 결과를 사실 판정으로 간주하지 않고 원문 및 공신력 있는 출처와
대조합니다.

## 체크리스트

- 제목, 저자, 발행기관, 발행일, 원문 URL이 실제 출처와 일치하는가?
- 저작권 상태와 라이선스 근거가 기록되어 있는가?
- RAG, 모델 학습, 재배포 허용 범위를 각각 확인했는가?
- 요구되는 출처 표시문이 정확하고 완전한가?
- 연도와 날짜가 원문 및 다른 신뢰할 수 있는 자료와 일치하는가?
- 인물의 이름, 한자·이명, 역할을 과도하게 단정하지 않았는가?
- 지명과 당시 행정구역, 현재 명칭을 혼동하지 않았는가?
- 기관과 단체의 정식 명칭 및 존속 시기를 확인했는가?
- 사건의 발생 시점, 장소, 당사자와 인과관계를 출처가 뒷받침하는가?
- 직접 인용은 원문과 일치하고 페이지·절 등 위치를 추적할 수 있는가?
- 서로 다른 해석이나 논쟁이 있는 내용을 한 관점으로 확정하지 않았는가?
- OCR의 숫자, 한자, 고유명사, 줄바꿈, 탈락·중복 문자를 원문과 대조했는가?
- 정제 과정에서 역사적 고유명사, 연도, 지명 또는 의미가 바뀌지 않았는가?
- 청크 경계가 문맥이나 인용을 오해하게 만들지 않는가?

## 검수 기록

검수 전 자료는 `draft` → `extracted` → `cleaned` → `metadata_added` 흐름으로
관리합니다. 최종 승인 시 manifest에 다음을 기록합니다.

- `review_status`: `reviewed`
- `reviewed_by`: 조직에서 식별 가능한 검수자명 또는 검수자 ID
- `reviewed_at`: 시간대가 포함된 ISO 8601 시각
- `verification_notes`: 확인한 출처, 판단 근거, 남은 한계

중대한 사실 오류, 출처 불명, 권리 문제를 해결할 수 없으면 `rejected`로 기록하고
서비스 색인 및 `data/reviewed` 승격을 금지합니다. 수정 후 재검수가 필요하면
`metadata_added`로 되돌려 검수 이력을 별도로 보존합니다.

## 검수 CLI

저장소 루트의 Windows PowerShell에서 실행합니다.

```powershell
$env:PYTHONPATH = "src"

python -m history_chatbot.ingestion.cli review show `
  --document-id "mokpo-source-001"

python -m history_chatbot.ingestion.cli review approve `
  --document-id "mokpo-source-001" `
  --reviewer "검수자" `
  --notes "원본 URL, 기관, 권리 조건을 확인함"

python -m history_chatbot.ingestion.cli review reject `
  --document-id "mokpo-source-001" `
  --reviewer "검수자" `
  --reason "역사 본문이 없고 자료 적합성을 확인할 수 없음"
```

다른 manifest 또는 감사 로그를 사용할 때는 각 하위 명령에 `--manifest`와
`--audit-log`를 지정합니다. 기본 경로는 각각
`data/manifests/sources.jsonl`과 `data/manifests/review_audit.jsonl`입니다.

승인 시 다음 조건을 모두 확인합니다.

- 필수 메타데이터, 유효한 HTTP(S) 원본 URL과 출처 기관이 존재함
- 원본 파일이 `data/raw` 아래에 실제로 존재함
- `copyright_status`가 `unknown` 또는 `restricted`가 아님
- 출처표시가 필요하면 `attribution_text`가 존재함

승인과 거절은 `reviewed_by`, `reviewed_at`, `verification_notes`를 manifest에
기록하고 append-only JSONL 감사 로그에 이전·이후 상태와 사유를 남깁니다.
원본 파일은 읽거나 수정하지 않습니다. 승인 명령은 `allowed_for_rag`나
`allowed_for_training`을 변경하지 않습니다.
