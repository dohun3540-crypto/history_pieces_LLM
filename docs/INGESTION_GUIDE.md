# 자료 등록·정제 가이드

이 파이프라인은 외부 사이트를 크롤링하거나 다운로드하지 않습니다. 원문 파일과
권리 정보를 담당자가 직접 확보하고 확인해야 합니다.

## 1. 원문 준비

UTF-8 TXT 또는 Markdown 파일을 `data/raw` 아래에 둡니다. PDF는 텍스트 추출
인터페이스만 제공하며 기본 의존성에는 PDF 라이브러리가 없습니다. HTML과 웹페이지
수집은 지원하지 않습니다. 대용량 원문, PDF, 데이터셋은 Git에 커밋하지 않습니다.

## 2. 메타데이터 작성

아래는 역사 자료가 아닌 형식 예시입니다. 날짜는 `YYYY-MM-DD`, 검수 시각은 ISO
8601을 사용합니다.

```json
{
  "document_id": "mokpo-source-001",
  "title": "자료의 실제 제목",
  "source_type": "public_archive",
  "publisher": "실제 발행 기관",
  "author": "확인된 저자 또는 빈 문자열",
  "source_url": "https://공식-출처.example/item/001",
  "local_path": "data/raw/mokpo-source-001.txt",
  "published_date": "YYYY-MM-DD",
  "accessed_date": "2026-07-30",
  "language": "ko",
  "license_name": "확인된 라이선스명",
  "license_url": "https://라이선스-원문.example",
  "copyright_status": "open_license",
  "allowed_for_rag": true,
  "allowed_for_training": false,
  "redistribution_allowed": false,
  "attribution_required": true,
  "attribution_text": "요구되는 정확한 출처 표시문",
  "notes": "권리 확인 근거와 주의사항",
  "review_status": "draft",
  "reviewed_by": "",
  "reviewed_at": "",
  "period_start": null,
  "period_end": null,
  "historical_period": "",
  "people": [],
  "places": [],
  "organizations": [],
  "events": [],
  "keywords": [],
  "source_reliability": "",
  "verification_notes": ""
}
```

인물·장소·기관·사건은 자동으로 사실 확정하지 않습니다. 확인된 수동 입력을
기본으로 하고, 자동 추출을 추가하더라도 검수 후보로만 취급합니다.

## 3. PowerShell 명령

저장소 루트에서 다음을 실행합니다.

```powershell
$env:PYTHONPATH = "src"

python -m history_chatbot.ingestion.cli register `
  --manifest "data/manifests/sources.jsonl" `
  --metadata "C:\작업\source-metadata.json"

python -m history_chatbot.ingestion.cli process `
  --manifest "data/manifests/sources.jsonl" `
  --document-id "mokpo-source-001"

python -m history_chatbot.ingestion.cli validate `
  --manifest "data/manifests/sources.jsonl" `
  --document-id "mokpo-source-001"

python -m history_chatbot.ingestion.cli list `
  --manifest "data/manifests/sources.jsonl"
```

`register`는 권리·필수 메타데이터를 검증하고 JSONL manifest에 추가합니다.
`process`는 원문을 `data/extracted`에 보존하고, 원문과 정제문을 구분해 정제한 뒤
청크를 `data/processed/<document_id>.jsonl`에 기록합니다. 처리 상태는
`metadata_added`가 되며 자동으로 `reviewed`가 되지 않습니다.

## 4. 검수와 승격

`REVIEW_GUIDE.md`에 따라 원문, 정제 로그, 청크와 메타데이터를 확인합니다. 검수자와
검수 시각을 기록하고 manifest 상태를 `reviewed`로 바꾼 뒤 다시 `validate`하여
서비스 색인 가능 여부를 확인합니다. 검수된 결과만 `data/reviewed`에 복사합니다.
원문 재배포가 금지된 자료는 청크 공개 범위도 별도로 검토해야 합니다.
