# 공식 자료 후보 수집 정책

## 안전 원칙

- 공식 API가 seed에 확인·등록되어 있으면 HTML보다 API를 우선합니다.
- 허용 도메인과 그 하위 도메인 외의 URL은 요청하지 않습니다.
- 요청 전에 robots.txt를 확인하며, 읽을 수 없거나 금지된 경우 안전하게 중단합니다.
- 출처별 페이지·결과 상한, 요청 간 지연, 타임아웃과 제한된 재시도를 적용합니다.
- 식별 가능한 전용 User-Agent를 사용합니다.
- 로그인, 캡차, 유료벽, 접근 제어를 우회하지 않습니다.
- 외부 링크를 따라 다른 도메인으로 확장하거나 무제한 순회하지 않습니다.
- 오류가 나면 해당 출처를 중단하고 이미 저장된 후보와 오류를 보고합니다.

사이트 약관이나 운영자 요청이 코드 설정보다 엄격하면 사이트 정책을 우선합니다.
수집기를 운영하기 전 정책 URL과 robots.txt의 변경 여부를 다시 확인합니다.

## 저장과 추적

응답 원본 바이트는 `data/raw/collected/<source_id>`, 사람이 읽을 수 있게 추출한
텍스트는 `data/extracted/collected/<source_id>`에 별도로 저장합니다. PDF/OCR
기능을 향후 추가하더라도 원본을 덮어쓰지 않고 OCR 결과를 별도 경로로 기록합니다.
자동 OCR은 역사적 고유명사, 연도, 지명이나 사실을 보정하지 않습니다.

후보 catalog에는 URL, 제목, 기관, 발행일(페이지에서 신뢰성 있게 확인된 경우),
접근일, 라이선스 상태, 원본·추출·OCR 경로와 SHA-256을 기록합니다. URL 정규화,
내용 해시, 높은 제목 유사도로 동일·유사 후보를 제외합니다.

## 권리 및 검수 격리

자동 수집 결과는 항상 다음 값으로 시작합니다.

- `review_status=draft`
- `copyright_status=unknown`
- `allowed_for_rag=false`
- `allowed_for_training=false`
- `redistribution_allowed=false`

공식 사이트의 공개 페이지라는 사실만으로 권리를 허용하지 않습니다. 검수자는
저작물별 라이선스명, 정책 URL, 출처 표시문, RAG·학습·재배포 가능 범위를 각각
확인해야 합니다. A/B 등급은 검수 우선 후보 자격일 뿐 이용 허가가 아닙니다.
기존 ingestion 검증기를 통과하고 사람이 `reviewed`로 승인하기 전에는 서비스
색인이나 학습 데이터에 포함하지 않습니다.

## 운영 예시

PowerShell에서 저장소 루트를 현재 디렉터리로 둡니다.

```powershell
$env:PYTHONPATH = "src"

# 기본값: 예정 URL과 이유만 출력하며 네트워크 요청 없음
python -m history_chatbot.collectors.cli `
  --seed "data/source_catalog/seed_sources.json" `
  --output "data/source_catalog/collected_sources.jsonl" `
  --source-id "national_archives" `
  --query "목포 해관" `
  --dry-run

# 실제 파일럿 실행은 명시적으로 요청
python -m history_chatbot.collectors.cli `
  --source-id "national_archives" `
  --query "목포 해관" `
  --execute
```

실행 전에 해당 사이트의 robots.txt와 정책을 사람이 확인합니다. 수집 원본과 추출
텍스트는 Git에 커밋하지 않으며, 후보 catalog도 로컬 운영 데이터로 취급합니다.

## 파일럿 실행 안전장치

- `collection_status=allowed`가 아닌 `manual_review`, `blocked`, `unknown` 출처는
  네트워크 요청 전에 건너뜁니다.
- `robots_verification=verified`가 아닌 출처도 네트워크 요청 전에 건너뜁니다.
- 전체 실행은 최대 10건, 출처별 최대 2건이며 CLI 인자로 늘릴 수 없습니다.
- 기본 CLI 동작은 dry-run입니다. `--execute`가 있어야만 네트워크 요청을 합니다.
- 로그인 URL·비밀번호 입력 폼, 캡차, 자동입력 방지, 유료회원·구독 장벽을 감지하면
  해당 출처 또는 상세 페이지를 중단하거나 건너뜁니다.
- 수집 결과는 강제로 `review_status=draft`가 됩니다.
- 라이선스가 `unknown`이면 RAG·학습 사용 플래그를 모두 `false`로 덮어씁니다.
- 파일럿 실행기는 ingestion 처리나 RAG 색인 함수를 호출하지 않습니다.
