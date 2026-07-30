# 공식 API 대기 상태

한국관광공사 국문 관광정보 API는 2026년 8월 2일 이후 자격증명을 설정해 파일럿
조회할 예정이다.

- 키 없음: `pending_credentials`, 네트워크 호출 금지
- 키 있음: `ready_for_dry_run`
- 키는 Git에서 제외된 로컬 `.env` 또는 프로세스 환경변수에만 설정

```powershell
$env:TOUR_API_SERVICE_KEY="<로컬에서만 설정>"
$env:TOUR_API_BASE_URL="https://apis.data.go.kr/B551011/KorService2"
python -m history_chatbot.collectors.cli tour-api dry-run
```

키를 추가해도 코드 변경은 필요 없다. `dry-run`은 파일이나 manifest를 변경하지
않는다. collect 결과도 권리 검증 전까지 `draft`, `allowed_for_rag=false`,
`allowed_for_training=false`이다.
