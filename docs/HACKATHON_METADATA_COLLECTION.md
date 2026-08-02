# 해커톤 metadata/excerpt 전용 수집기

이 도구는 registry에 승인된 국가기록원 공개 HTML 상세 페이지만 대상으로 합니다. 전체 HTML, PDF, 이미지, 첨부파일은 저장하지 않으며 신규 레코드는 항상 권리 미확정 `provisional_hackathon` 상태로 격리됩니다.

먼저 `--dry-run`으로 후보, 중복 검사, 요청·저장 예정 경로를 확인합니다. dry-run은 네트워크를 호출하거나 파일을 만들지 않습니다. 실제 `--execute`는 별도 승인을 받은 뒤에만 사용합니다.

```powershell
$env:PYTHONPATH = "src"
python scripts/collect_hackathon_metadata.py `
  --manifest data/provisional_hackathon/manifests/sources.jsonl `
  --extracted-dir data/provisional_hackathon/extracted `
  --candidate archives-cja0002271-0027148187 `
  --candidate archives-cja0002271-overview `
  --max-items 2 --delay-seconds 1.2 --timeout-seconds 15 `
  --max-response-bytes 1048576 --dry-run
```

실행 시 모든 응답을 검증한 다음 임시 영역에서 추출 파일과 manifest를 준비합니다. 기존 manifest 바이트는 유지하고 신규 행만 append합니다. Windows의 여러 파일 교체는 단일 파일시스템 트랜잭션이 아니므로, manifest 교체 실패 시 이번 실행에서 생성한 추출 파일을 삭제하는 보상 rollback을 수행합니다.
