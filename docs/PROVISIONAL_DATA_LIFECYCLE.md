# 임시 해커톤 자료 수명주기

흐름은 다음과 같다.

`source audit → dry-run(48/3 검증) → metadata prepare → 제한 GET → 텍스트 정제
→ 청크화 → hackathon 격리 인덱스 → 2026-08-31 재검토/비활성화`

원문과 청크는 각각 `data/provisional_hackathon/raw`,
`data/provisional_hackathon/processed`에 저장하며 Git에서 제외한다. Git에는
`manifests/sources.jsonl`의 출처·권리·제거 가능성 메타데이터만 남긴다.
인덱스는 `.runtime/indexes/hackathon`에 저장하여 production 인덱스와 물리적으로
분리한다.

```powershell
$env:APP_MODE = "hackathon"
python -m history_chatbot.provisional.cli dry-run
python -m history_chatbot.provisional.cli prepare
# 네트워크를 사용하는 명시적 단계
python -m history_chatbot.provisional.cli collect
python -m history_chatbot.provisional.cli rebuild
```

`collect`는 공식 상세 URL만 요청하고, 응답당 2MB·15초 timeout·요청 간 1초
대기를 적용한다. 실패는 유형만 기록하며 무한 재시도하지 않는다.

2026-07-30 파일럿 실행에서는 48건 중 7건의 텍스트 수집에 성공했고 41건은
공식 서버 접근 실패로 중단했다. 성공 자료는 중복 제거 후 26청크이며 readiness는
`hackathon_data_partial`이다. 실패 URL은 우회하거나 반복 요청하지 않았다.
