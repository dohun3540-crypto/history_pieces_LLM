# 임시 자료 제거와 롤백

```powershell
python -m history_chatbot.provisional.cli list
python -m history_chatbot.provisional.cli remove --source-id "SOURCE_ID"
python -m history_chatbot.provisional.cli remove --institution "독립기념관"
python -m history_chatbot.provisional.cli purge-all
python -m history_chatbot.provisional.cli expire
python -m history_chatbot.provisional.cli rebuild
```

제거는 manifest를 비활성화하고 청크를 필터링한 뒤 BM25와 dense 저장소를 함께
재생성한다. 해커톤 세션 캐시는 무효화한다. 인덱스 교체는 임시 파일과 원자적
교체를 사용하며 기존 인덱스는 snapshot으로 보존한다. 실패하면 manifest와
청크를 이전 상태로 복원한다.

감사 로그에는 source_id, 제거 시각, 사유만 남긴다. 원문·인덱스를 삭제해도
Git의 출처 메타데이터와 감사 정책은 유지할 수 있다. 기관 요청이나 권리 문제가
발생하면 source_id 또는 기관 단위로 즉시 제거하며, 대회 종료 후 일괄
재검토하거나 `purge-all`한다.

