# 해커톤 임시 자료 정책

`provisional_hackathon`은 정식 승인 상태가 아니다. 공식 후보 51건 가운데
공공누리 제4유형 3건을 제외한 권리 확인 대기 48건을 비상업적 대학 산학
해커톤 시연에서만 제한적으로 검색하기 위한 격리 상태다.

- `rights_status=unconfirmed`
- `usage_scope=noncommercial_hackathon_demo`
- `allowed_for_rag=false`
- `allowed_for_training=false`
- `public_release_allowed=false`
- `expires_or_review_after=2026-08-31`

원문 전체 공개와 파인튜닝은 금지한다. 사진·이미지·도면·영상·첨부파일과
제3자 저작물은 수집하지 않는다. 답변에는 기관명, 자료명, 공식 URL과
“해커톤 시연용 공식 참고자료”를 표시한다. 직접 인용은 출처당 160자 이내,
전체 생성 답변은 안전 상한을 적용하며 가능한 한 요약·재구성한다.

development/test는 가상 fixture만, hackathon은 승인 자료와 이 임시 자료만,
production은 정식 `approved_for_rag` 자료만 사용할 수 있다. production 경로에
임시 자료 또는 임시 인덱스가 감지되면 로드와 인덱싱을 거부한다.

현재 실제 파일럿은 48건 메타데이터를 모두 추적하지만, 공식 상세 GET에 성공한
7건(중복 제거 후 26청크)만 격리 인덱스에 존재한다. 나머지 41건은 권리 상태와
별개로 접근에 실패했으며 검색에 포함되지 않는다.
