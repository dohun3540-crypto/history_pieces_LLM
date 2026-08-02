# 목포 역사 데이터 감사 보고서

> 자동 분류는 추정이며 역사성·OCR 오류·production 승인을 확정하지 않습니다.

## 1. 집계 기준

- fixture는 실제 역사 데이터에서 제외했습니다.
- provisional과 개발 검증용 자료는 현황에는 포함하되 production 사용 가능으로 보지 않았습니다.
- 권리 상태가 unknown/unconfirmed/pending/missing인 자료는 production 가능 수에서 제외했습니다.

## 2. 전체 집계

| 항목 | 수 |
|---|---:|
| manifest 전체 | 54 |
| fixture 제외 실제/후보 문서 | 52 |
| fixture 문서 | 2 |
| production 사용 가능 | 0 |
| 원문 파일 | 49 |
| 추출 텍스트 파일 | 1 |
| fixture 제외 chunk | 139 |
| index 문서 | 3 |
| index chunk | 6 |

## 3. 상태 및 권리

### 상태

- `draft`: 0
- `fixture`: 2
- `other_review_state`: 3
- `production_approved`: 0
- `production_rejected`: 1
- `provisional`: 48
- `reference_only`: 0
- `review_status_missing`: 0

### 권리 상태

- `pending_review`: 3
- `unconfirmed`: 48
- `unknown`: 1

## 4. 누락 필드

| 의미 필드 | 누락 |
|---|---:|
| `document_id` | 0 |
| `title` | 0 |
| `institution` | 0 |
| `source_url` | 0 |
| `published_date` | 52 |
| `accessed_at` | 0 |
| `license_status` | 0 |
| `usage_scope` | 4 |
| `review_status` | 0 |
| `production_approved` | 49 |
| `historical_period` | 4 |
| `topic_tags` | 1 |
| `place_tags` | 52 |
| `person_tags` | 52 |
| `raw_text_path` | 51 |
| `clean_text_path` | 52 |

필드 대응은 JSON의 `schema_mapping`과 `schema_field_assessment`에 기록했습니다. 현재 스키마는 `publisher`, `copyright_status`, `accessed_date`, `local_path`, `keywords`, `places`, `people`를 사용하며, 개발 레인은 별도 필드 집합을 사용합니다. `usage_scope`, `production_approved`, `clean_text_path`는 일반 스키마에 없습니다. 이번 감사에서는 스키마를 변경하지 않았습니다.

## 5. 중복과 불일치

- document_id: 0건
- source_url: 0건
- similar_titles: 47건
- identical_body_hash: 0건
- similar_body: 6건
- duplicate_chunk_id: 0건
- identical_chunk_body: 0건
- raw_without_manifest: 0건
- manifest_missing_raw_file: 0건
- manifest_without_raw_path_or_resolvable_rule: 3건
- manifest_without_extracted_text: 51건
- chunks_without_manifest_document: 0건
- manifest_without_chunks: 1건
- index_documents_not_in_manifest: 0건
- manifest_documents_not_in_index: 49건

## 6. 품질 경고

- 경고 문서: 1건
- 상세 항목은 JSON 보고서에서 확인합니다. 휴리스틱 경고는 수동 검토 대상으로만 사용합니다.

## 7. 주제별 커버리지

| 주제 | 문서 | chunk | 기관 | 승인 | provisional | 권리 미확정 | 최대 출처 비중 | 부족 | 추가 권고 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 목포 개항 | 9 | 5 | 3 | 0 | 9 | 9 | 67% | 예 | 3 |
| 목포 해관 | 1 | 1 | 1 | 0 | 1 | 1 | 100% | 예 | 5 |
| 외국인 거류지·조계지 | 1 | 1 | 1 | 0 | 1 | 1 | 100% | 예 | 5 |
| 구 일본영사관 | 1 | 2 | 1 | 0 | 0 | 1 | 100% | 예 | 5 |
| 동양척식주식회사 목포지점 | 5 | 6 | 2 | 0 | 4 | 5 | 80% | 예 | 5 |
| 목포 근대역사문화공간 | 4 | 4 | 2 | 0 | 2 | 4 | 50% | 예 | 3 |
| 근대 항만과 철도 | 3 | 6 | 2 | 0 | 3 | 3 | 67% | 예 | 3 |
| 일제강점기 목포의 산업과 도시 변화 | 22 | 35 | 3 | 0 | 22 | 22 | 91% | 예 | 3 |

## 8. 다음 수집 배치

- 1차 배치는 부족도가 높은 핵심 주제 중심 10~20건을 권고합니다.
- 기관별 최대 3~5건으로 제한하고, 주제별 최소 2개 독립 기관을 목표로 합니다.
- 로그인·캡차·유료벽·이용조건 불명확 자료는 제외합니다.
- 원문과 추출 텍스트를 분리하고 기존 정책의 draft 상태를 유지합니다.
- 사람 검토 전 `production_approved`를 변경하지 않습니다.

## 9. 수동 검토 필요

- 역사적 사실성과 목포 직접 관련성
- OCR/인코딩 휴리스틱 경고
- 권리 상태가 unknown, unconfirmed 또는 pending_review인 전 자료
- manifest와 파일·chunk·index 불일치
