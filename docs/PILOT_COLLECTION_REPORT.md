# 파일럿 공식 자료 수집 품질 검사

- 검사일: 2026-07-30
- 검사 대상: `data/source_catalog/collected_sources.jsonl` 및
  `data/manifests/sources.jsonl`에 등록된 파일럿 문서
- 검사 원칙: 원문과 역사 사실을 수정·보충하지 않으며 `review_status`를 변경하지 않음

## 검사 결과 요약

| document_id | 제목 | 출처 기관 | 품질 판정 | 역사 내용 | RAG | 학습 |
|---|---|---|---|---|---|---|
| `auto-national_archives-1323bcbe9beea1f39fab` | 국가기록원&gt;메인 | 국가기록원 | 부적합 후보 | 없음 | 불가 | 불가 |

수집된 1건은 국가기록원의 역사 자료가 아니라 다른 화면으로 이동시키는 시작
HTML이다. 파일 저장과 텍스트 추출 자체는 성공했지만 RAG 문서로서 유효한 본문이
없으므로 사람이 검토하여 거절 여부를 결정해야 한다. 이 검사에서는 기존
`review_status=draft`를 그대로 유지했다.

## 문서별 상세 검사

### auto-national_archives-1323bcbe9beea1f39fab

| 검사 항목 | 결과 |
|---|---|
| document_id | `auto-national_archives-1323bcbe9beea1f39fab` |
| 제목 | 국가기록원&gt;메인 |
| 출처 기관 | 국가기록원 |
| 원본 URL | https://www.archives.go.kr/next/ |
| 원본 저장 여부 | 저장됨 — `data/raw/collected/national_archives/approved-archives-next.html` |
| 텍스트 추출 성공 여부 | 기술적으로 성공 — `data/extracted/collected/national_archives/approved-archives-next.txt` |
| 본문 길이 | 줄바꿈 포함 9자, 공백 제외 8자 |
| 실제 역사 내용 여부 | 아니요. 사이트 제목만 있고 역사 본문은 없음 |
| 메뉴·배너·푸터 과다 포함 여부 | 아니요. 메뉴·배너·푸터 자체가 추출되지 않았으나 유효 본문도 없음 |
| 중복 여부 | 중복 없음. catalog와 manifest에 각각 동일 document_id 1건만 존재 |
| 문자 깨짐 여부 | 추출문에는 깨짐 및 U+FFFD 대체 문자가 없음. 다만 원본은 HTML에서 EUC-KR을 선언하지만 실제 바이트는 UTF-8로 해석해야 정상이며, 인코딩 선언 불일치가 있음 |
| OCR 필요 여부 | 불필요. 텍스트 HTML이며 이미지·스캔 문서가 아님 |
| 저작권 상태 | `unknown` |
| RAG 사용 가능 여부 | 불가 — `allowed_for_rag=false`; 라이선스 불명확 및 역사 본문 없음 |
| 학습 사용 가능 여부 | 불가 — `allowed_for_training=false`; 라이선스 불명확 및 학습 가치가 있는 본문 없음 |
| review_status | `draft` 유지 |

원본 SHA-256은
`77d93472cebb8aa10b6060f2b5a11b48f56ba184c560bde6ddcc0306516b9f86`이며,
catalog 기록과 실제 파일이 일치한다.

### 누락되거나 미확정인 메타데이터

- `published_date`
- `author`
- `license_name`
- 자료별 `license_url` 또는 구체적인 이용 유형
- `attribution_text` 및 실제 출처표시 의무
- `period_start`, `period_end`, `historical_period`
- `people`, `places`, `organizations`, `events`, `keywords`
- 실제 역사 자료의 상세 URL
- 원문 페이지·기록물 식별자·소장 기록 정보
- 본문 언어를 확인할 충분한 내용

현재 `license_url`에는 국가기록원의 일반 저작권 정책 URL만 기록되어 있으며,
해당 문서의 구체적인 라이선스를 확인한 것으로 간주할 수 없다.

### 사람이 검토해야 할 항목

1. 시작 페이지가 아닌 실제 역사 기록물의 공식 상세 URL을 별도로 승인할지 결정
2. JavaScript가 가리키는 `/next/viewMainNew.do`를 새 수집 후보로 검토할지 결정
3. 실제 기록물별 공개 상태, 저작권자, 공공누리 유형과 재사용 조건 확인
4. HTML charset 선언과 실제 응답 바이트 불일치가 재현되는지 확인
5. 역사 본문이 없는 현재 문서를 `rejected`로 전환할지 결정

## 미수집 시도

| 출처 | 승인 URL | 결과 | 원인 | 저장·등록 여부 |
|---|---|---|---|---|
| OAK 국가리포지터리 | https://oak.go.kr/ | 실패 | 목록에 없던 `http://oak.go.kr/main/main.do`로 HTTP 302 이동하여 정책에 따라 중단 | 원본·추출문·metadata 모두 없음 |

OAK 항목은 문서가 생성되지 않았으므로 문서별 품질 검사 대상에는 포함하지
않았다. 리디렉션 URL을 자동 승인하거나 따라가지 않았다.

## 최종 판정

- 품질 검사 문서: 1건
- 역사 본문이 있는 문서: 0건
- RAG 사용 가능 문서: 0건
- 모델 학습 사용 가능 문서: 0건
- 중복 문서: 0건
- OCR 필요 문서: 0건
- 사람의 후속 검토가 필요한 문서: 1건
- `review_status` 변경: 없음

현재 파일럿 결과는 수집 안전장치가 목록 밖 이동을 차단하는지는 확인했지만,
목포 근대역사 RAG에 사용할 실제 콘텐츠를 확보한 결과는 아니다.
