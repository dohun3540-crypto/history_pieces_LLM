# 공공누리 텍스트·문서 조사 보고서

조사일: 2026-07-30

## 판정 기준

- 개별 저작물 상세 화면에 공공누리 제0유형 또는 제1유형이 표시되어야 한다.
- 목포 근대역사에 직접 관련된 역사 설명 본문이 있어야 한다.
- 검색 결과나 목록 화면은 후보로 인정하지 않는다.
- `allowed_for_rag`는 정제·분할·검색·답변 인용이 허용되는 경우에만 `true` 후보로 판단한다.
- `allowed_for_training`은 공공누리 AI유형 또는 별도의 명시적 학습 허락이 없으면 `false`로 판단한다.

공공누리 유형별 공식 조건은 [공공누리 이용조건 안내](https://www.kogl.or.kr/info/license.do)를
기준으로 했다. 2026년에 신설된 AI유형은 인공지능 학습용 데이터 이용을 별도로 규정하므로,
기존 제0·1유형만으로 모델 학습을 승인하지 않는다.
[공공누리 AI유형 공식 안내](https://www.kogl.or.kr/news/noticeView.do?dataIdx=245)

## 조건을 모두 충족한 텍스트·PDF

**0건**

공공누리 상세 페이지에서 목포 관련 제1유형 자료는 발견했지만 이미지 중심이거나 역사 본문이
없었다. 실제 역사 본문이 있는 목포시·국가유산포털 페이지는 제3·4유형이거나 개별 라이선스가
표시되지 않아 승인 후보로 올리지 않았다.

## 확인된 근접 후보

| 자료 | 기관 | 형식 | 개별 표시 | 역사 본문 | RAG | 학습 | 판정과 다음 조치 |
|---|---|---|---|---|---|---|---|
| [사적 제289호 구 목포 일본영사관](https://www.kogl.or.kr/recommend/recommendDivView.do?division=img&recommendIdx=1759) | 국가유산청 | JPG 묶음 | 공공누리 제1유형 | 없음 | `false` | `false` | 사진은 재사용 가능하지만 텍스트 RAG 자료가 아니다. 이미지 메타데이터 컬렉션으로만 별도 검토 |
| [목포 양동교회](https://biz.mokpo.go.kr/tour/attraction/cultural_assets/list?idx=7490&mode=view) | 목포시 | HTML | 사진 출처 부분만 제1유형, 페이지 전체는 제4유형 | 있음 | `false` | `false` | 제1유형을 전체 본문에 확대 적용할 수 없다. 목포시에 본문 이용 허락 문의 |
| [목포근대역사관 1관](https://biz.mokpo.go.kr/tour/attraction/cultural_assets/list?idx=7449&mode=view) | 목포시 | HTML | 페이지 제4유형 | 있음 | `false` | `false` | 변경금지 조건이 정제·청크화와 충돌. 별도 허락 문의 |
| [목포근대역사관 2관](https://www.mokpo.go.kr/tour/attraction/area?idx=7451&mode=view) | 목포시 | HTML | 페이지 제4유형 | 있음 | `false` | `false` | 변경금지 조건이 정제·청크화와 충돌. 별도 허락 문의 |
| [목포 근대역사문화공간](https://www.heritage.go.kr/heri/cul/culSelectDetail.do?VdkVgwKey=79%2C07180000%2C36&pageNo=1_1_1_0) | 국가유산청 | HTML | 개별 표시 미확인 | 있음 | `false` | `false` | `copyright_status=unknown`. 국가유산청에 설명문 이용 범위 문의 |
| [구 목포 일본영사관](https://www.heritage.go.kr/heri/cul/culSelectDetail.do?ccbaCpno=1333602890000) | 국가유산청 | HTML | 개별 표시 미확인 | 있음 | `false` | `false` | `copyright_status=unknown`. 국가유산청에 설명문 이용 범위 문의 |
| [구 동양척식주식회사 목포지점 안내판](https://www.heritage.go.kr/heri/cul/culGuidePostDetail.do?ccbaCpno=2333601740000&ccgbGbtype=IND&ccgbGbtypeNo=1&pageNo=1_5_0_0) | 국가유산청 | HTML | 개별 표시 미확인 | 국·영문 있음 | `false` | `false` | 안내판 문안의 저작권자·공공누리 적용 여부 문의 |

## 사용 불가 또는 자동 수집 제외

- 공공누리 제3·4유형: 변경금지 때문에 정제, 청크 분할, 검색 문맥 구성 및 요약과 충돌한다.
- 사이트 전체 정책만 확인되고 개별 저작물 표시가 없는 자료: `copyright_status=unknown`.
- 한국사데이터베이스와 우리역사넷: 일반 사용자 에이전트에 대해 robots.txt가 전체 경로를
  금지하므로 자동 수집하지 않는다.
- 검색 결과, 시작 화면, 사진 제목만 있고 설명문이 없는 항목은 텍스트 RAG 후보가 아니다.

## 결론

경로 A에서 즉시 수집 가능한 텍스트 자료는 아직 없다. 우선 허락 문의 대상은 목포시 관광
상세 본문 2건과 국가유산청 상세 설명 3건이다. 허락 회신에는 저장, 정제, 청크화, 임베딩,
검색 결과 인용 및 공개 서비스 범위가 모두 포함되어야 한다.

