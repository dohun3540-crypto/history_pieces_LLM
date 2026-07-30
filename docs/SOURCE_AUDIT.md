# 공식 출처 Seed 감사 기록

감사일: 2026-07-30

범위: `data/source_catalog/seed_sources.json`에 등록된 기관·정책·API 정보

제외: 원문, 검색 결과, 첨부파일 및 데이터셋 수집

## 판정 기준

- `allowed`: robots.txt와 이용정책, 인증 조건, 수집 범위가 확인되어 설정 범위에서
  자동 수집을 진행할 수 있음
- `manual_review`: 공식 출처이지만 robots.txt 내용, 개별 저작물 권리, API 적용 범위
  또는 이용조건 중 하나 이상을 사람이 확인해야 함
- `blocked`: 정책 또는 접근 통제가 자동 수집을 금지하거나 우회가 필요한 상태
- `unknown`: 공식 근거로 확인하지 못했으며 추정하지 않음

이번 감사에서는 robots.txt의 표준 주소를 기록했지만 이 실행 환경에서 본문을
검증하지 못했습니다. 따라서 `allowed`로 승인한 출처는 없으며 7개 출처 모두
`manual_review`입니다. 실제 수집 전 운영자가 robots.txt와 정책 변경 여부를 다시
확인해야 합니다.

## 감사 결과 요약

| ID | 공식 기본 도메인 | robots.txt | 정책·저작권 | 라이선스 표시 확인 위치 | 공식 API | 인증·신청 | 판정 |
|---|---|---|---|---|---|---|---|
| `heritage_portal` | [heritage.go.kr](https://www.heritage.go.kr/) | `https://www.heritage.go.kr/robots.txt` (내용 `unknown`) | 정확한 정책 본문 URL `unknown` | 사이트 하단 저작권정책 링크는 확인, 개별 자료 위치 `unknown` | 기관 API는 있음. 현재 목적에 맞는 범용 포털 검색 API는 `unknown` | API별 공공데이터포털/제공기관 절차가 달라 수동 확인 | `manual_review` |
| `history_database` | [db.history.go.kr](https://db.history.go.kr/) | `https://db.history.go.kr/robots.txt` (내용 `unknown`) | [국사편찬위원회 자료 이용 안내](https://contents.history.go.kr/front/about/use.do) | 개별 이미지·자료의 출처와 이용조건, 원 소장처 조건 | 별도 [역사지리정보DB API](https://hgis.history.go.kr/api/app.do)는 있음. 한국사DB 전체 API는 `unknown` | 신청·약관 동의·관리자 승인, 1년 | `manual_review` |
| `mokpo_city` | [mokpo.go.kr](https://www.mokpo.go.kr/) | `https://www.mokpo.go.kr/robots.txt` (내용 `unknown`) | [목포시 저작권정책](https://youth.mokpo.go.kr/www/operation_guide/copyright), [이용약관](https://m.mokpo.go.kr/www/operation_guide/member_agree) | 개별 게시물 하단 공공누리 유형 마크·문구 | 목포 근대역사 자료 API `unknown` | `unknown` | `manual_review` |
| `national_archives` | [archives.go.kr](https://www.archives.go.kr/) | `https://www.archives.go.kr/robots.txt` (내용 `unknown`) | [국가기록원 저작권정책](https://www.archives.go.kr/next/neworgan/copyrightProtection.do) | 개별 저작물의 공공누리 유형과 정책 페이지의 유형별 조건 | [나라기록물 검색 OpenAPI](https://contents.archives.go.kr/next/newsearch/openAPI01.do) 있음 | 공공데이터포털 가입·활용신청·인증키, 기본 일 1,000건 미만 | `manual_review` |
| `public_nuri` | [kogl.or.kr](https://www.kogl.or.kr/) | `https://www.kogl.or.kr/robots.txt` (내용 `unknown`) | [저작권정책](https://www.kogl.or.kr/etc/copyright.do), [유형·이용조건](https://www.kogl.or.kr/info/license.do) | 저작물 상세의 공공누리 유형 마크·사용 조건 | 공개 API 문서 `unknown` | API 키 관련 공식 Q&A는 있으나 공개 신청 절차 `unknown` | `manual_review` |
| `oak` | [oak.go.kr](https://oak.go.kr/) | `https://oak.go.kr/robots.txt` (내용 `unknown`) | 포털 전체 약관 `unknown`; [KJCI](https://copyright.oak.go.kr/)에서 학술지별 정책 확인 | 개별 리포지터리 메타데이터·원문 접근 조건 및 KJCI | [OAK Open API](https://oak.go.kr/about/aboutOak.do?menuSeq=93) 있음 | 기관명·기관 IP·이메일 등을 제출해 신청, 비영리 활용 대상 | `manual_review` |
| `kci` | [kci.go.kr](https://www.kci.go.kr/kciportal/) | `https://www.kci.go.kr/robots.txt` (내용 `unknown`) | 포털 전체 저작권 이용약관 `unknown`; [KJCI](https://copyright.oak.go.kr/)에서 학술지별 정책 확인 | 논문 상세의 원문 제공 여부·학술지 저작권 정책 링크 | [KCI Open API](https://kci.go.kr/kciportal/po/openapi/openApiList.kci) 있음 | KCI 인증키 신청·발급 필요 | `manual_review` |

## 출처별 상세 판단

### 국가유산포털

국가유산청의 공공데이터 API가 존재하는 것은 확인했습니다. 다만 확인된 예시는
문화유산 공간정보 등 개별 데이터셋이며, 목포 근대역사 후보를 포털 전체에서 찾는
범용 검색 API로 검증되지 않았습니다. seed의 `api_url`은 비워 두고 API 적용 범위와
인증 절차를 사람이 확인하도록 했습니다. 기존에 기록했던 저작권정책 URL도 공식
본문으로 검증하지 못해 `unknown`으로 변경했습니다.

### 한국사데이터베이스·국사편찬위원회

국사편찬위원회 이용 안내는 소장 자료가 아닌 경우 원 권리자 또는 원 소장처에
이용 가능 여부를 문의하도록 안내합니다. 확인된 Open API는 별도
역사지리정보DB이며 신청과 관리자 승인이 필요합니다. 이를 한국사데이터베이스
전체 원문 API로 확대 해석하지 않습니다.

### 목포시청·목포문화관광

목포시 저작권정책은 공공누리 마크가 붙은 자료만 표시 유형의 범위에서 이용하고,
마크가 없는 자료는 담당자와 사전 협의하도록 안내합니다. 따라서 기관 페이지라는
이유만으로 자동 허용하지 않습니다. 이 감사에서 목포 근대역사 전용 공식 API는
확인되지 않았습니다.

### 국가기록원

나라기록물 검색 OpenAPI와 인증키 발급 절차가 명시되어 있습니다. API는 검색
메타데이터 접근 수단이며 원문 재사용 허가가 아닙니다. 개별 기록물의 공개 상태와
공공누리 유형을 별도 확인해야 합니다.

### 공공누리

유형 안내에서 제0~4유형과 AI유형의 조건 및 표시 위치를 확인할 수 있습니다. 각
저작물에 부착된 유형을 기준으로 판단해야 합니다. 공식 Q&A 목록에서 API 키 문의는
확인했지만 공개 API 문서와 신청 절차는 확인하지 못해 API 상태를 `unknown`으로
유지했습니다.

### OAK

OAK Open API는 REST 기반이며 신청 양식에 기관명, 기관 IP, 이메일 등을 요구하고
비영리 활용 대상을 명시합니다. OAK에서 접근 가능한 논문이라고 해서 RAG·학습·재배포
권한이 자동으로 생기지 않으므로 리포지터리·논문별 라이선스와 KJCI 정책을 확인해야
합니다.

### KCI

KCI는 논문·인용정보 Open API와 인증키 신청을 제공합니다. 이는 서지·인용
메타데이터 제공이며 개별 논문 원문의 저작권 허가가 아닙니다. 논문 상세와 KJCI에서
발행기관·학술지별 정책을 확인해야 합니다.

## 운영 전 필수 확인

1. 브라우저 또는 승인된 운영 환경에서 각 robots.txt 본문과 대상 경로 허용 여부를
   확인합니다.
2. `manual_review`가 `allowed`로 변경되기 전에는 자동 수집을 실행하지 않습니다.
3. API를 사용할 경우 해당 API의 범위, 인증키, 일일 한도, 이용허락을 별도로
   기록하고 `api_url`을 출처별 어댑터 구현 후 설정합니다.
4. 공공누리·OA 표시는 문서별로 저장하며 기관 전체 정책으로 대체하지 않습니다.
5. `unknown` 항목은 담당 기관 확인 또는 공식 문서 발견 전까지 사용 금지로
   유지합니다.
