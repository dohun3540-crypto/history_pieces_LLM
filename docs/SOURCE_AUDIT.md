# 공식 출처 수동 검토 보고서

- 검토일: 2026-07-30
- 대상: `data/source_catalog/seed_sources.json`에 등록된 7개 출처
- 범위: 공식 도메인, robots.txt, 이용정책, 라이선스 표시, API, HTML/PDF 접근 및 이용 가능성
- 제외: 원문 수집, 대량 요청, 로그인, API 신청 및 인증키 발급

이 보고서에서 `가능`은 확인한 대표 공개 경로에 한정한다. HTML 또는 PDF에 접속할 수 있다는 사실만으로 저작권상 RAG·학습 이용이 허용되지는 않는다. 문서별 라이선스와 권리자를 확인할 수 없는 경우는 `unknown`이며, 프로젝트 정책상 RAG와 모델 학습 모두 금지한다.

robots.txt는 검토일에 `MokpoHistoryRAG-Audit/0.1` User-Agent로 각 공식 URL을 1회 직접 조회했다. robots.txt는 접근 제어와 저작권 허락을 대신하지 않으며, 향후 수집 시 다시 확인해야 한다.

## 요약

| 출처 | 공식 도메인 | robots.txt | 공식 API | HTML 수집 | PDF 다운로드 | RAG | 모델 학습 | 상태 |
|---|---|---|---|---|---|---|---|---|
| 국가유산포털 | 확인 | 접근 가능, 일부 경로 금지 | 제한된 개별 API 확인 | 조건부 가능 | `unknown` | 문서별 `unknown` | 문서별 `unknown` | `manual_review` |
| 한국사데이터베이스 | 확인 | 일반 봇 전체 금지 | 대상 전체 API는 `unknown` | 불가 | 자동 다운로드 불가 | 자동 수집 불가 | 자동 수집 불가 | `blocked` |
| 목포시청·목포문화관광 | 확인 | HTTP 400, 규칙 확인 실패 | `unknown` | `unknown` | `unknown` | 문서별 `unknown` | 문서별 `unknown` | `manual_review` |
| 국가기록원 | 확인 | 접근 가능, 금지 규칙 없음 | 확인 | 조건부 가능 | 대표 공개 첨부 확인 | 문서별 재확인 | 문서별 재확인 | `allowed` |
| 공공누리 | 확인 | 접근 가능, 일부 봇/경로 규칙 | 공개 API 문서 `unknown` | 조건부 가능 | 대표 공개 PDF 확인 | 표시 유형별 가능 | 표시 유형별 가능 | `manual_review` |
| OAK | 확인 | 접근 가능, 일부 경로 금지 | 확인 | 조건부 가능 | 대표 공개 PDF 확인 | 문서별 재확인 | 문서별 재확인 | `allowed` |
| KCI | 확인 | 일반 봇 전체 금지 | 확인 | 불가 | 자동 다운로드 불가 | 자동 수집 불가 | 자동 수집 불가 | `blocked` |

이번 판정에서 국가기록원과 OAK를 `allowed`, 한국사데이터베이스와 KCI를
`blocked`로 변경했다. 나머지 3개 출처는 `manual_review`로 유지했다. `allowed`는
출처 탐색을 허용한다는 뜻이며, 개별 문서의 RAG·학습 이용을 자동 허용하지 않는다.

## 1. 국가유산포털

- 공식 도메인: **확인** — 국가유산청의 [국가유산포털](https://www.heritage.go.kr/)
- robots.txt: **접근 가능** — [robots.txt](https://www.heritage.go.kr/robots.txt)
  - `User-agent: *`
  - `/heri/unified/` 금지
  - 사이트맵 주소 명시
- 이용약관·저작권 정책: 정확한 공식 본문 URL은 **`unknown`**
- 공공누리·오픈라이선스 표시: 사이트 하단에 저작권 정책 안내가 있으나 개별 자료의 표시 위치와 조건은 **`unknown`**
- 공식 API: **일부 존재** — 공공데이터포털의 [국가유산청 문화재 공간정보 서비스](https://www.data.go.kr/data/3070426/openapi.do)
  - 국가유산포털 전체 본문 검색·다운로드 API인지는 확인되지 않았다.
  - 인증·신청 조건은 해당 API별 공공데이터포털 안내를 따라야 하며 이번 검토에서는 `unknown`
- HTML 수집: **조건부 가능** — `/heri/unified/`를 제외한 대표 공개 경로는 robots.txt에서 명시적으로 금지되지 않았다.
- PDF 다운로드: **`unknown`**
- 인증·캡차·유료벽: 대표 공개 페이지에는 확인되지 않았으나 전체 경로는 **`unknown`**
- RAG 사용: **`unknown`** — 개별 자료의 라이선스 확인 전 금지
- 모델 학습 사용: **`unknown`** — 개별 자료의 명시적 허락 확인 전 금지

**allowed 전환 후보:** 조건부 후보. 금지 경로를 제외하고, 정확한 저작권 정책과 개별 자료 라이선스를 먼저 확인한 소수 HTML 상세 페이지만 별도 허용 목록으로 등록할 수 있다.

## 2. 한국사데이터베이스·국사편찬위원회

- 공식 도메인: **확인** — 국사편찬위원회의 [한국사데이터베이스](https://db.history.go.kr/)
- robots.txt: **접근 가능** — [robots.txt](https://db.history.go.kr/robots.txt)
  - Yeti, Daum, Googlebot, bingbot에는 `/` 허용
  - 그 밖의 `User-agent: *`에는 `/` 전체 금지
  - 현재 프로젝트의 일반 수집기 User-Agent에는 전체 금지가 적용된다.
- 이용약관·저작권 정책: [국사편찬위원회 자료 이용 안내](https://contents.history.go.kr/front/about/use.do)
  - 다른 기관·개인이 소장하거나 권리를 가진 자료는 원 소장처 또는 권리자의 조건을 따르도록 안내한다.
- 공공누리·오픈라이선스 표시: 개별 이미지·자료 화면의 출처와 이용조건에서 확인해야 하며 일괄 라이선스는 아님
- 공식 API: **부분 확인** — [역사지리정보DB API](https://hgis.history.go.kr/api/app.do)
  - 신청, 약관 동의, 관리자 승인 필요
  - 이용 기간 1년으로 안내
  - 한국사데이터베이스 전체 원문 API는 **`unknown`**
- HTML 수집: **불가** — 현재 User-Agent는 robots.txt의 전체 금지 대상
- PDF 다운로드: 웹 UI에서 간행물 PDF 제공은 [공식 간행물 목록](https://db.history.go.kr/diachronic/publication/list.do)에서 확인되지만, 자동 다운로드는 robots.txt 때문에 **불가**
- 인증·캡차·유료벽: 공개 열람 페이지의 로그인·캡차·유료벽은 확인되지 않았으나 자동 수집 금지와 별개
- RAG 사용: **자동 수집 불가**. 사람이 적법하게 확보한 파일도 문서별 권리 검토가 필요
- 모델 학습 사용: **자동 수집 불가**. 별도 명시 허락 없이는 금지

**판정: `blocked`.** robots.txt 정책이 바뀌거나 국사편찬위원회의 명시적 허가를
받기 전에는 HTML/PDF 수집기를 허용하지 않는다. 승인받은 API가 목적과 범위에
맞는 경우 API 전용 출처로 별도 검토한다.

## 3. 목포시청·목포문화관광

- 공식 도메인: **확인** — [목포시청](https://www.mokpo.go.kr/)과 동일 등록 도메인의 목포문화관광
- robots.txt: **접근 불가** — [robots.txt](https://www.mokpo.go.kr/robots.txt) 요청이 HTTP 400 `Request Blocked`로 응답하여 주요 규칙은 **`unknown`**
- 이용약관·저작권 정책:
  - [목포시 저작권 정책](https://youth.mokpo.go.kr/www/operation_guide/copyright)
  - [목포시 이용약관](https://m.mokpo.go.kr/www/operation_guide/member_agree)
- 공공누리·오픈라이선스 표시: 개별 게시물 하단의 공공누리 유형 표시에서 확인. 표시가 없는 자료는 담당자와 사전 협의가 필요하다고 저작권 정책이 안내한다.
- 공식 API: 목포 근대역사 자료용 공식 API는 **`unknown`**
- HTML 수집: **`unknown`** — 공개 HTML은 브라우저로 열리지만 robots.txt 규칙을 확인하지 못했다.
- PDF 다운로드: 일부 게시물의 첨부파일 표시는 확인되나, 자동 다운로드 가능 여부와 개별 권리는 **`unknown`**
- 인증·캡차·유료벽: 대표 공개 페이지에서는 확인되지 않았으나 전체 경로는 **`unknown`**
- RAG 사용: 개별 공공누리 표시 자료만 유형 조건에 따라 검토 가능. 그 외는 **`unknown`**
- 모델 학습 사용: 공공누리 유형과 AI 학습 허용 범위를 개별 확인하기 전 **`unknown`**

**allowed 전환 후보:** 현재 제외. 운영 환경에서 robots.txt를 정상 확인하고 허용 경로를 기록한 뒤, 공공누리 표시가 명확한 개별 게시물만 후보로 재검토한다.

## 4. 국가기록원

- 공식 도메인: **확인** — [국가기록원](https://www.archives.go.kr/)
- robots.txt: **접근 가능** — [robots.txt](https://www.archives.go.kr/robots.txt)
  - `User-agent: *`만 있고 `Disallow` 규칙은 없음
- 이용약관·저작권 정책: [국가기록원 저작권보호정책](https://www.archives.go.kr/next/neworgan/copyrightProtection.do)
- 공공누리·오픈라이선스 표시: 개별 저작물에 표시된 공공누리 유형과 정책의 유형별 조건에서 확인
- 공식 API: **존재** — [나라기록물 검색 OpenAPI](https://contents.archives.go.kr/next/newsearch/openAPI01.do)
  - 공공데이터포털 개발계정 신청과 서비스 인증키 필요
  - 기본 트래픽은 일 1,000건 미만으로 안내
- HTML 수집: **조건부 가능** — robots.txt에 금지 규칙은 없으나 요청 제한과 개별 페이지 이용조건을 지켜야 함
- PDF 다운로드: **가능한 공개 첨부 존재** — [기록관리 표준·지침 목록](https://www.archives.go.kr/next/newdata/standardCondition.do?glSeCd=01)에서 PDF 형식 자료를 확인
- 인증·캡차·유료벽: 대표 공개 페이지에는 없음. API는 인증키 필요. 개별 기록물 공개·열람 조건은 다를 수 있음
- RAG 사용: **문서별 `unknown`** — A·B 신뢰도와 별개로 공공누리 표시 또는 권리 허락 필요
- 모델 학습 사용: **문서별 `unknown`** — 명시적 학습 허용 범위 확인 전 금지

**판정: `allowed`.** 공식 기관 도메인이고 robots.txt에 금지 규칙이 없으며,
저작권 정책과 개별 공공누리 표시 위치가 확인된다. 대표 공개 페이지는
로그인·캡차·유료벽 우회 없이 접근할 수 있다. 다만 API는 인증키가 필요하고,
개별 기록물의 공개 상태와 공공누리 유형은 수집 때마다 다시 확인해야 한다.

## 5. 공공누리

- 공식 도메인: **확인** — 한국문화정보원의 [공공누리](https://www.kogl.or.kr/)
- robots.txt: **접근 가능** — [robots.txt](https://www.kogl.or.kr/robots.txt)
  - Googlebot 그룹에서 `/recommend/recommendDivList.do`, `/search/search.do`, `/search/searchList.do` 금지
  - `User-agent: *` 그룹은 없음
  - 이 결과만으로 프로젝트 User-Agent의 검색 경로를 허용한다고 추정하지 않는다.
- 이용약관·저작권 정책:
  - [공공누리 사이트 저작권정책](https://www.kogl.or.kr/etc/copyright.do)
  - [공공누리 유형별 이용조건](https://www.kogl.or.kr/info/license.do)
- 공공누리·오픈라이선스 표시: 각 저작물 상세 화면의 공공누리 유형 마크와 이용조건에서 확인. 제1~4유형 및 별도의 AI 유형을 구분해야 함
- 공식 API: 공개 API 문서와 신청 절차는 **`unknown`**
- HTML 수집: **조건부 가능** — 공개 안내·상세 페이지 접근은 가능하지만 검색 경로의 자동 수집은 운영자 확인 전 보류
- PDF 다운로드: **대표 공개 PDF 확인** — [공공저작물 이슈리포트 PDF](https://www.kogl.or.kr/namoEditor/binary/files/000001/2026%EB%85%84_1%EB%B6%84%EA%B8%B0_%EA%B3%B5%EA%B3%B5%EC%A0%80%EC%9E%91%EB%AC%BC_%EC%9D%B4%EC%8A%88%EB%A6%AC%ED%8F%AC%ED%8A%B8_-_AI_%EC%8B%9C%EB%8C%80__%EA%B3%B5%EA%B3%B5%EC%A0%80%EC%9E%91%EB%AC%BC_%27%EC%B6%9C%EC%B2%98%EB%AA%85%EC%8B%9C_%EC%9D%98%EB%AC%B4%27_%EB%8B%A4%EC%8B%9C_%EB%AC%BB%EB%8B%A4_1.pdf)
- 인증·캡차·유료벽: 공개 안내 페이지에는 없음. 일부 원문 제공 절차의 로그인·동의 여부는 **`unknown`**
- RAG 사용: **공공누리 유형별 조건부 가능** — 출처표시, 상업·변경 제한 등 해당 유형 준수 필요
- 모델 학습 사용: **AI 유형 또는 명시 허락 자료만 조건부 가능**. 일반 공공누리 표시만으로 자동 허용하지 않음

**allowed 전환 후보:** 조건부 후보. 검색 페이지 크롤링이 아니라 사전 승인된 상세 URL 목록을 사용하고, 각 문서의 공공누리 유형을 기계적으로 검증할 때만 가능하다.

## 6. OAK 국가리포지터리

- 공식 도메인: **확인** — 국립중앙도서관의 [OAK 국가리포지터리](https://oak.go.kr/)
- robots.txt: **접근 가능** — [robots.txt](https://oak.go.kr/robots.txt)
  - `User-agent: *`에 `/` 허용
  - `/login/`, `/_omSystem/`, `/cmm/fms/`, `/adminManager/`, `/manager/`, `/openapi/`, `/qualityControl/` 금지
- 이용약관·저작권 정책:
  - [OAK 소개·오픈액세스 안내](https://oak.go.kr/about/aboutOak.do?menuSeq=28)
  - 학술지별 정책은 [KJCI 저작권 안내](https://copyright.oak.go.kr/)
- 공공누리·오픈라이선스 표시: 개별 리포지터리·학술지·논문의 라이선스 및 KJCI 정책에서 확인. OAK에 공개됐다는 이유만으로 재이용을 허용하지 않음
- 공식 API: **존재** — [OAK Open API 안내](https://oak.go.kr/about/aboutOak.do?menuSeq=93)
  - 기관명, 기관 IP, 이메일 등을 제출하는 신청 절차
  - 비영리·학술 목적 개발자 대상
  - `/openapi/` 경로는 robots.txt에서 금지되므로 크롤링 대신 승인된 API를 사용해야 함
- HTML 수집: **조건부 가능** — robots.txt 허용 경로의 공개 상세 페이지만 가능
- PDF 다운로드: **대표 공개 PDF 확인** — [OAK 저장소 PDF 예시](https://oak.go.kr/repository/journal/14397/BBROBV_2013_v24n4_5.pdf)
- 인증·캡차·유료벽: 대표 공개 HTML/PDF에는 확인되지 않음. API는 신청 필요하며 개별 저장소 조건은 **`unknown`**
- RAG 사용: **문서별 `unknown`** — OA 접근과 RAG 재이용 허락은 동일하지 않음
- 모델 학습 사용: **문서별 `unknown`** — 명시적 오픈라이선스 또는 허락 전 금지

**판정: `allowed`.** 공식 기관 도메인이고 robots.txt가 공개 경로를 허용하며,
오픈액세스 안내·KJCI를 통해 자료별 정책을 다시 확인할 수 있다. 대표 공개
HTML/PDF는 로그인·캡차·유료벽 우회 없이 접근 가능하다. 위험요소는 `/openapi/`
등 금지 경로를 피해야 하고, API 신청이 필요하며, OA 공개가 곧 RAG·학습 허락을
뜻하지 않는다는 점이다.

## 7. KCI 한국학술지인용색인

- 공식 도메인: **확인** — 한국연구재단의 [KCI](https://www.kci.go.kr/kciportal/)
- robots.txt: **접근 가능** — [robots.txt](https://www.kci.go.kr/robots.txt)
  - Googlebot에는 `/` 허용
  - 그 밖의 `User-agent: *`에는 `/` 전체 금지
  - 현재 프로젝트의 일반 수집기 User-Agent에는 전체 금지가 적용된다.
- 이용약관·저작권 정책: KCI 전체 원문에 적용되는 재이용 정책 URL은 **`unknown`**. 학술지별 정책은 [KJCI](https://copyright.oak.go.kr/)에서 확인
- 공공누리·오픈라이선스 표시: 논문 상세의 원문 제공 여부와 학술지별 KJCI 정책에서 확인. 플랫폼 전체의 오픈라이선스는 확인되지 않음
- 공식 API: **존재** — [KCI Open API 목록](https://kci.go.kr/kciportal/po/openapi/openApiList.kci)
  - KCI Open API 인증키 신청·발급 필요
  - 서지·인용 메타데이터 API이며 논문 원문 이용 허락을 뜻하지 않음
- HTML 수집: **불가** — 현재 User-Agent는 robots.txt의 전체 금지 대상
- PDF 다운로드: 일부 학술지는 공개 PDF 링크를 제공하지만 자동 수집은 KCI 및 실제 PDF 제공 서브도메인의 robots·정책을 각각 확인해야 하므로 **불가**
- 인증·캡차·유료벽: API 인증키 필요. 개별 원문은 발행기관 정책에 따라 달라 전체 값은 **`unknown`**
- RAG 사용: **자동 수집 불가**. API 메타데이터도 약관 범위에서만 이용하고 원문은 별도 권리 검토 필요
- 모델 학습 사용: **자동 수집 불가**. 논문별 명시적 허락 없이는 금지

**판정: `blocked`.** HTML/PDF 수집기는 허용하지 않는다. KCI의 승인을 받은
Open API에 한해 메타데이터 전용 수집기로 별도 후보가 될 수 있다.

## 상태 판정과 위험요소

1. **`allowed` — 국가기록원**: 공개 경로 수집 조건은 충족한다. API 키가 필요한
   경로는 인증 전 호출하지 않고, 원문은 공개·라이선스 상태가 명확한 항목만
   draft로 저장한다.
2. **`allowed` — OAK**: 공개 상세 경로만 대상이다. robots.txt의 금지 경로와
   개별 저장소 조건을 준수하고, 라이선스가 없거나 모호한 원문은 RAG·학습 모두
   금지한다.
3. **`blocked` — 한국사데이터베이스·KCI**: `User-agent: *`에 `/` 전체 금지가
   명시되어 일반 HTML/PDF 수집을 실행하지 않는다. 향후 명시적 허가 또는 승인된
   API 전용 범위가 확인되면 별도 재검토한다.
4. **`manual_review` — 국가유산포털·목포시청·공공누리**: 각각 정확한 저작권
   본문, robots.txt 규칙, 검색 경로 정책 중 필요한 근거가 부족하므로 유지한다.

## 후속 검토 체크리스트

- 수집 직전 robots.txt와 정책 본문을 다시 확인하고 감사 시각과 응답을 보존한다.
- 허용 후보는 도메인뿐 아니라 경로 단위 allowlist로 제한한다.
- API는 신청 승인, 인증키, 일일 제한, 결과 재이용·재배포 조건을 별도로 기록한다.
- PDF는 파일 접근 가능 여부와 저작권·라이선스를 별개 필드로 관리한다.
- 공공누리·OA 표시는 문서별로 저장하며 표시가 없거나 모호하면 `unknown`으로 유지한다.
- 로그인, 캡차, 유료벽 또는 접근 차단이 나타나면 우회하지 않고 수집을 중단한다.
- 모든 후보는 `review_status=draft`로 저장하고 사람의 `reviewed` 승인 전 RAG 색인과 학습에서 제외한다.
