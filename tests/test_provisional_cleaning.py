from __future__ import annotations

from history_chatbot.provisional.cleaning import clean_sections
from history_chatbot.provisional.service import ProvisionalDataService


def record(institution: str, title: str = "검증 문서") -> dict:
    return {
        "source_id": "doc-1",
        "source_title": title,
        "institution": institution,
        "source_url": "https://example.invalid/source",
    }


def texts(sections) -> str:
    return "\n".join(
        paragraph for section in sections for paragraph in section.paragraphs
    )


def test_independence_header_and_references_are_removed_but_history_is_preserved() -> None:
    source = """독립운동인명사전 - 한국독립운동정보시스템
모바일 메뉴 열기/닫기
닫기
박종식
朴鍾殖
기본정보
한글명
박종식
관련 사건
광주학생운동
포상훈격(연도)
건국포장(1993)
1929년 목포상업학교 학생들과 동조 시위를 준비하였다.
11월 19일 목포역 앞에서 선전문을 배포하였다.
⋮집필자⋮
|참고문헌|
「판결문」, 1930.
프린트"""

    cleaned = texts(clean_sections(record("독립기념관"), source))

    assert "모바일 메뉴" not in cleaned
    assert "참고문헌" not in cleaned
    assert "판결문" not in cleaned
    assert "박종식" in cleaned
    assert "1929년 목포상업학교" in cleaned
    assert "11월 19일 목포역" in cleaned


def test_independence_profile_excludes_searchable_activity_metadata_but_preserves_source() -> None:
    source = """박종식
朴鍾殖
기본정보
한글명
박종식
한자명
朴鍾殖
본 관
밀양(密陽)
출신지
전남 진도(珍島)
생몰년월일
1911. 12. 20 ~ 1948. 10. 20
운동계열
학생운동
관련 단체
목포공립상업학교
관련 사건
광주학생운동
주요 활동
1929년 목포상업학교 학생들과 광주학생운동 동조 시위
포상훈격(연도)
건국포장(1993)
1929년 목포상업학교 학생들과 광주학생운동 동조 시위를 준비하였다.
대한민국 정부는 1993년 건국포장을 추서하였다."""

    sections = clean_sections(record("독립기념관"), source)
    profile = sections[0]
    profile_text = "\n".join(profile.paragraphs)
    narrative_text = texts(sections[1:])

    assert profile.title == "박종식 기본정보"
    assert "박종식" in profile_text
    assert "밀양(密陽)" in profile_text
    assert "전남 진도(珍島)" in profile_text
    assert "1911. 12. 20 ~ 1948. 10. 20" in profile_text
    assert "운동계열" not in profile_text
    assert "광주학생운동" not in profile_text
    assert "동조 시위" not in profile_text
    assert profile.metadata["source_profile_metadata"] == (
        "운동계열",
        "학생운동",
        "관련 단체",
        "목포공립상업학교",
        "관련 사건",
        "광주학생운동",
        "주요 활동",
        "1929년 목포상업학교 학생들과 광주학생운동 동조 시위",
        "포상훈격(연도)",
        "건국포장(1993)",
    )
    assert "1929년 목포상업학교 학생들과 광주학생운동 동조 시위" in narrative_text
    assert "대한민국 정부는 1993년 건국포장을 추서하였다." in narrative_text


def test_profile_source_metadata_is_preserved_outside_searchable_chunk_text() -> None:
    source = """박종식
朴鍾殖
기본정보
한글명
박종식
출신지
전남 진도
생몰년월일
1911 ~ 1948
관련 사건
광주학생운동
포상훈격(연도)
건국포장(1993)
1929년 광주학생운동 동조 시위에 참여하였다.
대한민국 정부는 1993년 건국포장을 추서하였다."""

    chunks = ProvisionalDataService()._chunks(record("독립기념관"), source)
    profile = chunks[0]

    assert profile["section_title"] == "박종식 기본정보"
    assert "전남 진도" in profile["text"]
    assert "1911 ~ 1948" in profile["text"]
    assert "광주학생운동" not in profile["text"]
    assert profile["source_profile_metadata"] == (
        "관련 사건",
        "광주학생운동",
        "포상훈격(연도)",
        "건국포장(1993)",
    )
    assert any("1929년 광주학생운동" in item["text"] for item in chunks[1:])
    assert any("건국포장을 추서" in item["text"] for item in chunks[1:])


def test_reference_marker_keeps_substantive_history_before_it() -> None:
    source = """박종식
朴鍾殖
기본정보
포상훈격(연도)
건국포장(1993)
1929년 학생운동에 참여하였다.
대한민국 정부는 1993년 건국포장을 추서하였다.
|참고문헌|
자료 목록"""

    cleaned = texts(clean_sections(record("독립기념관"), source))

    assert "1929년 학생운동" in cleaned
    assert "건국포장을 추서" in cleaned
    assert "자료 목록" not in cleaned


def test_biography_periods_are_kept_in_separate_chunks() -> None:
    source = """오상록
吳上祿
기본정보
포상훈격(연도)
애족장(1990)
1929년 목포상업학교 학생운동을 준비하였다.
11월 19일 목포역을 거쳐 시위를 전개하였다.
출옥 이후 일본 나고야에서 노동운동을 전개하였다.
1931년 12월 11일 붙잡혔다."""

    chunks = ProvisionalDataService()._chunks(record("독립기념관"), source)

    assert any("1929년 목포상업학교" in item["text"] for item in chunks)
    assert any("일본 나고야" in item["text"] for item in chunks)
    assert all(
        not ("11월 19일 목포역" in item["text"] and "일본 나고야" in item["text"])
        for item in chunks
    )


def test_heritage_basic_information_survives_without_page_ui() -> None:
    source = """본문
국가유산 검색
페이지 인쇄
이 페이지의 구성
목포 근대역사문화공간
이전
이전 문화유산
다음
다음 문화유산
기본 정보
국가등록문화유산
목포 근대역사문화공간
수량/면적
114,038㎡(602필지)
지정(등록)일
2018.08.06
소재지
목포시 유달동 7
이전 시기의 건축 양식을 보여준다.
다음 문화유산
QR코드 바로보기
목록으로 이동하기"""

    cleaned = texts(clean_sections(record("국가유산청 국가유산포털"), source))

    assert "국가등록문화유산" in cleaned
    assert "114,038㎡(602필지)" in cleaned
    assert "2018.08.06" in cleaned
    assert "목포시 유달동 7" in cleaned
    assert "이전 시기의 건축 양식을 보여준다." in cleaned
    assert "페이지 인쇄" not in cleaned
    assert "QR코드" not in cleaned
    assert "이전 문화유산" not in cleaned
    assert "다음 문화유산" not in cleaned


def test_mokpo_chronology_preserves_modern_facts_and_conflicting_source_dates() -> None:
    source = """공통 메뉴
삼국 이전
마한의 세력권에 있었다.
조선시대
조선시대의 역사이다.
1895년 무안군에서 성립하고 1897년 10월 1일 개항하였다.
1899년 호안석축공사를 하고 1905년 기선을 사용하였다.
상수도는 1911년 5월 완비되었다.
일제강점기
1932년 행정구역을 확장하고 인구 6만의 도시가 되었다.
호남선은 1914년 1월 14일 전선이 개통되었다.
8·15 광복 이후
1949년 목포시로 개칭하였다.
요약
삼국 이전
마한에 속함
고종32년(1895) 무안군에서 성립
1897.10.1 개항
1914.1.22 호남선 철도 개통
8·15 광복 이후
1949.8.15 목포시로 개칭
담당자도시유산과"""

    sections = clean_sections(record("목포시", "목포 역사·지명유래"), source)
    cleaned = texts(sections)

    assert "공통 메뉴" not in cleaned
    assert "1897년 10월 1일 개항" in cleaned
    assert "1899년 호안석축공사" in cleaned
    assert "1911년 5월" in cleaned
    assert "1932년 행정구역" in cleaned
    assert "1914년 1월 14일" in cleaned
    assert "1914.1.22" in cleaned
    assert "담당자" not in cleaned
    assert {section.title for section in sections} >= {
        "개항·거류·무역",
        "항만·도로·해상교통",
        "철도·교통",
        "일제강점기 도시 확장",
        "개항 이후 근대 도시 변화 요약",
        "광복 이후 현대 행정구역 변화",
    }
    modern_overview = next(
        section
        for section in sections
        if section.title == "개항 이후 근대 도시 변화 요약"
    )
    post_liberation = next(
        section
        for section in sections
        if section.title == "광복 이후 현대 행정구역 변화"
    )
    assert modern_overview.paragraphs == (
        "고종32년(1895) 무안군에서 성립",
        "1897.10.1 개항",
        "1914.1.22 호남선 철도 개통",
    )
    assert post_liberation.paragraphs == (
        "8·15 광복 이후",
        "1949년 목포시로 개칭하였다.",
    )
    assert "개항 이후 근대 도시 변화 요약" not in cleaned
    assert "광복 이후 현대 행정구역 변화" not in cleaned


def test_chunk_provenance_and_searchable_section_title_are_preserved() -> None:
    source_record = record("국가유산청 국가유산포털", "목포 근대역사문화공간")
    source = """기본 정보
국가등록문화유산
목포 근대역사문화공간
지정(등록)일
2018.08.06
소재지
목포시 유달동 7"""

    chunk = ProvisionalDataService()._chunks(source_record, source)[0]

    assert chunk["document_id"] == "doc-1"
    assert chunk["source_id"] == "doc-1"
    assert chunk["source_url"] == "https://example.invalid/source"
    assert chunk["source_title"] == "목포 근대역사문화공간"
    assert chunk["section_title"] == "기본 정보"
    assert chunk["title"] == "목포 근대역사문화공간 — 기본 정보"
    assert chunk["text"].startswith("기본 정보\n")
