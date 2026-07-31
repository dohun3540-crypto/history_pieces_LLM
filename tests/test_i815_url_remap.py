from __future__ import annotations

from history_chatbot.provisional.remap import (
    SearchCandidate,
    classify_mapping,
    classify_old_response,
    ensure_unique_exact_urls,
    old_record_id,
    parse_detail_fields,
    parse_search_candidates,
    review_manual_record,
    search_key,
)


def record(title: str = "김옥실 - 독립운동인명사전") -> dict:
    return {
        "source_id": "mokpo-a50452bd9b1092c3",
        "source_title": title,
        "source_url": "https://search.i815.or.kr/dictionary/detail.do?id=4106",
        "usage_status": "provisional_hackathon",
        "allowed_for_rag": False,
        "allowed_for_training": False,
    }


def detail_html(
    name: str = "김옥실",
    body: str = "목포 정명여학교 만세운동",
    *,
    mokpo_metadata: bool = True,
) -> str:
    place = "목포" if mokpo_metadata else "광주"
    return (
        "<html><body><span class='entry-name'>"
        f"{name}</span><span class='entry-name-hanja'>金玉實</span>"
        f"<table><tr><th>출신지</th><td>전남 {place}</td></tr>"
        "<tr><th>생몰년월일</th><td>1900 ~ 1980</td></tr>"
        "<tr><th>운동계열</th><td>학생운동</td></tr>"
        "<tr><th>관련 단체</th><td>정명여학교</td></tr>"
        f"<tr><th>관련 사건</th><td>{place} 만세시위</td></tr>"
        f"<tr><th>주요 활동</th><td>{place} 만세운동 참여</td></tr></table>"
        f"<section>{body * 20}.</section></body></html>"
    )


def test_old_500_error_page_is_classified_as_removed() -> None:
    assert (
        classify_old_response(
            500, "페이지를 찾을 수 없습니다. 주소가 변경 혹은 삭제되었습니다."
        )
        == "source_removed"
    )


def test_exact_candidate_is_extracted_from_search_result() -> None:
    html = """
    <li><div onclick="goDetail(4106);">
      <strong class="dict_txt_title">김옥실 (金玉實)</strong>
    </div></li>
    """
    assert parse_search_candidates(html) == [
        SearchCandidate("4106", "김옥실 (金玉實)")
    ]


def test_same_name_multiple_ids_is_ambiguous() -> None:
    status, _, _, _ = classify_mapping(
        record(),
        status=200,
        content_type="text/html; charset=UTF-8",
        html=detail_html(),
        search_candidates=[
            SearchCandidate("4106", "김옥실"),
            SearchCandidate("9999", "김옥실"),
        ],
    )
    assert status == "ambiguous"


def test_only_exact_name_and_mokpo_is_auto_approved() -> None:
    exact, _, _, current_title = classify_mapping(
        record(),
        status=200,
        content_type="text/html",
        html=detail_html(),
    )
    probable, _, _, _ = classify_mapping(
        record(),
        status=200,
        content_type="text/html",
        html=detail_html(body="광주 만세운동", mokpo_metadata=False),
    )
    assert exact == "remapped_exact"
    assert current_title == "김옥실"
    assert probable == "remapped_probable"


def test_generic_title_requires_manual_review_and_keeps_source_id() -> None:
    item = record("독립운동인명사전 참고 레코드(ID 4106)")
    status, _, _, _ = classify_mapping(
        item,
        status=200,
        content_type="text/html",
        html=detail_html(),
    )
    assert status == "manual_review_required"
    assert item["source_id"] == "mokpo-a50452bd9b1092c3"
    assert old_record_id(item["source_url"]) == "4106"
    assert search_key(item) == "4106"


def test_duplicate_exact_current_url_is_rejected() -> None:
    items = [
        {
            "source_id": "one",
            "match_status": "remapped_exact",
            "current_detail_url": "https://search.i815.or.kr/dictionary/detail/print.do?id=1",
        },
        {
            "source_id": "two",
            "match_status": "remapped_exact",
            "current_detail_url": "https://search.i815.or.kr/dictionary/detail/print.do?id=1",
        },
    ]
    try:
        ensure_unique_exact_urls(items)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate current URL must be rejected")


def test_mapping_dry_run_does_not_mutate_manifest_record() -> None:
    item = record()
    before = dict(item)
    classify_mapping(
        item,
        status=200,
        content_type="text/html",
        html=detail_html(),
    )
    assert item == before


def test_manual_generic_title_is_promoted_with_direct_mokpo_evidence() -> None:
    item = record("독립운동인명사전 참고 레코드(ID 4106)")
    before = dict(item)
    result = review_manual_record(
        item,
        current_url="https://search.i815.or.kr/dictionary/detail/print.do?id=4106",
        status=200,
        content_type="text/html; charset=UTF-8",
        html=detail_html(),
    )
    assert result["review_status"] == "promoted_to_exact"
    assert result["mokpo_relevance"] == "direct"
    assert result["current_name"] == "김옥실"
    assert parse_detail_fields(detail_html()).movement_family == "학생운동"
    assert item == before
    assert item["source_id"] == result["source_id"]
    assert item["allowed_for_rag"] is False
    assert item["allowed_for_training"] is False


def test_manual_record_without_mokpo_evidence_is_not_promoted() -> None:
    result = review_manual_record(
        record("독립운동인명사전 참고 레코드(ID 4106)"),
        current_url="https://search.i815.or.kr/dictionary/detail/print.do?id=4106",
        status=200,
        content_type="text/html",
        html=detail_html(body="광주 지역 독립운동", mokpo_metadata=False),
    )
    assert result["review_status"] == "reference_only"


def test_manual_duplicate_url_or_record_id_is_rejected() -> None:
    item = record("독립운동인명사전 참고 레코드(ID 4106)")
    result = review_manual_record(
        item,
        current_url="https://search.i815.or.kr/dictionary/detail/print.do?id=4106",
        status=200,
        content_type="text/html",
        html=detail_html(),
        existing_results=[
            {
                "source_id": "another",
                "current_url": "https://search.i815.or.kr/dictionary/detail/print.do?id=4106",
                "record_id": "4106",
            }
        ],
    )
    assert result["review_status"] == "rejected"
    assert result["duplicate_check"] == "duplicate"
