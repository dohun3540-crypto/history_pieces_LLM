from __future__ import annotations

from history_chatbot.provisional.remap import (
    SearchCandidate,
    classify_mapping,
    classify_old_response,
    ensure_unique_exact_urls,
    old_record_id,
    parse_search_candidates,
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


def detail_html(name: str = "김옥실", body: str = "목포 정명여학교 만세운동") -> str:
    return (
        "<html><body><span class='entry-name'>"
        f"{name}</span><section>{body * 20}</section></body></html>"
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
        html=detail_html(body="광주 만세운동"),
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
