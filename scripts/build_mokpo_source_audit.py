"""검증된 공식 상세 URL을 정규화된 조사 산출물로 생성한다.

이 스크립트는 네트워크 요청이나 파일 다운로드를 수행하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "source_audit"


def source_id(institution: str, record_id: str) -> str:
    digest = hashlib.sha256(f"{institution}:{record_id}".encode("utf-8")).hexdigest()[:16]
    return f"mokpo-{digest}"


def record(
    *,
    title: str,
    institution: str,
    url: str,
    record_id: str,
    material_type: str,
    topic: str,
    period: str,
    relevance: str = "direct",
    status: str = "candidate",
    body_exists: bool = True,
    priority: str = "high",
    kogl_type: str = "unknown",
    license_text: str = "개별 자료 이용조건 미확인",
    rights_url: str = "unknown",
    downloadable: bool = False,
    place_names: list[str] | None = None,
    person_names: list[str] | None = None,
) -> dict:
    return {
        "source_id": source_id(institution, record_id),
        "title": title,
        "institution": institution,
        "source_url": url,
        "canonical_url": url,
        "official_record_id": record_id,
        "material_type": material_type,
        "topic": topic,
        "period": period,
        "person_names": person_names or [],
        "place_names": place_names or ["목포"],
        "mokpo_relevance": relevance,
        "body_exists": body_exists,
        "body_length_estimate": "long" if body_exists else "unknown",
        "downloadable": downloadable,
        "public_access": True,
        "kogl_type": kogl_type,
        "license_text": license_text,
        "rights_evidence_url": rights_url,
        "rights_evidence_text": license_text,
        "commercial_use": "unknown",
        "modification_allowed": "unknown",
        "third_party_rights_risk": "unknown",
        "citation_traceable": True,
        "api_key_required": False,
        "permission_required": True,
        "allowed_for_rag": False,
        "allowed_for_training": False,
        "approval_recommended": False,
        "review_status": status,
        "rejection_reason": "",
        "duplicate_of": None,
        "priority": priority,
    }


records: list[dict] = []

# 검색엔진 결과에서 공식 상세 본문과 ID가 확인된 독립기념관 인명사전 레코드.
i815_direct = [
    (4106, "김옥실", "정명여학교·학생운동"),
    (6201, "박애순", "독립운동"),
    (5765, "박상렬", "목포 만세운동"),
    (6313, "독립운동인명사전 기록(ID 6313)", "학교·교육"),
    (4516, "독립운동인명사전 기록(ID 4516)", "노동운동·청년운동"),
    (14276, "조성인", "영흥학교·학생운동"),
    (317, "남궁혁", "목포 만세운동"),
    (15379, "조점환", "목포 학생운동"),
    (1575, "곽우영", "목포 만세운동"),
    (9416, "오상록", "목포공립상업학교·학생운동"),
    (7721, "서화일", "목포 만세운동"),
    (16316, "주유금", "정명여학교·학생운동"),
    (5950, "박복영", "노동운동·소작쟁의"),
    (5972, "박사배", "목포공립상업학교·학생운동"),
    (1685, "곽희주", "정명여학교·학생운동"),
    (13161, "장병준", "신간회 목포지회"),
    (5671, "박종식", "목포공립상업학교·학생운동"),
    (6194, "문복금", "정명여학교·학생운동"),
    (10803, "이인형", "목포공립상업학교·학생운동"),
    (8062, "서상봉", "목포 만세운동"),
    (4105, "김옥남", "영흥학교·학생운동"),
    (4108, "김용문", "영흥학교·학생운동"),
    (11087, "독립운동인명사전 기록(ID 11087)", "정명여학교·학생운동"),
    (6691, "박복술", "정명여학교·학생운동"),
    (15417, "조창섭", "목포공립상업학교·학생운동"),
    (7533, "독립운동인명사전 기록(ID 7533)", "노동운동·소작쟁의"),
]
for rid, name, topic in i815_direct:
    records.append(
        record(
            title=f"{name} - 독립운동인명사전",
            institution="독립기념관",
            url=f"https://search.i815.or.kr/dictionary/detail.do?id={rid}",
            record_id=f"i815-person-{rid}",
            material_type="official_biographical_entry",
            topic=topic,
            period="일제강점기",
            person_names=[] if "ID " in name else [name],
        )
    )

# 목포는 재판·수감·배포 경로로만 확인되어 직접 관련성이 낮은 공식 본문.
i815_reference = [329, 330, 13914, 13694, 2665, 4459, 5375, 9734, 9597, 7183, 4663, 6667, 6525, 7285, 13674]
for rid in i815_reference:
    records.append(
        record(
            title=f"독립운동인명사전 참고 레코드(ID {rid})",
            institution="독립기념관",
            url=f"https://search.i815.or.kr/dictionary/detail.do?id={rid}",
            record_id=f"i815-person-{rid}",
            material_type="official_biographical_entry",
            topic="독립운동·목포 사법/수감 네트워크",
            period="일제강점기",
            relevance="indirect",
            status="reference_only",
            priority="low",
        )
    )

heritage = [
    ("4413607180000", "목포 근대역사문화공간", "근대 도시계획·근대 건축물"),
    ("4413607180100", "목포 근대역사문화공간 내 일본인 가옥-1", "근대 건축물"),
    ("4413605880000", "구 목포부청 서고 및 방공호", "목포부 행정·근대 건축물"),
    ("4413607860000", "목포세관 구 목포지점 터와 세관창고", "목포 해관·세관"),
    ("4413607181500", "구 목포 화신연쇄점", "상업·금융"),
    ("4413606960000", "목포 정광정혜원", "근대 건축물·종교"),
]
for rid, title, topic in heritage:
    records.append(
        record(
            title=title,
            institution="국가유산청 국가유산포털",
            url=f"https://www.heritage.go.kr/heri/cul/culSelectDetail.do?ccbaCpno={rid}&pageNo=1_1_1_0",
            record_id=f"heritage-{rid}",
            material_type="heritage_detail_page",
            topic=topic,
            period="개항기~일제강점기",
        )
    )

city = [
    ("7449", "목포근대역사관 1관(구 목포 일본영사관)", "개항·목포 이사청·목포부 행정"),
    ("7451", "목포근대역사관 2관(구 동양척식주식회사 목포지점)", "동양척식주식회사 목포지점"),
]
for rid, title, topic in city:
    records.append(
        record(
            title=title,
            institution="목포시·목포문화관광",
            url=f"https://tour.mokpo.go.kr/tour/attraction/museum?idx={rid}&mode=view",
            record_id=f"mokpo-tour-{rid}",
            material_type="official_tourism_detail_page",
            topic=topic,
            period="개항기~일제강점기",
            kogl_type="KOGL-4",
            license_text="페이지 하단 공공누리 제4유형(출처표시+상업적 이용금지+변경금지)",
            rights_url=f"https://tour.mokpo.go.kr/tour/attraction/museum?idx={rid}&mode=view",
        )
    )
records.append(
    record(
        title="목포 역사·지명유래",
        institution="목포시",
        url="https://www.mokpo.go.kr/www/introduce/history/origin",
        record_id="mokpo-history-origin",
        material_type="official_history_page",
        topic="개항·목포부 행정·근대 도시계획",
        period="개항기~해방 이후",
    )
)
records.append(
    record(
        title="해설사와 함께 떠나는 근대역사문화거리 투어",
        institution="목포시·목포문화관광",
        url="https://www.mokpo.go.kr/tour/tourguide/tournews?idx=444631&mode=view&page=9",
        record_id="mokpo-tournews-444631",
        material_type="official_program_detail_page",
        topic="목포 근대역사문화공간·근대 건축물",
        period="개항기~일제강점기",
        kogl_type="KOGL-4",
        license_text="페이지 하단 공공누리 제4유형(출처표시+상업적 이용금지+변경금지)",
        rights_url="https://www.mokpo.go.kr/tour/tourguide/tournews?idx=444631&mode=view&page=9",
    )
)

# 상위 20건 상세 권리 검수. 등급은 실제 승인 상태가 아니라 조사 판정이다.
top20_ids = {
    "heritage-4413607180000",
    "heritage-4413607180100",
    "heritage-4413605880000",
    "heritage-4413607860000",
    "heritage-4413607181500",
    "heritage-4413606960000",
    "mokpo-tour-7449",
    "mokpo-tour-7451",
    "mokpo-history-origin",
    "mokpo-tournews-444631",
    "i815-person-1575",
    "i815-person-8062",
    "i815-person-7721",
    "i815-person-317",
    "i815-person-4106",
    "i815-person-6194",
    "i815-person-4105",
    "i815-person-4108",
    "i815-person-9416",
    "i815-person-13161",
}
for item in records:
    item.update(
        {
            "detailed_rights_review": False,
            "rights_grade": "not_assessed",
            "rights_decision_reason": "상위 20건 상세 검수 대상 아님",
            "copyright_policy_url": "unknown",
            "license_scope": "unknown",
            "text_image_rights_separable": "unknown",
            "database_terms": "unknown",
            "reuse_terms": "unknown",
            "required_action_before_approval": "상세 권리 검수",
        }
    )
    if item["official_record_id"] not in top20_ids:
        continue
    item["detailed_rights_review"] = True
    if item["institution"] == "독립기념관":
        item.update(
            {
                "rights_grade": "permission_request_required",
                "rights_decision_reason": (
                    "공식 상세 본문과 인명사전 ID는 확인되나 개별 공공누리 표시와 "
                    "정제·청크화·요약 허락을 확인하지 못함"
                ),
                "copyright_policy_url": "unknown",
                "license_scope": "기관 정책과 개별 본문의 적용 관계 미확인",
                "text_image_rights_separable": "yes_by_excluding_media",
                "database_terms": "한국독립운동정보시스템 데이터베이스 별도 조건 확인 필요",
                "reuse_terms": "명시적 재사용 문구 미확인",
                "required_action_before_approval": (
                    "독립기념관에 본문 텍스트 저장·청크화·임베딩·요약 허용 여부 서면 문의"
                ),
            }
        )
        item["rights_evidence_url"] = "https://www.i815.or.kr/"
        item["rights_evidence_text"] = "개별 인명사전 본문의 공공누리 유형 및 RAG 가공 허락 미확인"
        item["third_party_rights_risk"] = "medium"
    elif item["institution"] == "국가유산청 국가유산포털":
        item.update(
            {
                "rights_grade": "permission_request_required",
                "rights_decision_reason": (
                    "공식 상세 설명과 국가유산 식별자는 확인되나 상세 레코드 자체의 "
                    "공공누리 유형이 표시된 것으로 확인되지 않음"
                ),
                "copyright_policy_url": "https://www.heritage.go.kr/",
                "license_scope": "포털 저작권정책 링크는 있으나 개별 상세 본문 적용 범위 미확인",
                "text_image_rights_separable": "yes_by_excluding_media",
                "database_terms": "국가유산 정보 데이터베이스 재사용 조건 확인 필요",
                "reuse_terms": "자료별 공공누리 유형을 확인한 뒤 이용해야 함",
                "required_action_before_approval": (
                    "국가유산청에 상세 설명 텍스트의 공공누리 유형과 RAG 가공 허용 여부 문의"
                ),
            }
        )
        item["rights_evidence_url"] = "https://www.heritage.go.kr/"
        item["rights_evidence_text"] = "포털 하단 저작권정책 링크는 확인되나 개별 상세 레코드 유형 미확인"
        item["third_party_rights_risk"] = "medium"
    elif item["kogl_type"] == "KOGL-4":
        item.update(
            {
                "rights_grade": "unsuitable_for_rag",
                "rights_decision_reason": (
                    "개별 페이지 하단 공공누리 제4유형은 비상업 조건과 변경금지를 부과하여 "
                    "정제·청크화·요약·생성 답변과 충돌할 위험이 큼"
                ),
                "copyright_policy_url": "https://www.mokpo.go.kr/www/operation_guide/copyright",
                "license_scope": "개별 페이지 하단 표시",
                "text_image_rights_separable": "yes_by_excluding_media",
                "database_terms": "별도 데이터베이스 조건 미확인",
                "reuse_terms": "출처표시, 상업적 이용금지, 변경금지",
                "required_action_before_approval": (
                    "목포시에 텍스트 한정 청크화·임베딩·요약 별도 허락 요청"
                ),
            }
        )
        item["commercial_use"] = False
        item["modification_allowed"] = False
        item["permission_required"] = True
        item["third_party_rights_risk"] = "medium"
    else:
        item.update(
            {
                "rights_grade": "permission_request_required",
                "rights_decision_reason": (
                    "목포시 공식 본문이나 개별 공공누리 표시가 확인되지 않아 "
                    "사이트 일반 정책만으로 재사용을 허용할 수 없음"
                ),
                "copyright_policy_url": "https://www.mokpo.go.kr/www/operation_guide/copyright",
                "license_scope": "개별 자료 적용 표시 미확인",
                "text_image_rights_separable": "yes_by_excluding_media",
                "database_terms": "별도 데이터베이스 조건 미확인",
                "reuse_terms": "공공누리 표시가 있는 저작물만 유형별 조건으로 이용 가능",
                "required_action_before_approval": "목포시에 해당 본문 텍스트의 별도 이용 허락 문의",
            }
        )
        item["rights_evidence_url"] = "https://www.mokpo.go.kr/www/operation_guide/copyright"
        item["rights_evidence_text"] = "목포시 정책은 공공누리 표시가 부착된 저작물에만 자유이용 적용"
        item["third_party_rights_risk"] = "low"

assert len({r["canonical_url"] for r in records}) == len(records)
assert all(r["review_status"] in {"candidate", "reference_only"} for r in records)
assert not any(r["allowed_for_rag"] or r["allowed_for_training"] for r in records)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "mokpo_public_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for item in records:
        fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

candidate_count = sum(r["review_status"] == "candidate" for r in records)
reference_count = sum(r["review_status"] == "reference_only" for r in records)
body_count = sum(bool(r["body_exists"]) for r in records)
direct_count = sum(r["mokpo_relevance"] == "direct" for r in records)
summary = {
    "audit_date": "2026-07-30",
    "scope_note": "공공데이터포털 제외. 네트워크 원문 다운로드·승인·색인 없음.",
    "investigated_records": len(records),
    "unique_candidates": len(records),
    "candidate_count": candidate_count,
    "reference_only_count": reference_count,
    "direct_mokpo_count": direct_count,
    "body_exists_count": body_count,
    "kogl_counts": dict(Counter(r["kogl_type"] for r in records)),
    "immediately_rag_approvable": 0,
    "approval_recommended_count": 0,
    "rights_pending_count": sum(r["kogl_type"] == "unknown" for r in records),
    "institution_inquiry_needed_count": sum(r["permission_required"] for r in records),
    "excluded_count": 0,
    "duplicate_count": 0,
    "institution_counts": dict(Counter(r["institution"] for r in records)),
    "topic_counts": dict(Counter(r["topic"] for r in records)),
    "estimated_chunks": {"minimum": candidate_count * 2, "maximum": candidate_count * 5},
    "current_approved_documents": 0,
    "expected_cumulative_approved": 0,
    "shortfall_to_50_approved": 50,
    "shortfall_to_100_approved": 100,
    "termination_reason": (
        "소량 공식 상세 페이지 탐색에서 확인 가능한 고유 상세 레코드를 소진했으며, "
        "추가 확대에는 기관 검색 API/서지 UI의 대량 탐색 또는 권리 확인이 필요함"
    ),
    "detailed_review": {
        "assessed_count": 20,
        "approval_recommended": 0,
        "conditional_permission": 0,
        "permission_request_required": 17,
        "unsuitable_for_rag": 3,
        "expected_first_pilot_approvals_now": 0,
        "potential_after_written_permission": "5~10",
    },
}
(OUT / "mokpo_candidate_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"records": len(records), "candidate": candidate_count, "reference_only": reference_count}, ensure_ascii=False))
