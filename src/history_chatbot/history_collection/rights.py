"""Fail-closed robots, access-policy, and document-rights decisions."""

from __future__ import annotations

from dataclasses import dataclass

from history_chatbot.history_collection.models import HardRejectionCode, RightsEvidence


RIGHTS_ALLOWED = {"public_domain", "open_license", "permission_granted"}
RIGHTS_RESTRICTED = {"restricted", "prohibited"}
ROBOTS_ALLOWED = {"allowed", "verified_allowed"}
ACCESS_ALLOWED = {"public", "allowed"}


@dataclass(frozen=True, slots=True)
class RightsDecision:
    score: int
    usable_for_rag: bool
    needs_human_review: bool
    hard_rejections: tuple[HardRejectionCode, ...]
    reasons: tuple[str, ...]


def normalize_kogl(value: str) -> str:
    compact = value.strip().upper().replace("공공누리", "KOGL").replace("제", "").replace("유형", "")
    compact = compact.replace(" ", "").replace("_", "-")
    mapping = {"KOGL1": "KOGL-1", "KOGL2": "KOGL-2", "KOGL3": "KOGL-3", "KOGL4": "KOGL-4"}
    return mapping.get(compact, compact)


def evaluate_rights(robots_status: str, access_status: str, rights_status: str,
                    evidence: RightsEvidence, *, policy_status: str = "allowed") -> RightsDecision:
    hard: list[HardRejectionCode] = []
    reasons: list[str] = []
    if robots_status not in ROBOTS_ALLOWED:
        hard.append(HardRejectionCode.ROBOTS_BLOCKED)
        reasons.append("robots가 명시적으로 허용되지 않음")
    if policy_status != "allowed":
        hard.append(HardRejectionCode.POLICY_BLOCKED)
        reasons.append("source policy가 명시적으로 허용되지 않음")
    if access_status in {"login_required", "login"}:
        hard.append(HardRejectionCode.LOGIN_REQUIRED)
    elif access_status == "captcha":
        hard.append(HardRejectionCode.CAPTCHA)
    elif access_status == "paywall":
        hard.append(HardRejectionCode.PAYWALL)
    elif access_status not in ACCESS_ALLOWED:
        hard.append(HardRejectionCode.POLICY_BLOCKED)
        reasons.append("공개 접근 상태가 확인되지 않음")
    if rights_status in RIGHTS_RESTRICTED:
        hard.append(HardRejectionCode.RIGHTS_RESTRICTED)
    normalized_kogl = normalize_kogl(evidence.kogl_type)
    explicit = rights_status in RIGHTS_ALLOWED
    evidence_complete = bool(evidence.checked_at and (evidence.document_rights_url or evidence.policy_url))
    safe_kogl = normalized_kogl == "KOGL-1"
    usable = explicit and evidence_complete and (safe_kogl or bool(evidence.license_text)) and not hard
    needs_review = evidence.human_review_required or not usable or normalized_kogl in {"KOGL-2", "KOGL-3", "KOGL-4"}
    score = 10 if usable and not needs_review else 7 if explicit and evidence_complete else 3 if evidence_complete else 0
    return RightsDecision(score, usable, needs_review, tuple(dict.fromkeys(hard)), tuple(reasons))
