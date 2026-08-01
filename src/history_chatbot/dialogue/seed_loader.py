"""Non-destructive loader for reviewed and proposed Giroksae seed bundles."""

from __future__ import annotations

import json
from pathlib import Path

from history_chatbot.dialogue.personalization_tags import classify_scope, TagScope
from history_chatbot.dialogue.situation_models import (
    ActionCode, FallbackBehavior, RequiredContext, ResponseLengthMode,
    ReviewStatus, ScreenType, SituationExample, SituationId, SourceType,
)


DEFAULT_SEED_PATH = Path("configs/giroksae_situations.json")
DEFAULT_ADDITIONS_PATH = Path("configs/giroksae_situations_v03_additions.json")
REQUIRED_SOURCE_FIELDS = {
    "screen_type", "user_input", "response_goal", "response_draft",
    "next_action", "personalization_tags",
}
ALLOWED_POLICY_FLAGS = {
    "no_rag", "requires_rag", "requires_evidence", "requires_citations",
    "requires_clarification", "requires_time_scope", "requires_visual_context",
    "requires_completed_piece_context", "requires_external_comparison_source",
    "requires_app_state", "requires_journey_state", "requires_map_data",
    "requires_location", "requires_verified_facility_data",
    "requires_storage_capability", "requires_consent", "source_insufficient",
    "incorrect_premise", "ambiguous_question", "compare_prior_citations",
    "safety_first", "do_not_press", "do_not_mirror_abuse",
}
CONTEXT_TAGS = {
    "technical_issue", "navigation_issue", "accessibility_request",
    "current_fatigue", "heat_discomfort", "frustration", "emotion_sadness",
    "engagement_low", "engagement_very_low", "service_dissatisfaction",
}


class SituationSeedLoader:
    """Load the immutable 54-record baseline and optionally merge V03 proposals."""

    def __init__(
        self,
        path: Path = DEFAULT_SEED_PATH,
        *,
        additions_path: Path | None = DEFAULT_ADDITIONS_PATH,
    ) -> None:
        self.path = path
        self.additions_path = additions_path

    def load(self) -> tuple[SituationExample, ...]:
        legacy = self._load_bundle(
            self.path, expected_schema=1, expected_count=54,
            expected_source=SourceType.HUMAN_AUTHORED_SEED,
            expected_review=ReviewStatus.REVIEWED_SEED,
        )
        if self.additions_path is None:
            return legacy
        additions = self._load_bundle(
            self.additions_path, expected_schema=2, expected_count=9,
            expected_source=SourceType.HUMAN_PROPOSED_SEED,
            expected_review=ReviewStatus.REVIEW_PENDING,
        )
        examples = legacy + additions
        ids = [item.example_id for item in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("example_id가 중복되었습니다.")
        if len(examples) != 63 or {item.situation_id for item in examples} != set(SituationId):
            raise ValueError("통합 seed는 20개 상황과 63개 사례여야 합니다.")
        return examples

    def _load_bundle(
        self, path: Path, *, expected_schema: int, expected_count: int,
        expected_source: SourceType, expected_review: ReviewStatus,
    ) -> tuple[SituationExample, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if type(payload) is not dict or payload.get("schema_version") != expected_schema:
            raise ValueError("지원하지 않는 seed schema_version입니다.")
        if payload.get("source_type") != expected_source.value or payload.get("review_status") != expected_review.value:
            raise ValueError("seed bundle의 출처 또는 검토 상태가 올바르지 않습니다.")
        records = payload.get("examples")
        situations = payload.get("situations")
        if type(records) is not list or type(situations) is not list or len(records) != expected_count:
            raise ValueError(f"seed bundle은 정확히 {expected_count}개 사례여야 합니다.")
        ids = [item.get("example_id") for item in records if type(item) is dict]
        if len(ids) != len(records) or len(ids) != len(set(ids)):
            raise ValueError("example_id가 중복되었거나 유효하지 않습니다.")
        examples = tuple(self._parse(item, expected_source, expected_review) for item in records)
        declared = {SituationId(item["situation_id"]): item["example_count"] for item in situations}
        actual = {s: sum(x.situation_id == s for x in examples) for s in declared}
        if declared != actual:
            raise ValueError("상황별 선언 수와 실제 사례 수가 다릅니다.")
        return examples

    @staticmethod
    def _string_list(item: dict, name: str) -> tuple[str, ...]:
        value = item.get(name, [])
        if type(value) is not list or any(type(entry) is not str for entry in value):
            raise ValueError(f"{name}은 문자열 배열이어야 합니다.")
        return tuple(value)

    @classmethod
    def _parse(cls, item: dict, expected_source: SourceType, expected_review: ReviewStatus) -> SituationExample:
        if type(item) is not dict:
            raise ValueError("seed 레코드는 객체여야 합니다.")
        required = {
            "example_id", "situation_id", "screen_type", "user_input",
            "response_goal", "response_draft", "next_action",
            "personalization_tags", "requires_rag", "requires_clarification",
            "allows_follow_up", "response_length_mode", "next_action_code",
            "locale", "source_type", "review_status", "source_fields",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"seed 필드 누락: {', '.join(sorted(missing))}")
        for name in ("example_id", "user_input", "response_goal", "response_draft", "next_action", "next_action_code", "locale"):
            if type(item[name]) is not str or not item[name]:
                raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다.")
        for name in ("requires_rag", "requires_clarification", "allows_follow_up"):
            if type(item[name]) is not bool:
                raise ValueError(f"{name}은 bool이어야 합니다.")
        source_type = SourceType(item["source_type"])
        review_status = ReviewStatus(item["review_status"])
        if source_type != expected_source or review_status != expected_review:
            raise ValueError("seed 출처 또는 검토 상태가 bundle과 일치하지 않습니다.")
        source_fields = item["source_fields"]
        if type(source_fields) is not dict or set(source_fields) != REQUIRED_SOURCE_FIELDS:
            raise ValueError("원본 필드 보존 정보가 불완전합니다.")
        for name in ("user_input", "response_goal", "response_draft", "next_action"):
            if item[name] != source_fields[name]:
                raise ValueError(f"원본 문구가 변경되었습니다: {item['example_id']}:{name}")
        raw_tags = cls._string_list(item, "personalization_tags")
        context_state = list(cls._string_list(item, "context_state"))
        personalization: list[str] = []
        for tag in raw_tags:
            scope = classify_scope(tag)
            if scope in {TagScope.JOURNEY_INTEREST, TagScope.PREFERENCE_CANDIDATE}:
                personalization.append(tag)
            elif scope == TagScope.SESSION_OBSERVATION:
                context_state.append(tag)
        if any(tag not in CONTEXT_TAGS and not tag.startswith(("prefers_", "emotion_")) for tag in context_state):
            raise ValueError("알 수 없는 context_state 값입니다.")
        policy_flags = cls._string_list(item, "policy_flags")
        if any(flag not in ALLOWED_POLICY_FLAGS for flag in policy_flags):
            raise ValueError("알 수 없는 policy_flags 값입니다.")
        required_context = tuple(RequiredContext(value) for value in cls._string_list(item, "required_context"))
        if expected_source == SourceType.HUMAN_PROPOSED_SEED:
            ActionCode(item["next_action_code"])
        response_original = item.get("response_draft_original", item["response_draft"])
        response_template = item.get("response_template", "")
        if type(response_original) is not str or type(response_template) is not str:
            raise ValueError("응답 원문과 템플릿은 문자열이어야 합니다.")
        if response_original != item["response_draft"]:
            raise ValueError("response_draft_original은 보존 원문과 일치해야 합니다.")
        return SituationExample(
            example_id=item["example_id"], situation_id=SituationId(item["situation_id"]),
            screen_type=ScreenType(item["screen_type"]), user_input=item["user_input"],
            response_goal=item["response_goal"], response_draft=item["response_draft"],
            response_draft_original=response_original, response_template=response_template,
            next_action=item["next_action"], personalization_tags=tuple(dict.fromkeys(personalization)),
            context_state=tuple(dict.fromkeys(context_state)), policy_flags=policy_flags,
            required_context=required_context, requires_rag=item["requires_rag"],
            requires_clarification=item["requires_clarification"], allows_follow_up=item["allows_follow_up"],
            response_length_mode=ResponseLengthMode(item["response_length_mode"]),
            next_action_code=item["next_action_code"],
            fallback_behavior=FallbackBehavior(item.get("fallback_behavior", "none")),
            locale=item["locale"], source_type=source_type, review_status=review_status,
            source_fields=dict(source_fields),
        )
