"""구조화된 human-authored seed의 엄격한 loader와 원본 보존 검증."""

from __future__ import annotations

import json
from pathlib import Path

from history_chatbot.dialogue.situation_models import (
    ResponseLengthMode,
    ScreenType,
    SituationExample,
    SituationId,
)


DEFAULT_SEED_PATH = Path("configs/giroksae_situations.json")
REQUIRED_SOURCE_FIELDS = {
    "screen_type",
    "user_input",
    "response_goal",
    "response_draft",
    "next_action",
    "personalization_tags",
}


class SituationSeedLoader:
    def __init__(self, path: Path = DEFAULT_SEED_PATH) -> None:
        self.path = path

    def load(self) -> tuple[SituationExample, ...]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("source_type") != "human_authored_seed":
            raise ValueError("생성형 데이터는 human-authored seed loader로 읽을 수 없습니다.")
        situations = payload.get("situations", [])
        if len(situations) != 17 or {x["situation_id"] for x in situations} != {x.value for x in SituationId}:
            raise ValueError("상황 분류는 정의된 17개와 정확히 일치해야 합니다.")
        records = payload.get("examples", [])
        if len(records) != 54:
            raise ValueError("human-authored seed는 정확히 54개여야 합니다.")
        ids = [str(item.get("example_id", "")) for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError("example_id가 중복되었습니다.")
        examples = tuple(self._parse(item) for item in records)
        declared = {item["situation_id"]: item["example_count"] for item in situations}
        actual = {situation.value: sum(x.situation_id == situation for x in examples) for situation in SituationId}
        if declared != actual:
            raise ValueError("상황별 선언 사례 수와 실제 사례 수가 다릅니다.")
        return examples

    @staticmethod
    def _parse(item: dict) -> SituationExample:
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
        if item["source_type"] != "human_authored_seed" or item["review_status"] != "reviewed_seed":
            raise ValueError("seed 출처 또는 검수 상태가 올바르지 않습니다.")
        source_fields = item["source_fields"]
        if set(source_fields) != REQUIRED_SOURCE_FIELDS:
            raise ValueError("원본 필드 보존 정보가 불완전합니다.")
        for name in ("user_input", "response_goal", "response_draft", "next_action"):
            if item[name] != source_fields[name]:
                raise ValueError(f"원본 문구가 변경되었습니다: {item['example_id']}:{name}")
        return SituationExample(
            example_id=item["example_id"],
            situation_id=SituationId(item["situation_id"]),
            screen_type=ScreenType(item["screen_type"]),
            user_input=item["user_input"],
            response_goal=item["response_goal"],
            response_draft=item["response_draft"],
            next_action=item["next_action"],
            personalization_tags=tuple(item["personalization_tags"]),
            requires_rag=bool(item["requires_rag"]),
            requires_clarification=bool(item["requires_clarification"]),
            allows_follow_up=bool(item["allows_follow_up"]),
            response_length_mode=ResponseLengthMode(item["response_length_mode"]),
            next_action_code=item["next_action_code"],
            locale=item["locale"],
            source_type=item["source_type"],
            review_status=item["review_status"],
            source_fields=dict(source_fields),
        )
