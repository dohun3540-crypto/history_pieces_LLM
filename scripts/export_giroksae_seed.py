"""사람이 작성한 기록새 Markdown 원본을 편집 가능한 JSON seed로 변환한다."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "giroksae_situation_seed.md"
TARGET = ROOT / "configs" / "giroksae_situations.json"

SCREEN_TYPES = {
    "첫 등장": "intro",
    "조각 연동 대화": "piece_chat",
    "전용 채팅창": "free_chat",
    "조각 연동 대화 또는 전용 채팅창": "piece_chat_or_free_chat",
}

RAG_DEFAULT = {
    "INTEREST_PEOPLE",
    "INTEREST_DAILY_CITY",
    "HISTORY_FACT_QUESTION",
    "EVIDENCE_AND_CORRECTION",
    "CROSS_CULTURAL_COMPARISON",
}
MIXED = {
    "INTEREST_ARCHITECTURE",
    "JOURNEY_CONTEXT_QUESTION",
    "COMPARISON_CONTEXT",
    "EMOTION_NEGATIVE_HISTORY",
    "PERSONAL_AND_LIGHT_CHAT",
    "RESPONSE_STYLE_REQUEST",
}


def _cells(line: str) -> list[str]:
    return [value.strip() for value in line.strip().strip("|").split("|")]


def _tags(value: str) -> list[str]:
    if value == "없음":
        return []
    return re.findall(r"`([^`]+)`", value)


def _derived(situation_id: str, example_id: str, user_input: str, next_action: str) -> dict:
    fact_markers = ("왜", "언제", "누구", "관계", "자료", "출처", "역사", "건립", "구조")
    requires_rag = situation_id in RAG_DEFAULT or (
        situation_id in MIXED and any(marker in user_input for marker in fact_markers)
    )
    requires_clarification = example_id in {"EVIDENCE_03", "CULTURE_03"}
    if example_id == "DISSATISFIED_04":
        length = "very_short"
    elif any(value in user_input for value in ("짧게", "말 ㅈㄴ 많")):
        length = "short"
    elif "자세히" in user_input:
        length = "detailed"
    elif "정리" in user_input:
        length = "summary"
    elif "출처" in user_input:
        length = "source_view"
    else:
        length = "default"
    return {
        "requires_rag": requires_rag,
        "requires_clarification": requires_clarification,
        "allows_follow_up": next_action not in {"다음 단계", "재답변 또는 종료"},
        "response_length_mode": length,
        "next_action_code": re.sub(r"[^A-Za-z0-9가-힣]+", "_", next_action).strip("_").lower(),
        "locale": "ko",
        "source_type": "human_authored_seed",
        "review_status": "reviewed_seed",
    }


def parse(text: str) -> dict:
    examples: list[dict] = []
    situations: list[dict] = []
    section_pattern = re.compile(
        r"\*\*상황 코드:\*\*\s*`(?P<sid>[A-Z_]+)`(?P<body>.*?)(?=\n# \d+\.|\n# 현재 초안 규모)",
        re.S,
    )
    for match in section_pattern.finditer(text):
        sid = match.group("sid")
        body = match.group("body")
        screen_match = re.search(r"\*\*화면:\*\*\s*([^\r\n]+)", body)
        if not screen_match or screen_match.group(1).strip() not in SCREEN_TYPES:
            raise ValueError(f"알 수 없는 화면 값: {sid}")
        screen_label = screen_match.group(1).strip()
        screen_type = SCREEN_TYPES[screen_label]
        count = 0
        for line in body.splitlines():
            if not re.match(r"^\| [A-Z]+(?:_[A-Z]+)*_\d{2} \|", line):
                continue
            cells = _cells(line)
            if len(cells) != 6:
                raise ValueError(f"원본 표 열 수가 다릅니다: {line}")
            example_id, user_input, goal, draft, action, raw_tags = cells
            item = {
                "example_id": example_id,
                "situation_id": sid,
                "screen_type": screen_type,
                "user_input": user_input,
                "response_goal": goal,
                "response_draft": draft,
                "next_action": action,
                "personalization_tags": _tags(raw_tags),
                **_derived(sid, example_id, user_input, action),
                "source_fields": {
                    "screen_type": screen_label,
                    "user_input": user_input,
                    "response_goal": goal,
                    "response_draft": draft,
                    "next_action": action,
                    "personalization_tags": raw_tags,
                },
            }
            examples.append(item)
            count += 1
        situations.append({"situation_id": sid, "screen_type": screen_type, "example_count": count})
    return {
        "schema_version": 1,
        "source_document": "docs/giroksae_situation_seed.md",
        "source_type": "human_authored_seed",
        "review_status": "reviewed_seed",
        "situations": situations,
        "examples": examples,
    }


def main() -> None:
    payload = parse(SOURCE.read_text(encoding="utf-8"))
    if len(payload["situations"]) != 17 or len(payload["examples"]) != 54:
        raise ValueError("원본은 정확히 17개 상황과 54개 사례여야 합니다.")
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
