import json

import pytest

from history_chatbot.dialogue.seed_loader import SituationSeedLoader
from history_chatbot.dialogue.situation_models import SituationId
from scripts.export_giroksae_seed import SOURCE, parse


def test_all_17_situations_and_54_human_authored_examples_load() -> None:
    examples = SituationSeedLoader().load()
    assert len(SituationId) == 17
    assert len(examples) == 54
    assert len({item.example_id for item in examples}) == 54
    assert {item.situation_id for item in examples} == set(SituationId)
    assert all(item.source_type == "human_authored_seed" for item in examples)
    assert all(item.review_status == "reviewed_seed" for item in examples)


def test_exported_source_fields_match_markdown_parser_exactly() -> None:
    generated = parse(SOURCE.read_text(encoding="utf-8"))["examples"]
    saved = json.loads(open("configs/giroksae_situations.json", encoding="utf-8").read())["examples"]
    assert [item["example_id"] for item in saved] == [item["example_id"] for item in generated]
    assert [item["source_fields"] for item in saved] == [item["source_fields"] for item in generated]
    assert "사용자 발화 예시: **54개**" in SOURCE.read_text(encoding="utf-8")


def test_loader_rejects_duplicate_example_id(tmp_path) -> None:
    payload = json.loads(open("configs/giroksae_situations.json", encoding="utf-8").read())
    payload["examples"][1]["example_id"] = payload["examples"][0]["example_id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        SituationSeedLoader(path).load()


def test_every_example_has_required_preserved_fields() -> None:
    for item in SituationSeedLoader().load():
        assert item.situation_id and item.screen_type and item.user_input
        assert item.response_goal and item.response_draft and item.next_action
        assert set(item.source_fields) == {
            "screen_type", "user_input", "response_goal", "response_draft",
            "next_action", "personalization_tags",
        }
