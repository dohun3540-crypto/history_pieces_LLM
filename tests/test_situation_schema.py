import json

import pytest

from history_chatbot.dialogue.seed_loader import SituationSeedLoader
from history_chatbot.dialogue.situation_models import SituationId
from scripts.export_giroksae_seed import SOURCE, parse


def test_all_20_situations_and_63_examples_load() -> None:
    examples = SituationSeedLoader().load()
    assert len(SituationId) == 20
    assert len(examples) == 63
    assert len({item.example_id for item in examples}) == 63
    assert {item.situation_id for item in examples} == set(SituationId)
    reviewed = examples[:54]
    proposed = examples[54:]
    assert all(item.source_type == "human_authored_seed" for item in reviewed)
    assert all(item.review_status == "reviewed_seed" for item in reviewed)
    assert all(item.source_type == "human_proposed_seed" for item in proposed)
    assert all(item.review_status == "review_pending" for item in proposed)


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


def test_legacy_loader_and_original_text_are_unchanged() -> None:
    saved = json.loads(open("configs/giroksae_situations.json", encoding="utf-8").read())["examples"]
    legacy = SituationSeedLoader(additions_path=None).load()
    assert len(legacy) == 54
    assert [item.example_id for item in legacy] == [item["example_id"] for item in saved]
    assert [item.response_draft_original for item in legacy] == [item["response_draft"] for item in saved]


def test_v03_response_original_and_runtime_template_are_separate() -> None:
    proposed = SituationSeedLoader().load()[54:]
    assert {item.example_id for item in proposed} == {
        "TECH_01", "TECH_02", "TECH_03", "NAV_01", "NAV_02", "NAV_03",
        "SAFE_01", "SAFE_02", "SAFE_03",
    }
    assert all(item.response_draft_original == item.response_draft for item in proposed)
    assert all(item.response_template for item in proposed)


def test_loader_rejects_unknown_enum_and_wrong_type(tmp_path) -> None:
    payload = json.loads(open("configs/giroksae_situations_v03_additions.json", encoding="utf-8").read())
    payload["examples"][0]["required_context"] = ["unknown_context"]
    path = tmp_path / "bad-enum.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        SituationSeedLoader(additions_path=path).load()

    payload["examples"][0]["required_context"] = "app_state"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="문자열 배열"):
        SituationSeedLoader(additions_path=path).load()
