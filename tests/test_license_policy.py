from history_chatbot.ingestion.license_policy import (
    can_use_for_rag,
    can_use_for_training,
    license_policy_errors,
)
from history_chatbot.ingestion.models import CopyrightStatus


def test_unknown_copyright_cannot_be_used(tmp_path, source_factory) -> None:
    source = source_factory(
        tmp_path / "raw.txt",
        copyright_status=CopyrightStatus.UNKNOWN,
        allowed_for_rag=True,
        allowed_for_training=True,
    )
    assert not can_use_for_rag(source)
    assert not can_use_for_training(source)


def test_restricted_document_cannot_be_used_for_training(tmp_path, source_factory) -> None:
    source = source_factory(
        tmp_path / "raw.txt",
        copyright_status=CopyrightStatus.RESTRICTED,
        allowed_for_rag=False,
        allowed_for_training=True,
    )
    assert not can_use_for_training(source)


def test_required_attribution_must_have_text(tmp_path, source_factory) -> None:
    source = source_factory(tmp_path / "raw.txt", attribution_text="")
    assert any("attribution_text" in error for error in license_policy_errors(source))
