from pathlib import Path
from typing import Any

import pytest

from history_chatbot.ingestion.models import (
    CopyrightStatus,
    ReviewStatus,
    SourceDocument,
)


@pytest.fixture
def source_factory():
    def factory(local_path: Path, **overrides: Any) -> SourceDocument:
        values: dict[str, Any] = {
            "document_id": "test-virtual-001",
            "title": "테스트용 가상 자료",
            "source_type": "test_fixture",
            "publisher": "테스트 전용 발행처",
            "author": "테스트 작성자",
            "source_url": "https://example.invalid/test-virtual-001",
            "local_path": str(local_path),
            "published_date": "2026-01-01",
            "accessed_date": "2026-07-30",
            "language": "ko",
            "license_name": "테스트 전용 허가",
            "license_url": "https://example.invalid/license",
            "copyright_status": CopyrightStatus.PERMISSION_GRANTED,
            "allowed_for_rag": True,
            "allowed_for_training": False,
            "redistribution_allowed": False,
            "attribution_required": True,
            "attribution_text": "테스트용 가상 자료 — 실제 역사 자료 아님",
            "notes": "자동 테스트 전용",
            "review_status": ReviewStatus.DRAFT,
            "reviewed_by": "",
            "reviewed_at": "",
        }
        values.update(overrides)
        return SourceDocument(**values)

    return factory
