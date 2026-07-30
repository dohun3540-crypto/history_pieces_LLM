"""UTF-8 JSON 문서 로더."""

import json
from pathlib import Path

from history_chatbot.retrieval.document import Document


def load_json_documents(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    items = payload["documents"] if isinstance(payload, dict) else payload
    return [
        Document(
            id=str(item["id"]),
            title=item["title"],
            source=item["source"],
            content=item["content"],
            language=item.get("language", "ko"),
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]
