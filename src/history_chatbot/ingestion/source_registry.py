"""한 줄에 한 자료를 보관하는 JSONL 출처 레지스트리."""

from __future__ import annotations

import json
from pathlib import Path

from history_chatbot.ingestion.models import SourceDocument


class SourceRegistry:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def list(self) -> list[SourceDocument]:
        if not self.manifest_path.exists():
            return []
        documents: list[SourceDocument] = []
        with self.manifest_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    documents.append(SourceDocument.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"manifest {line_number}번째 줄이 유효하지 않습니다: {error}"
                    ) from error
        return documents

    def get(self, document_id: str) -> SourceDocument:
        for document in self.list():
            if document.document_id == document_id:
                return document
        raise KeyError(f"등록되지 않은 document_id입니다: {document_id}")

    def register(self, document: SourceDocument) -> None:
        if any(item.document_id == document.document_id for item in self.list()):
            raise ValueError(f"이미 등록된 document_id입니다: {document.document_id}")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")

    def update(self, document: SourceDocument) -> None:
        documents = self.list()
        for index, item in enumerate(documents):
            if item.document_id == document.document_id:
                documents[index] = document
                break
        else:
            raise KeyError(f"등록되지 않은 document_id입니다: {document.document_id}")
        self._write_all(documents)

    def _write_all(self, documents: list[SourceDocument]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in documents
        )
        self.manifest_path.write_text(content, encoding="utf-8", newline="\n")
