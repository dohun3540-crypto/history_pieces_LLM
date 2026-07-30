"""등록 자료를 추출·정제·청킹하여 JSONL로 출력한다."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from history_chatbot.ingestion.chunker import DocumentChunker
from history_chatbot.ingestion.cleaner import TextCleaner
from history_chatbot.ingestion.models import PipelineResult, ReviewStatus, SourceDocument
from history_chatbot.ingestion.source_registry import SourceRegistry
from history_chatbot.ingestion.text_extractor import extract_text
from history_chatbot.ingestion.validator import (
    can_index_for_service,
    validate_local_path,
    validate_source_document,
)


class IngestionPipeline:
    def __init__(
        self,
        registry: SourceRegistry,
        raw_root: Path,
        extracted_dir: Path,
        processed_dir: Path,
        cleaner: TextCleaner | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.registry = registry
        self.raw_root = raw_root
        self.extracted_dir = extracted_dir
        self.processed_dir = processed_dir
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or DocumentChunker()

    def process(self, document_id: str) -> PipelineResult:
        document = self.registry.get(document_id)
        errors = validate_source_document(document)
        errors.extend(validate_local_path(document, self.raw_root))
        if errors:
            raise ValueError("; ".join(errors))

        extraction = extract_text(Path(document.local_path))
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        extracted_path = self.extracted_dir / f"{document.document_id}.txt"
        extracted_path.write_text(extraction.original_text, encoding="utf-8", newline="\n")

        cleaned = self.cleaner.clean(extraction.original_text)
        updated = replace(document, review_status=ReviewStatus.METADATA_ADDED)
        chunks = tuple(self.chunker.split(cleaned.cleaned_text, updated))
        if not chunks:
            raise ValueError("정제 후 생성된 청크가 없습니다.")

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / f"{document.document_id}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for chunk in chunks:
                record = chunk.to_dict()
                record["cleaning_log"] = list(cleaned.cleaning_log)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.registry.update(updated)
        return PipelineResult(updated, cleaned, chunks, str(output_path))


def copy_reviewed_output(
    processed_path: Path, reviewed_dir: Path, document: SourceDocument
) -> Path:
    """검수 완료 후 호출하는 명시적 승격 도우미."""
    if not can_index_for_service(document):
        raise ValueError("검수 완료되고 RAG 사용이 허용된 자료만 reviewed로 승격할 수 있습니다.")
    if not processed_path.is_file():
        raise FileNotFoundError(f"처리 결과를 찾을 수 없습니다: {processed_path}")
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    target = reviewed_dir / f"{document.document_id}.jsonl"
    target.write_text(processed_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return target
