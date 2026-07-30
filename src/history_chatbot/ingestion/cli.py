"""자료 등록·처리·검증·목록 조회 CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from history_chatbot.ingestion.models import SourceDocument
from history_chatbot.ingestion.pipeline import IngestionPipeline
from history_chatbot.ingestion.source_registry import SourceRegistry
from history_chatbot.ingestion.validator import (
    can_index_for_service,
    validate_local_path,
    validate_source_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="목포 근대역사 자료 수집·정제 도구")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "process", "validate", "list"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", required=True, type=Path)
        if name in {"process", "validate"}:
            command.add_argument("--document-id", required=True)
        if name == "register":
            command.add_argument("--metadata", required=True, type=Path)
    return parser


def _project_data_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def main() -> None:
    args = _parser().parse_args()
    registry = SourceRegistry(args.manifest)
    if args.command == "register":
        payload = json.loads(args.metadata.read_text(encoding="utf-8"))
        document = SourceDocument.from_dict(payload)
        errors = validate_source_document(document)
        if errors:
            raise SystemExit("등록 실패: " + "; ".join(errors))
        registry.register(document)
        print(f"등록 완료: {document.document_id}")
    elif args.command == "list":
        for document in registry.list():
            print(
                f"{document.document_id}\t{document.review_status.value}\t{document.title}"
            )
    elif args.command == "validate":
        document = registry.get(args.document_id)
        errors = validate_source_document(document)
        errors.extend(validate_local_path(document, _project_data_root() / "raw"))
        if errors:
            raise SystemExit("검증 실패: " + "; ".join(errors))
        eligibility = "가능" if can_index_for_service(document) else "불가"
        print(f"메타데이터 검증 성공 | 서비스 RAG 색인: {eligibility}")
    elif args.command == "process":
        root = _project_data_root()
        result = IngestionPipeline(
            registry,
            root / "raw",
            root / "extracted",
            root / "processed",
        ).process(args.document_id)
        print(f"처리 완료: {len(result.chunks)}개 청크 -> {result.output_path}")


if __name__ == "__main__":
    main()
