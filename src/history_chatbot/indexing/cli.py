"""검수 완료 자료의 RAG 인덱스 준비 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from history_chatbot.indexing.builder import IndexBuilder
from history_chatbot.indexing.loader import ReviewedChunkLoader
from history_chatbot.ingestion.source_registry import SourceRegistry


EMPTY_MESSAGE = "현재 인덱싱 가능한 검수 완료 문서가 없습니다"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="검수 완료 RAG 입력 준비")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "prepare", "validate", "list-eligible", "list-rejected"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest",
            type=Path,
            default=Path("data/manifests/sources.jsonl"),
        )
        command.add_argument(
            "--raw-root",
            type=Path,
            default=Path("data/raw"),
        )
        command.add_argument(
            "--processed-dir",
            type=Path,
            default=Path("data/processed"),
        )
        command.add_argument(
            "--output-dir",
            type=Path,
            default=Path("data/index_ready"),
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    loader = ReviewedChunkLoader(
        SourceRegistry(args.manifest),
        args.raw_root,
        args.processed_dir,
    )
    builder = IndexBuilder(loader, args.manifest, args.output_dir)
    report = loader.load()

    if args.command == "status":
        print(f"인덱싱 가능 문서: {len(report.eligible)}")
        print(f"제외 문서: {len(report.rejected)}")
        print(f"인덱스 manifest: {'있음' if builder.manifest_path.exists() else '없음'}")
        if not report.eligible:
            print(EMPTY_MESSAGE)
    elif args.command == "list-eligible":
        if not report.eligible:
            print(EMPTY_MESSAGE)
        for loaded in report.eligible:
            print(
                f"{loaded.document.document_id}\t{loaded.document.title}\t"
                f"{len(loaded.chunks)}개 청크"
            )
    elif args.command == "list-rejected":
        for rejected in report.rejected:
            print(
                f"{rejected.document_id}\t{rejected.title}\t"
                f"{'; '.join(rejected.reasons)}"
            )
    elif args.command == "prepare":
        result = builder.prepare()
        if not result.eligible_documents:
            print(EMPTY_MESSAGE)
        print(
            f"준비 완료: 문서 {result.eligible_documents}건, "
            f"청크 {result.chunk_count}건, 중복 제거 {result.duplicate_chunks}건"
        )
        print(f"청크: {result.chunks_path}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "validate":
        errors = builder.validate()
        if errors:
            raise SystemExit("검증 실패: " + "; ".join(errors))
        print("index_ready 검증 성공")


if __name__ == "__main__":
    main()
