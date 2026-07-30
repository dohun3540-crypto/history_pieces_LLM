"""하이브리드 검색 인덱스 관리 및 진단 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from history_chatbot.retrieval.service import HybridRetrievalService, RetrievalConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="목포 근대역사 하이브리드 검색")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/retrieval.yaml")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-models")
    build = commands.add_parser("build-index")
    build.add_argument("--rebuild", action="store_true")
    commands.add_parser("status")
    search = commands.add_parser("search")
    search.add_argument("question")
    commands.add_parser("benchmark")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "inspect-models":
        print("docs/RETRIEVAL_MODEL_REPORT.md를 확인하세요. 모델은 다운로드되지 않았습니다.")
        return
    config = RetrievalConfig.load(args.config)
    service = HybridRetrievalService(config)
    if args.command == "build-index":
        report = service.build_index(force=args.rebuild)
        if report.chunks == 0:
            print("현재 검색 인덱스에 넣을 index_ready 청크가 없습니다.")
        print(json.dumps(asdict(report), ensure_ascii=False, default=str, indent=2))
    elif args.command == "status":
        print(json.dumps(service.status(), ensure_ascii=False, indent=2))
    elif args.command == "search":
        results = service.search(args.question)
        if not results:
            print("검색 근거가 없습니다.")
        for result in results:
            print(
                f"{result.score:.4f}\t{'+'.join(result.methods)}\t"
                f"{result.chunk.chunk_id}\t{result.chunk.title}\t"
                f"{result.chunk.source_url}"
            )
    elif args.command == "benchmark":
        questions = (
            "목포는 언제 개항했나요?",
            "목포 해관의 역할은 무엇인가요?",
            "목포 근대역사문화공간은 무엇인가요?",
            "목포출신 최초 우주비행사는 누구인가요?",
            "서울의 궁궐을 알려줘.",
            "자료에 전혀 없는 질문",
        )
        for question in questions:
            print(f"{question}\t{len(service.search(question))}건")


if __name__ == "__main__":
    main()
