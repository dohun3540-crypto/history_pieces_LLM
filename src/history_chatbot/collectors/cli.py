"""제한된 공식 출처 후보 수집 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from history_chatbot.collectors.base import load_collector_configs
from history_chatbot.collectors.registry import CandidateRegistry, build_collector


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="목포 근대역사 공식 자료 후보 수집기")
    parser.add_argument("--seed", type=Path, default=Path("data/source_catalog/seed_sources.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/source_catalog/collected_sources.jsonl")
    )
    parser.add_argument("--source-id", help="지정하지 않으면 seed의 모든 출처를 사용합니다.")
    parser.add_argument("--query", default="목포", help="후보 링크 필터 검색어")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/collected"))
    parser.add_argument("--extracted-dir", type=Path, default=Path("data/extracted/collected"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    configs = load_collector_configs(args.seed)
    if args.source_id:
        configs = [config for config in configs if config.source_id == args.source_id]
        if not configs:
            raise SystemExit(f"seed에 없는 source-id입니다: {args.source_id}")

    registry = CandidateRegistry(args.output)
    total_found = total_added = 0
    for config in configs:
        collector = build_collector(config)
        report = collector.collect(
            args.query, raw_dir=args.raw_dir, extracted_dir=args.extracted_dir
        )
        added = registry.add_new(report.candidates)
        total_found += len(report.candidates)
        total_added += len(added)
        print(
            f"{config.source_id}: 발견 {len(report.candidates)}, "
            f"신규 {len(added)}, 오류 {len(report.errors)}"
        )
        for error in report.errors:
            print(f"  경고: {error}")
    print(f"완료: 발견 {total_found}, 중복 제거 후 신규 {total_added}")


if __name__ == "__main__":
    main()
