"""제한된 공식 출처 후보 수집 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from history_chatbot.collectors.base import load_collector_configs
from history_chatbot.collectors.pilot import build_pilot_plan, run_pilot
from history_chatbot.collectors.registry import CandidateRegistry
from history_chatbot.collectors.tour_api import TourApiError, TourApiCollector


def _tour_api_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="한국관광공사 Tour API 파일럿")
    parser.add_argument("action", choices=("dry-run", "collect"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/tour_api"))
    parser.add_argument("--extracted-dir", type=Path, default=Path("data/extracted/tour_api"))
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/source_catalog/collected_sources.jsonl"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/sources.jsonl")
    )
    return parser


def _tour_api_main(argv: list[str]) -> None:
    args = _tour_api_parser().parse_args(argv)
    try:
        collector = TourApiCollector.from_environment()
        items = collector.dry_run()
    except TourApiError as error:
        raise SystemExit(str(error)) from None

    print(f"수집 예정 {len(items)}건 (전체 최대 20건, 검색어별 최대 5건)")
    for item in items:
        print(f"- [{item.keyword}] {item.title} (contentid={item.content_id})")
        print(f"  원본 URL: {item.source_url}")
    if args.action == "dry-run":
        print("dry-run 완료: 파일, manifest, RAG 인덱스를 변경하지 않았습니다.")
        return

    result = collector.collect(
        raw_dir=args.raw_dir,
        extracted_dir=args.extracted_dir,
        catalog_path=args.catalog,
        manifest_path=args.manifest,
        prepared_items=items,
    )
    print(f"수집 완료: {len(result.candidates)}건, 제외 {len(result.excluded)}건")


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="네트워크 요청 없이 예정 URL과 수집·건너뜀 이유만 표시합니다(기본값).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="안전 정책을 통과한 출처만 실제 수집합니다.",
    )
    return parser


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "tour-api":
        _tour_api_main(sys.argv[2:])
        return
    args = _parser().parse_args()
    configs = load_collector_configs(args.seed)
    if args.source_id:
        configs = [config for config in configs if config.source_id == args.source_id]
        if not configs:
            raise SystemExit(f"seed에 없는 source-id입니다: {args.source_id}")

    plan = build_pilot_plan(configs)
    print("파일럿 수집 사전 점검:")
    for item in plan:
        disposition = "수집 예정" if item.eligible else "건너뜀"
        urls = ", ".join(item.urls) if item.urls else "(URL 없음)"
        print(f"- [{disposition}] {item.source_id}: {urls}")
        print(f"  이유: {item.reason}")

    if not args.execute:
        print("dry-run 완료: 네트워크 요청과 파일 저장을 수행하지 않았습니다.")
        return

    registry = CandidateRegistry(args.output)
    result = run_pilot(
        configs,
        query=args.query,
        raw_dir=args.raw_dir,
        extracted_dir=args.extracted_dir,
        registry=registry,
    )
    for source_id, count in result.per_source.items():
        print(f"{source_id}: 수집 {count}건")
    for error in result.errors:
        print(f"경고: {error}")
    print(
        f"완료: 총 수집 {len(result.candidates)}건, "
        f"중복 제거 후 신규 {len(result.added)}건"
    )


if __name__ == "__main__":
    main()
