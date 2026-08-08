"""Bounded CLI for public Mokpo history batch discovery and collection."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

from history_chatbot.collectors.public_history_batch import (
    ADAPTERS, BatchError, BatchPipeline, RequestController, UrllibBatchTransport,
)


DEFAULT_KEYWORDS = (
    "목포 개항", "목포 해관", "목포 외국인 거류지 조계지", "목포 근대역사문화공간",
    "구 목포 일본영사관", "동양척식주식회사 목포지점", "목포 근대 항만 철도",
    "일제강점기 목포",
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--batch-id", required=True)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--catalog", type=Path)
    value.add_argument("--extracted-dir", type=Path)
    value.add_argument("--report-json", type=Path)
    value.add_argument("--report-md", type=Path)
    value.add_argument("--source", action="append", required=True, choices=sorted(ADAPTERS))
    value.add_argument("--keyword", action="append")
    value.add_argument("--max-accepted", type=int, default=10)
    value.add_argument("--max-per-source", type=int, default=2)
    value.add_argument("--max-requests", type=int, default=75)
    value.add_argument("--delay-seconds", type=float, default=1.2)
    value.add_argument("--timeout-seconds", type=float, default=15.0)
    value.add_argument("--max-response-bytes", type=int, default=1048576)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--discover", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    value.add_argument("--no-write", action="store_true",
                       help="required safety acknowledgement for smoke-test mode")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    path_names = ("manifest", "catalog", "extracted_dir", "report_json", "report_md")
    if args.smoke_test:
        if not args.no_write:
            parser().error("--smoke-test requires --no-write")
        if any(getattr(args, name) is not None for name in path_names):
            parser().error("smoke-test does not accept output paths")
    elif any(getattr(args, name) is None for name in path_names):
        parser().error("dry-run, discover, and execute require all output path options")
    paths = {"manifest": args.manifest, "catalog": args.catalog, "extracted_dir": args.extracted_dir,
             "report_json": args.report_json, "report_md": args.report_md}
    limits = {"max_accepted": args.max_accepted, "max_per_source": args.max_per_source,
              "max_requests": args.max_requests, "delay_seconds": args.delay_seconds}
    try:
        if args.dry_run:
            result = BatchPipeline(ADAPTERS).dry_run(args.source, paths, limits)
        else:
            controller = RequestController(
                args.max_requests, args.delay_seconds,
                lambda hosts: UrllibBatchTransport(hosts), max_retries=1,
            )
            pipeline = BatchPipeline(ADAPTERS, controller)
            if args.smoke_test:
                result = pipeline.smoke_test(args.source, os.environ,
                                             args.timeout_seconds, args.max_response_bytes)
            elif args.discover:
                result = pipeline.discover(args.batch_id, args.source, args.keyword or DEFAULT_KEYWORDS,
                                           args.catalog, args.report_json, args.report_md, os.environ,
                                           args.timeout_seconds, args.max_response_bytes, limits)
            else:
                result = pipeline.execute(args.batch_id, args.source, args.catalog, args.manifest,
                                          args.extracted_dir, args.report_json, args.report_md,
                                          args.timeout_seconds, args.max_response_bytes, limits,
                                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), os.environ)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BatchError as exc:
        print("batch refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
