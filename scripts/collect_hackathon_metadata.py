"""CLI for network-safe, metadata/excerpt-only hackathon collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from history_chatbot.collectors.hackathon_metadata import CollectionError, collect, dry_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extracted-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.2)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-response-bytes", type=int, default=1048576)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    try:
        if args.dry_run:
            result = dry_run(args.candidate, args.manifest, args.extracted_dir, args.max_items,
                             args.delay_seconds, args.timeout_seconds, args.max_response_bytes, root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        records = collect(args.candidate, args.manifest, args.extracted_dir, args.max_items,
                          args.delay_seconds, args.timeout_seconds, args.max_response_bytes,
                          repository_root=root)
        print(json.dumps({"saved": [item["document_id"] for item in records]}, ensure_ascii=False))
        return 0
    except CollectionError as exc:
        print("collection refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
