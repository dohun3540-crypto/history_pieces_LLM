"""해커톤 임시 자료의 조회·수집·제거 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from history_chatbot.provisional.service import ProvisionalDataService


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m history_chatbot.provisional.cli")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    commands.add_parser("prepare")
    collect = commands.add_parser("collect")
    collect.add_argument("--force", action="store_true")
    collect.add_argument("--source-id")
    collect.add_argument("--dry-run", action="store_true")
    commands.add_parser("list")
    remove = commands.add_parser("remove")
    group = remove.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-id")
    group.add_argument("--institution")
    commands.add_parser("purge-all")
    commands.add_parser("expire")
    commands.add_parser("rebuild")
    commands.add_parser("reprocess-local")
    return root


def main() -> None:
    args = parser().parse_args()
    service = ProvisionalDataService()
    if args.command == "dry-run":
        print(json.dumps(asdict(service.dry_run()), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        print(json.dumps({"prepared": len(service.prepare_manifest())}, ensure_ascii=False))
    elif args.command == "collect":
        print(
            json.dumps(
                service.collect(
                    force=args.force,
                    source_id=args.source_id,
                    dry_run=args.dry_run,
                ),
                ensure_ascii=False,
                indent=2 if args.dry_run else None,
            )
        )
    elif args.command == "list":
        print(json.dumps(service.load_manifest(), ensure_ascii=False, indent=2))
    elif args.command == "remove":
        removed = service.remove(
            source_id=args.source_id, institution=args.institution
        )
        print(json.dumps({"removed_source_ids": removed}, ensure_ascii=False))
    elif args.command == "purge-all":
        removed = service.remove(purge_all=True, reason="purge_all")
        service.purge_runtime_index()
        print(json.dumps({"removed_source_ids": removed}, ensure_ascii=False))
    elif args.command == "expire":
        print(json.dumps({"expired_source_ids": service.expire()}, ensure_ascii=False))
    elif args.command == "rebuild":
        report = service.rebuild_index()
        print(json.dumps({"report": str(report)}, ensure_ascii=False))
    elif args.command == "reprocess-local":
        print(json.dumps(service.reprocess_local(), ensure_ascii=False))


if __name__ == "__main__":
    main()
