"""development fixture 기반 대화형 RAG CLI."""

from __future__ import annotations

import argparse
import json

from history_chatbot.chat.service import create_development_orchestrator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="개발용 가상 fixture RAG 대화")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--session-id")
    ask.add_argument("--locale", default="ko")
    ask.add_argument("--top-k", type=int, default=3)
    reset = commands.add_parser("reset")
    reset.add_argument("--session-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    orchestrator = create_development_orchestrator()
    if args.command == "reset":
        print(
            json.dumps(
                {"session_id": args.session_id, "reset": orchestrator.reset(args.session_id)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    response = orchestrator.ask(
        args.question,
        session_id=args.session_id,
        locale=args.locale,
        top_k=args.top_k,
    )
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
