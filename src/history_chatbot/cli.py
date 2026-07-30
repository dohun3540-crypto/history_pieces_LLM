"""한국어 입출력을 지원하는 명령행 인터페이스."""

from __future__ import annotations

import argparse

from history_chatbot.app import HistoryChatbot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="목포 근대역사 RAG 챗봇 프로토타입")
    parser.add_argument("question", nargs="?", help="한 번만 질문하고 종료합니다.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chatbot = HistoryChatbot.from_settings()
    if args.question:
        print(chatbot.ask(args.question))
        return

    print("목포 근대역사 챗봇입니다. 종료하려면 '종료'를 입력하세요.")
    while True:
        try:
            question = input("질문> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.strip().lower() in {"종료", "quit", "exit"}:
            break
        print(chatbot.ask(question))


if __name__ == "__main__":
    main()
