"""Render a phase checkpoint JSON as Markdown without touching source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from history_chatbot.history_collection.reporting import render_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    print(render_markdown(json.loads(args.checkpoint.read_text(encoding="utf-8"))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
