"""Run deterministic conversation evaluation on the actual orchestrator path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from history_chatbot.chat.service import create_hackathon_orchestrator
from history_chatbot.evaluation.conversation_quality import (
    evaluate_scenarios,
    load_dataset_directory,
)
from history_chatbot.models.mock_llm import MockLLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation/conversation_quality"))
    parser.add_argument("--split", choices=("train_dev", "validation", "holdout_test", "all"), default="all")
    parser.add_argument("--backend", choices=("mock", "configured"), default="mock")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenarios = load_dataset_directory(args.dataset_dir)
    if args.split != "all":
        scenarios = [item for item in scenarios if item["split"] == args.split]
    llm = MockLLM("evaluation fallback") if args.backend == "mock" else None
    orchestrator = create_hackathon_orchestrator(
        runtime_dir=Path(".runtime/conversation-evaluation"),
        session_path=Path(".runtime/conversation-evaluation/sessions.json"),
        llm=llm,
    )
    report = evaluate_scenarios(orchestrator, scenarios)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
