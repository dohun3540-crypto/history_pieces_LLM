"""Deterministic checks and review queues for grounded multi-turn conversations."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


REQUIRED_SCENARIO_FIELDS = {
    "scenario_id", "split", "topic_group", "source_evidence",
    "conversation_turns", "training_eligible", "notes",
}
REQUIRED_TURN_FIELDS = {
    "turn_id", "user_message", "expected_behavior", "answerability",
    "evaluation_tags", "hallucination_risk", "notes",
}
VALID_SPLITS = {"train_dev", "validation", "holdout_test"}
VALID_ANSWERABILITY = {
    "answerable", "partially_answerable", "unanswerable", "out_of_scope",
}


class DatasetValidationError(ValueError):
    pass


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for scenario in scenarios:
        missing = REQUIRED_SCENARIO_FIELDS - set(scenario)
        if missing:
            raise DatasetValidationError(
                f"{scenario.get('scenario_id', path.name)} missing {sorted(missing)}"
            )
        if scenario["split"] not in VALID_SPLITS:
            raise DatasetValidationError(f"invalid split: {scenario['split']}")
        if scenario["training_eligible"] is not False:
            raise DatasetValidationError(
                "current conversation corpus must remain evaluation-only"
            )
        seen_turns: set[int] = set()
        for turn in scenario["conversation_turns"]:
            missing_turn = REQUIRED_TURN_FIELDS - set(turn)
            if missing_turn:
                raise DatasetValidationError(
                    f"{scenario['scenario_id']} turn missing {sorted(missing_turn)}"
                )
            if turn["answerability"] not in VALID_ANSWERABILITY:
                raise DatasetValidationError("invalid answerability")
            if turn["turn_id"] in seen_turns:
                raise DatasetValidationError("duplicate turn_id")
            seen_turns.add(turn["turn_id"])
    return scenarios


def validate_splits(scenarios: Iterable[dict[str, Any]]) -> dict[str, int]:
    scenario_split: dict[str, str] = {}
    topic_split: dict[str, str] = {}
    evidence_split: dict[str, str] = {}
    counts: Counter[str] = Counter()
    turn_counts: Counter[str] = Counter()
    for scenario in scenarios:
        split = str(scenario["split"])
        _claim_unique(scenario_split, str(scenario["scenario_id"]), split, "scenario")
        _claim_unique(topic_split, str(scenario["topic_group"]), split, "topic_group")
        for source in scenario["source_evidence"]:
            _claim_unique(
                evidence_split, str(source["document_id"]), split, "document_id"
            )
            if source.get("allowed_for_training") is not False:
                raise DatasetValidationError(
                    "source evidence permission must be explicit and false"
                )
        counts[split] += 1
        turn_counts[split] += len(scenario["conversation_turns"])
    return {
        **{f"{split}_scenarios": counts[split] for split in sorted(VALID_SPLITS)},
        **{f"{split}_turns": turn_counts[split] for split in sorted(VALID_SPLITS)},
    }


def _claim_unique(index: dict[str, str], key: str, split: str, kind: str) -> None:
    previous = index.setdefault(key, split)
    if previous != split:
        raise DatasetValidationError(
            f"leakage: {kind} {key!r} occurs in {previous} and {split}"
        )


def answer_is_complete(answer: str) -> bool:
    value = answer.strip()
    if not value:
        return False
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in value:
        if character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return not stack and not bool(re.search(r"(?:\(|\[|\{|[,;:]|[A-Za-z]\.)\s*$", value))


def repeated_sentence_count(answer: str) -> int:
    sentences = [
        re.sub(r"\s+", " ", value.strip())
        for value in re.split(r"(?<=[.!?。！？])\s+", answer)
        if value.strip()
    ]
    return len(sentences) - len(set(sentences))


def evaluate_scenarios(orchestrator: Any, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the real orchestrator path; model-dependent dimensions stay review-only."""

    outcomes: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    automatic: Counter[str] = Counter()
    fallback_answers: list[str] = []
    for scenario in scenarios:
        session_id = None
        for turn in scenario["conversation_turns"]:
            response = orchestrator.ask(
                turn["user_message"],
                session_id=session_id,
                conversation_mode="free_chat",
                screen_type="free_chat",
            )
            session_id = response.session_id
            metadata = response.context_metadata or {}
            complete = answer_is_complete(response.answer)
            grounded_contract = (
                (response.grounded and bool(response.evidence) and bool(response.sources))
                or (not response.grounded and not response.evidence and not response.sources)
            )
            context_expected = turn.get("expected_contextualized_query")
            context_ok = (
                context_expected is None
                or str(context_expected) == str(metadata.get("search_query", ""))
            )
            out_of_scope_ok = not (
                turn["answerability"] == "out_of_scope" and response.grounded
            )
            if response.status == "insufficient_evidence":
                fallback_answers.append(response.answer)
            if not complete:
                failures["TRUNCATION_FAILURE"] += 1
            if not grounded_contract:
                failures["HALLUCINATION"] += 1
            if not context_ok:
                failures["CONTEXTUALIZATION_FAILURE"] += 1
            if not out_of_scope_ok:
                failures["OUT_OF_SCOPE_FAILURE"] += 1
            if turn["answerability"] == "answerable" and not response.grounded:
                failures["RETRIEVAL_FAILURE"] += 1
            automatic["turns"] += 1
            automatic["complete"] += int(complete)
            automatic["grounded_contract"] += int(grounded_contract)
            automatic["context_expected"] += int(context_expected is not None)
            automatic["context_correct"] += int(context_expected is not None and context_ok)
            automatic["out_of_scope"] += int(turn["answerability"] == "out_of_scope")
            automatic["out_of_scope_correct"] += int(
                turn["answerability"] == "out_of_scope" and out_of_scope_ok
            )
            outcomes.append({
                "scenario_id": scenario["scenario_id"],
                "turn_id": turn["turn_id"],
                "status": response.status,
                "grounded": response.grounded,
                "used_chunks": response.used_chunks,
                "search_query": metadata.get("search_query"),
                "complete": complete,
                "grounded_contract": grounded_contract,
                "context_ok": context_ok,
                "answer_preview": response.answer[:240],
            })
    repeated_fallbacks = sum(
        count - 1 for count in Counter(fallback_answers).values() if count > 1
    )
    if repeated_fallbacks:
        failures["FALLBACK_FAILURE"] += repeated_fallbacks
    backend = str(getattr(orchestrator.llm, "backend_name", "unknown"))
    return {
        "backend": backend,
        "llm_dependent_metrics": {
            "groundedness_claim_review": "unavailable" if backend == "mock" else "manual_review_required",
            "directness": "unavailable" if backend == "mock" else "manual_review_required",
            "fallback_naturalness": "unavailable" if backend == "mock" else "manual_review_required",
            "overall_conversational_quality": "unavailable" if backend == "mock" else "manual_review_required",
        },
        "automatic_metrics": {
            "turns": automatic["turns"],
            "grounded_contract_pass": automatic["grounded_contract"],
            "completeness_pass": automatic["complete"],
            "context_retention_pass": automatic["context_correct"],
            "context_retention_total": automatic["context_expected"],
            "out_of_scope_pass": automatic["out_of_scope_correct"],
            "out_of_scope_total": automatic["out_of_scope"],
            "repeated_fallbacks": repeated_fallbacks,
        },
        "failure_counts": dict(sorted(failures.items())),
        "outcomes": outcomes,
    }


def load_dataset_directory(directory: Path) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for name in ("train_dev.jsonl", "validation.jsonl", "holdout_test.jsonl"):
        scenarios.extend(load_scenarios(directory / name))
    validate_splits(scenarios)
    return scenarios
