"""작은 고정 평가셋을 위한 검색 backend 비교 도구."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from history_chatbot.retrieval.base import RankedChunk


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    locale: str
    query: str
    topic: str
    expected_source_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    expected_top_k: int
    should_answer: bool
    should_refuse: bool
    notes: str


def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("corpus_lane") != "provisional_hackathon":
        raise ValueError("평가 fixture의 corpus lane이 올바르지 않습니다.")
    return tuple(
        EvaluationCase(
            case_id=item["case_id"], locale=item["locale"], query=item["query"],
            topic=item["topic"], expected_source_ids=tuple(item["expected_source_ids"]),
            expected_chunk_ids=tuple(item["expected_chunk_ids"]),
            expected_top_k=int(item["expected_top_k"]), should_answer=bool(item["should_answer"]),
            should_refuse=bool(item["should_refuse"]), notes=item["notes"],
        )
        for item in payload["cases"]
    )


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * .95) - 1)]


def evaluate_backend(
    name: str,
    cases: tuple[EvaluationCase, ...],
    search: Callable[[str], list[RankedChunk]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results = search(case.query)
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        ids = [item.chunk.chunk_id for item in results]
        expected = set(case.expected_chunk_ids)
        first_rank = next((rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in expected), None)
        rows.append({
            "case_id": case.case_id,
            "locale": case.locale,
            "query": case.query,
            "returned_chunk_ids": ids[:5],
            "returned_source_ids": [str(x.chunk.payload.get("source_id", x.chunk.document_id)) for x in results[:5]],
            "scores": [round(x.score, 6) for x in results[:5]],
            "first_relevant_rank": first_rank,
            "refused": not results,
            "latency_ms": round(latency, 3),
        })
    answerable = [(case, row) for case, row in zip(cases, rows) if case.should_answer]
    refuse = [(case, row) for case, row in zip(cases, rows) if case.should_refuse]

    def recall(k: int, subset=answerable) -> float:
        if not subset:
            return 0.0
        return sum(bool(set(row["returned_chunk_ids"][:k]) & set(case.expected_chunk_ids)) for case, row in subset) / len(subset)

    def source_recall(k: int, subset=answerable) -> float:
        if not subset:
            return 0.0
        return sum(bool(set(row["returned_source_ids"][:k]) & set(case.expected_source_ids)) for case, row in subset) / len(subset)

    zh = [(case, row) for case, row in answerable if case.locale == "zh-CN"]
    ko = [(case, row) for case, row in answerable if case.locale == "ko"]
    metrics = {
        "recall_at_1": round(recall(1), 4),
        "recall_at_3": round(recall(3), 4),
        "recall_at_5": round(recall(5), 4),
        "source_recall_at_1": round(source_recall(1), 4),
        "source_recall_at_3": round(source_recall(3), 4),
        "source_recall_at_5": round(source_recall(5), 4),
        "mrr": round(sum(1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0 for _, row in answerable) / len(answerable), 4),
        "no_evidence_precision": round(sum(row["refused"] for _, row in refuse) / len(refuse), 4),
        "unrelated_rejection_rate": round(sum(row["refused"] for case, row in refuse if case.topic == "무관") / sum(case.topic == "무관" for case, _ in refuse), 4),
        "insufficient_rejection_rate": round(sum(row["refused"] for case, row in refuse if case.topic == "근거 부족") / sum(case.topic == "근거 부족" for case, _ in refuse), 4),
        "zh_cross_language_success_at_3": round(recall(3, zh), 4),
        "zh_source_success_at_3": round(source_recall(3, zh), 4),
        "ko_recall_at_1": round(recall(1, ko), 4),
        "ko_recall_at_3": round(recall(3, ko), 4),
        "ko_recall_at_5": round(recall(5, ko), 4),
        "zh_recall_at_1": round(recall(1, zh), 4),
        "zh_recall_at_3": round(recall(3, zh), 4),
        "zh_recall_at_5": round(recall(5, zh), 4),
        "average_latency_ms": round(statistics.mean(latencies), 3),
        "p95_latency_ms": round(percentile95(latencies), 3),
        "answerable_cases": len(answerable),
        "refusal_cases": len(refuse),
    }
    return {"backend": name, "metrics": metrics, "cases": rows}
