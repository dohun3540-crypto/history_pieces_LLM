"""Explicit phase gates; thresholds never relax automatically."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from history_chatbot.history_collection.balance import calculate_balance
from history_chatbot.history_collection.models import (
    CandidateDocument, DuplicateStatus, PHASE_TARGETS, Phase, ReviewStatus,
)
from history_chatbot.history_collection.quality import ContentQuality


@dataclass(frozen=True, slots=True)
class GateThresholds:
    min_extraction_success_rate: float = 0.85
    min_rights_known_rate: float = 0.60
    min_mokpo_relevance_rate: float = 0.80
    max_duplicate_rate: float = 0.20
    max_noise_rate: float = 0.20
    max_publisher_share: float = 0.40
    max_topic_share: float = 0.45


@dataclass(frozen=True, slots=True)
class PhaseCheckpoint:
    phase: Phase
    target_unique: int
    discovered: int
    fetched: int
    unique_candidates: int
    duplicate_count: int
    accepted_candidate: int
    needs_human_review: int
    auto_rejected: int
    extraction_success_rate: float
    rights_known_rate: float
    mokpo_relevance_rate: float
    duplicate_rate: float
    noise_rate: float
    publisher_distribution: dict[str, int]
    topic_distribution: dict[str, int]
    largest_publisher_share: float
    largest_topic_share: float
    top_rejection_reasons: dict[str, int]
    gate_status: str
    stop_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_checkpoint(phase: Phase, candidates: Iterable[CandidateDocument],
                     quality: dict[str, ContentQuality], *, discovered: int,
                     fetched: int, thresholds: GateThresholds | None = None) -> PhaseCheckpoint:
    thresholds = thresholds or GateThresholds()
    records = list(candidates)
    unique = [item for item in records if item.duplicate_status != DuplicateStatus.CONFIRMED]
    duplicate_count = len(records) - len(unique)
    balance = calculate_balance(unique)
    count = max(len(records), 1)
    unique_count = len(unique)
    extraction_rate = sum(item.extraction_status == "success" for item in records) / count
    rights_rate = sum(item.rights_status not in {"", "unknown", "unconfirmed"} for item in unique) / max(unique_count, 1)
    relevance_rate = sum(item.mokpo_relevance_score > 0 for item in unique) / max(unique_count, 1)
    duplicate_rate = duplicate_count / count
    noise_rate = sum(quality.get(item.candidate_id) and quality[item.candidate_id].noise for item in records) / count
    rejection_counts = Counter(reason.value for item in records for reason in item.rejection_reasons)
    stop: list[str] = []
    if unique_count < PHASE_TARGETS[phase]: stop.append("target_unique_not_met")
    if extraction_rate < thresholds.min_extraction_success_rate: stop.append("extraction_success_rate")
    if rights_rate < thresholds.min_rights_known_rate: stop.append("rights_known_rate")
    if relevance_rate < thresholds.min_mokpo_relevance_rate: stop.append("mokpo_relevance_rate")
    if duplicate_rate > thresholds.max_duplicate_rate: stop.append("duplicate_rate")
    if noise_rate > thresholds.max_noise_rate: stop.append("noise_rate")
    if balance.largest_publisher_share > thresholds.max_publisher_share: stop.append("publisher_concentration")
    if balance.largest_topic_share > thresholds.max_topic_share: stop.append("topic_concentration")
    return PhaseCheckpoint(
        phase, PHASE_TARGETS[phase], discovered, fetched, unique_count, duplicate_count,
        sum(item.review_status == ReviewStatus.AUTO_CANDIDATE for item in unique),
        sum(item.review_status == ReviewStatus.NEEDS_HUMAN_REVIEW for item in unique),
        sum(item.review_status == ReviewStatus.AUTO_REJECTED for item in unique),
        round(extraction_rate, 4), round(rights_rate, 4), round(relevance_rate, 4),
        round(duplicate_rate, 4), round(noise_rate, 4), balance.publisher_distribution,
        balance.topic_distribution, balance.largest_publisher_share,
        balance.largest_topic_share, dict(rejection_counts.most_common()),
        "STOP" if stop else "PASS", tuple(stop),
    )
