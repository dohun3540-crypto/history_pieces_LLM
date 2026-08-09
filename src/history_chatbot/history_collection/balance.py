"""Publisher and topic concentration metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from history_chatbot.history_collection.models import CandidateDocument, DuplicateStatus


@dataclass(frozen=True, slots=True)
class BalanceReport:
    publisher_distribution: dict[str, int]
    topic_distribution: dict[str, int]
    largest_publisher_share: float
    largest_topic_share: float


def calculate_balance(candidates: Iterable[CandidateDocument]) -> BalanceReport:
    unique = [item for item in candidates if item.duplicate_status != DuplicateStatus.CONFIRMED]
    publishers = Counter(item.publisher_family or item.publisher or "unknown" for item in unique)
    topics = Counter(topic.value for item in unique for topic in set(item.topic_categories))
    total = len(unique)
    return BalanceReport(dict(publishers.most_common()), dict(topics.most_common()),
                         round(max(publishers.values(), default=0) / max(total, 1), 4),
                         round(max(topics.values(), default=0) / max(total, 1), 4))
