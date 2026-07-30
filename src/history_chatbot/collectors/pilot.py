"""모든 수집 진입점에 적용되는 파일럿 안전 정책."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from history_chatbot.collectors.base import (
    BaseCollector,
    CollectedCandidate,
    CollectorConfig,
    collection_skip_reason,
)
from history_chatbot.collectors.registry import CandidateRegistry, build_collector


MAX_TOTAL_RESULTS = 10
MAX_RESULTS_PER_SOURCE = 2


@dataclass(frozen=True, slots=True)
class PilotPlanItem:
    source_id: str
    name: str
    urls: tuple[str, ...]
    eligible: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PilotRunResult:
    candidates: tuple[CollectedCandidate, ...]
    added: tuple[CollectedCandidate, ...]
    errors: tuple[str, ...]
    per_source: dict[str, int]


def build_pilot_plan(configs: list[CollectorConfig]) -> list[PilotPlanItem]:
    plan: list[PilotPlanItem] = []
    for config in configs:
        skip_reason = collection_skip_reason(config)
        eligible = skip_reason is None
        reason = (
            "collection_status=allowed, robots_verification=verified; "
            "출처별 최대 2건·전체 최대 10건 적용"
            if eligible
            else skip_reason or "수집 불가"
        )
        urls = tuple(
            ([config.api_url] if config.api_url else list(config.discovery_urls))[
                : config.max_pages
            ]
        )
        plan.append(PilotPlanItem(config.source_id, config.name, urls, eligible, reason))
    return plan


def enforce_candidate_safety(candidate: CollectedCandidate) -> CollectedCandidate:
    updates: dict[str, object] = {
        "review_status": "draft",
    }
    if candidate.copyright_status == "unknown":
        updates["allowed_for_rag"] = False
        updates["allowed_for_training"] = False
    return replace(candidate, **updates)


def run_pilot(
    configs: list[CollectorConfig],
    *,
    query: str,
    raw_dir: Path,
    extracted_dir: Path,
    registry: CandidateRegistry,
    collector_factory: Callable[[CollectorConfig], BaseCollector] = build_collector,
) -> PilotRunResult:
    collected: list[CollectedCandidate] = []
    errors: list[str] = []
    counts: Counter[str] = Counter()

    for config in configs:
        if len(collected) >= MAX_TOTAL_RESULTS:
            break
        skip_reason = collection_skip_reason(config)
        if skip_reason:
            errors.append(f"{config.source_id}: {skip_reason}")
            continue

        safe_config = replace(config, max_results=MAX_RESULTS_PER_SOURCE)
        report = collector_factory(safe_config).collect(
            query, raw_dir=raw_dir, extracted_dir=extracted_dir
        )
        remaining = MAX_TOTAL_RESULTS - len(collected)
        source_candidates = tuple(
            enforce_candidate_safety(candidate)
            for candidate in report.candidates[: min(MAX_RESULTS_PER_SOURCE, remaining)]
        )
        collected.extend(source_candidates)
        counts[config.source_id] += len(source_candidates)
        errors.extend(f"{config.source_id}: {error}" for error in report.errors)

    added = registry.add_new(tuple(collected))
    return PilotRunResult(
        tuple(collected),
        tuple(added),
        tuple(errors),
        dict(counts),
    )
