"""Offline-safe orchestration and append-only collection audit records."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from history_chatbot.history_collection.dedup import DuplicateIndex
from history_chatbot.history_collection.models import CandidateDocument, Phase
from history_chatbot.history_collection.quality import ContentQuality, classify_topics, evaluate_content
from history_chatbot.history_collection.rights import evaluate_rights
from history_chatbot.history_collection.scoring import score_candidate


class NetworkDisabledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    candidate_id: str
    batch_id: str
    phase: str
    occurred_at: str
    action: str
    previous_status: str
    new_status: str
    reasons: tuple[str, ...]


class AppendOnlyJsonl:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(str(self.path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class DecisionAuditLog(AppendOnlyJsonl):
    def append_event(self, event: DecisionEvent) -> None:
        self.append(asdict(event))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".history-collection-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class CandidateProcessor:
    def __init__(self, *, policy_status: dict[str, str] | None = None) -> None:
        self.dedup = DuplicateIndex()
        self.policy_status = policy_status or {}

    def process(self, candidate: CandidateDocument) -> tuple[CandidateDocument, ContentQuality]:
        if not candidate.topic_categories:
            candidate.topic_categories = classify_topics(candidate.source_title, candidate.body_text)
        duplicate = self.dedup.add(candidate)
        quality = evaluate_content(candidate.source_title, candidate.body_text,
                                   extraction_status=candidate.extraction_status,
                                   language=candidate.language)
        rights = evaluate_rights(candidate.robots_status, candidate.access_status,
                                 candidate.rights_status, candidate.rights_evidence,
                                 policy_status=self.policy_status.get(candidate.publisher_family, "unknown"))
        score_candidate(candidate, quality, rights)
        if duplicate.status.value == "suspected" and candidate.review_status.value == "auto_candidate":
            from history_chatbot.history_collection.models import ReviewStatus
            candidate.review_status = ReviewStatus.NEEDS_HUMAN_REVIEW
        return candidate, quality


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dry_run(phase: Phase, config: dict[str, Any], *, batch_id: str) -> dict[str, Any]:
    target = int(config["phase_targets"][phase.value])
    plan = config["phase_a_source_plan"] if phase == Phase.A else []
    return {
        "mode": "dry-run", "network": False, "network_requests": 0,
        "files_created": 0, "batch_id": batch_id, "phase": phase.value,
        "target_unique": target, "source_plan": plan,
        "source_plan_total": sum(int(item.get("unique_target", 0)) for item in plan),
        "historical_unique_target": sum(int(item.get("unique_target", 0)) for item in plan),
        "metadata_discovery_quota": sum(int(item.get("discovery_metadata_quota", 0)) for item in plan),
        "priority_topics": config.get("priority_topics", []),
        "deprioritized_topics": config.get("deprioritized_topics", []),
        "baseline_protected": "data/provisional_hackathon",
        "next_action": "explicit --discover --allow-network --execute-approved-phase-a with PASS preflight report",
    }


def network_preflight(plan: Iterable[dict[str, Any]]) -> list[str]:
    blockers = []
    for item in plan:
        if item.get("readiness") not in {"ready", "metadata_discovery_only"}:
            blockers.append("%s:%s" % (item.get("source_id", "unknown"), item.get("readiness", "unknown")))
    return blockers


def historical_unique_target(plan: Iterable[dict[str, Any]]) -> int:
    return sum(int(item.get("unique_target", 0)) for item in plan)


def metadata_discovery_quota(plan: Iterable[dict[str, Any]]) -> int:
    return sum(int(item.get("discovery_metadata_quota", 0)) for item in plan)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
