"""Bounded source-readiness probes kept separate from document collection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from history_chatbot.collectors.public_history_batch import (
    BatchResponse, BatchTransport, GlobalSafetyError, UrllibBatchTransport,
    validate_public_url,
)


PHASE_A_USER_AGENT = "MokpoHistoryRAGCollector/1.0 (+bounded Phase A)"


@dataclass(frozen=True, slots=True)
class PreflightEvidence:
    source_id: str
    evidence_type: str
    url: str
    checked_at: str
    status: str
    http_status: int | None = None
    matched_rule: str = ""
    target_path: str = ""
    evidence: str = ""
    notes: str = ""


@dataclass(slots=True)
class SourceReadiness:
    source_id: str
    robots_status: str = "unknown"
    policy_status: str = "unknown"
    endpoint_status: str = "not_applicable"
    api_key_status: str = "not_applicable"
    rights_metadata_status: str = "unknown"
    public_access_status: str = "unknown"
    crawl_delay_seconds: float = 0.0
    collection_ready: bool = False
    source_role: str = ""
    human_review_required: bool = True
    blockers: list[str] = field(default_factory=list)
    evidence: list[PreflightEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    phase: str
    checked_at: str
    sources: tuple[SourceReadiness, ...]
    preflight_network_requests: int
    collection_network_requests: int = 0
    candidates_created: int = 0
    raw_history_documents_created: int = 0

    @property
    def status(self) -> str:
        historical = [item for item in self.sources if item.source_role != "metadata_discovery_only"]
        return "PASS" if historical and all(item.collection_ready for item in historical) else "STOP"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status
        return value


class PreflightController:
    """Exact-URL GET probes with a distinct request counter and no retry."""

    def __init__(self, max_requests: int, transport_factory: Callable[[Sequence[str]], BatchTransport],
                 *, timeout: float = 15.0, max_bytes: int = 262144) -> None:
        if not 1 <= max_requests <= 25:
            raise ValueError("preflight max_requests must be between 1 and 25")
        self.max_requests = max_requests
        self.transport_factory = transport_factory
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.preflight_network_requests = 0
        self.collection_network_requests = 0

    def get(self, url: str, allowed_hosts: Sequence[str]) -> BatchResponse:
        validate_public_url(url, allowed_hosts)
        if self.preflight_network_requests >= self.max_requests:
            raise GlobalSafetyError("preflight_request_budget_exceeded")
        self.preflight_network_requests += 1
        response = self.transport_factory(allowed_hosts).get(url, self.timeout, self.max_bytes)
        validate_public_url(response.final_url, allowed_hosts)
        return response


class PhaseAPreflight:
    def __init__(self, controller: PreflightController, environment: Mapping[str, str] | None = None,
                 *, now: Callable[[], datetime] | None = None) -> None:
        self.controller = controller
        self.environment = environment or os.environ
        self.now = now or (lambda: datetime.now(timezone.utc))

    def run(self, plan: Sequence[dict[str, Any]], source_ids: Sequence[str]) -> PreflightReport:
        requested = set(source_ids)
        selected = [item for item in plan if item["source_id"] in requested]
        unknown = requested - {item["source_id"] for item in selected}
        if unknown:
            raise ValueError("unknown preflight sources: " + ", ".join(sorted(unknown)))
        results = tuple(self._source(item) for item in selected)
        return PreflightReport("A", self.now().isoformat(), results,
                               self.controller.preflight_network_requests,
                               self.controller.collection_network_requests)

    def _source(self, item: dict[str, Any]) -> SourceReadiness:
        result = SourceReadiness(item["source_id"], source_role=item.get("source_role", ""))
        hosts = tuple(item.get("allowed_hosts", ()))
        checked_at = self.now().isoformat()
        robots_url = item.get("robots_url", "")
        if robots_url:
            try:
                response = self.controller.get(robots_url, hosts)
                text = response.body.decode("utf-8", errors="replace")
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(text.splitlines())
                target = item.get("robots_target_url", "")
                allowed = bool(target and parser.can_fetch(PHASE_A_USER_AGENT, target))
                user_agent_delay = parser.crawl_delay(PHASE_A_USER_AGENT)
                wildcard_delay = parser.crawl_delay("*")
                selected_delay = (
                    user_agent_delay
                    if user_agent_delay is not None
                    else wildcard_delay
                )
                result.crawl_delay_seconds = float(
                    selected_delay if selected_delay is not None else 0
                )
                result.robots_status = "verified_allowed" if allowed else "blocked"
                result.evidence.append(PreflightEvidence(item["source_id"], "robots", robots_url,
                    checked_at, result.robots_status, response.status,
                    "RobotFileParser.can_fetch; Crawl-Delay=%s" % result.crawl_delay_seconds,
                    urlsplit(target).path, text[:1000],
                    "robots 허용은 저작권 허용을 의미하지 않음"))
                configured_delay = float(item.get("minimum_delay_seconds", 0) or 0)
                if result.crawl_delay_seconds > configured_delay:
                    result.blockers.append("CONFIGURED_DELAY_BELOW_ROBOTS")
            except Exception as exc:
                result.robots_status = "unknown"
                result.evidence.append(PreflightEvidence(item["source_id"], "robots", robots_url,
                    checked_at, "unknown", notes=(type(exc).__name__ + ":" + str(exc))[:300]))
        elif result.source_role != "metadata_discovery_only":
            result.blockers.append("ROBOTS_EVIDENCE_MISSING")

        for kind, field_name in (("policy", "policy_url"), ("api_documentation", "api_docs_url")):
            url = item.get(field_name, "")
            if not url:
                continue
            if kind == "api_documentation":
                result.endpoint_status = "unknown"
            try:
                response = self.controller.get(url, hosts)
                status = "fetched_needs_human_review" if 200 <= response.status < 300 else "unknown"
                result.evidence.append(PreflightEvidence(item["source_id"], kind, url, checked_at,
                                                          status, response.status,
                                                          evidence=response.body[:1000].decode("utf-8", errors="replace")))
                if kind == "policy":
                    result.policy_status = "needs_human_review" if status.startswith("fetched") else "unknown"
                else:
                    result.endpoint_status = "documented_needs_verification" if status.startswith("fetched") else "unknown"
            except Exception as exc:
                result.evidence.append(PreflightEvidence(item["source_id"], kind, url, checked_at,
                                                          "unknown", notes=(type(exc).__name__ + ":" + str(exc))[:300]))

        key_name = item.get("api_key_environment", "")
        if key_name:
            result.api_key_status = "present" if self.environment.get(key_name, "").strip() else "KEY_MISSING"
            if result.api_key_status == "KEY_MISSING":
                result.blockers.append("API_KEY_MISSING:" + key_name)
        endpoint_name = item.get("endpoint_environment", "")
        if endpoint_name:
            result.endpoint_status = "configured_unverified" if self.environment.get(endpoint_name, "").strip() else "ENDPOINT_MISSING"
            if result.endpoint_status == "ENDPOINT_MISSING":
                result.blockers.append("ENDPOINT_MISSING:" + endpoint_name)

        if result.source_role == "metadata_discovery_only":
            result.collection_ready = False
            result.human_review_required = True
            result.blockers.append("METADATA_DISCOVERY_ONLY")
            return result
        if result.robots_status not in {"allowed", "verified_allowed"}:
            result.blockers.append("ROBOTS_NOT_ALLOWED")
        if result.policy_status != "allowed":
            result.blockers.append("POLICY_NOT_ALLOWED")
        if result.endpoint_status in {"unknown", "unverified", "documented_needs_verification", "configured_unverified", "ENDPOINT_MISSING"}:
            if item.get("api_docs_url") or item.get("endpoint_environment"):
                result.blockers.append("ENDPOINT_NOT_VERIFIED")
        result.blockers = list(dict.fromkeys(result.blockers))
        result.collection_ready = not result.blockers
        result.human_review_required = not result.collection_ready
        result.public_access_status = "public" if result.collection_ready else "unknown"
        return result


def default_preflight_controller(max_requests: int, timeout: float, max_bytes: int) -> PreflightController:
    return PreflightController(max_requests, lambda hosts: UrllibBatchTransport(hosts, PHASE_A_USER_AGENT),
                               timeout=timeout, max_bytes=max_bytes)


def load_preflight_report(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if value.get("collection_network_requests") != 0:
        raise ValueError("preflight report contains collection requests")
    return value
