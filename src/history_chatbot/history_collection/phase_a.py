"""The only authorized bridge from Phase A readiness evidence to batch transport."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from history_chatbot.collectors.public_history_batch import (
    ADAPTERS, BatchCandidate, BatchError, BatchPipeline, BatchTransport, PublicSourceAdapter,
    RequestController, SourceSpec, UrllibBatchTransport, atomic_write,
    canonicalize_public_url, markdown_report, normalize_space, read_jsonl,
    validate_public_url,
)
from history_chatbot.history_collection.models import (
    AcceptanceStatus, CandidateDocument, DuplicateStatus, HardRejectionCode,
    Phase, ReviewStatus, RightsEvidence, SourceTier,
)
from history_chatbot.history_collection.preflight import PHASE_A_USER_AGENT
from history_chatbot.history_collection.pipeline import now_iso
from history_chatbot.history_collection.quality import classify_topics, evaluate_content
from history_chatbot.history_collection.rights import evaluate_rights
from history_chatbot.history_collection.scoring import score_candidate


EXECUTION_ACKNOWLEDGEMENT = "I_APPROVE_PHASE_A_DOCUMENT_FETCH"


def phase_a_candidate_record_builder(*, batch_id: str,
                                     source_plan: Mapping[str, Mapping[str, Any]],
                                     readiness: Mapping[str, Mapping[str, Any]],
                                     candidate_only: bool = False) -> Callable[..., dict[str, Any]]:
    """Normalize a fetched batch result into the fail-closed candidate schema."""
    def build(**values: Any) -> dict[str, Any]:
        candidate = values["candidate"]
        detail = values.get("detail")
        response = values["response"]
        raw_target = values.get("raw_target")
        extracted_target = values.get("extracted_target")
        decision = str(values["decision"])
        collected_at = str(values["collected_at"])
        body_hash = str(values.get("body_hash", ""))
        extracted_hash = str(values.get("extracted_hash", ""))
        plan = source_plan[candidate.source_id]
        state = readiness[candidate.source_id]
        detail_text = detail.text if detail is not None else ""
        detail_metadata = detail.metadata if detail is not None else {}
        media = response.content_type.split(";", 1)[0].strip().lower()
        evidence_rows = list(state.get("evidence", []))
        policy = next((item for item in evidence_rows
                       if item.get("evidence_type") == "policy"), {})
        kogl_type = str(detail_metadata.get("kogl_type", ""))
        rights_evidence = RightsEvidence(
            publisher=candidate.institution,
            rights_holder=candidate.original_institution or candidate.institution,
            policy_url=str(policy.get("url", plan.get("policy_url", ""))),
            document_rights_url=str(detail_metadata.get("document_rights_url", "")),
            license_text=str(detail_metadata.get("rights_evidence_text", "")),
            kogl_type=kogl_type,
            checked_at=str(policy.get("checked_at", collected_at)),
            human_review_required=True,
        )
        duplicate = decision == "rejected_duplicate"
        duplicate_of = str(candidate.discovery_metadata.get("baseline_duplicate_of", "")) if duplicate else ""
        duplicate_reason = str(values.get("reasons", [""])[0]) if values.get("reasons") else ""
        extraction_ok = detail is not None and bool(normalize_space(detail_text))
        document = CandidateDocument(
            candidate_id=candidate.document_id,
            batch_id=batch_id,
            phase=Phase.A,
            source_id=candidate.source_id,
            source_url=candidate.source_url,
            canonical_url=str(detail_metadata.get("document_canonical_url", candidate.canonical_url)),
            source_title=str(detail_metadata.get("page_title", candidate.title)),
            publisher=candidate.institution,
            institution=candidate.original_institution or candidate.institution,
            publisher_family=str(plan["publisher_family"]),
            source_tier=SourceTier(str(plan["source_tier"])),
            document_type=candidate.document_type,
            topic_categories=classify_topics(candidate.title, detail_text),
            historical_period=candidate.published_date,
            location=list(candidate.place_tags),
            discovered_at=str(candidate.discovery_metadata.get("discovered_at", collected_at)),
            fetched_at=collected_at,
            raw_path=raw_target.as_posix() if raw_target is not None else "",
            raw_sha256=hashlib.sha256(response.body).hexdigest() if raw_target is not None else "",
            extracted_path=extracted_target.as_posix() if extracted_target is not None else "",
            extracted_sha256=extracted_hash,
            normalized_body_sha256=body_hash,
            response_final_url=response.final_url,
            response_http_status=response.status,
            response_content_type=media,
            extraction_status="success" if extraction_ok else "failed",
            extraction_method=media or "unknown",
            language="ko",
            robots_status=str(state.get("robots_status", "unknown")),
            access_status=str(state.get("public_access_status", "unknown")),
            rights_status="unknown",
            license="",
            kogl_type=kogl_type,
            rights_evidence=rights_evidence,
            provenance={
                "record_id": candidate.document_id,
                "source_id": candidate.source_id,
                "discovered_from": candidate.discovery_metadata.get("discovery_request_url", ""),
                "discovery_request_url": candidate.discovery_metadata.get("discovery_request_url", ""),
                "discovery_response_final_url": candidate.discovery_metadata.get(
                    "discovery_response_final_url", ""
                ),
                "discovery_query": candidate.discovery_metadata.get("discovery_query", ""),
                "detail_requested_url": candidate.source_url,
                "detail_final_url": response.final_url,
                "response_http_status": response.status,
                "response_content_type": media,
                "portal_name": candidate.portal_name,
                "original_institution": candidate.original_institution,
                "batch_decision": decision,
                "policy_status": state.get("policy_status", "unknown"),
                "rights_metadata_status": state.get("rights_metadata_status", "unknown"),
                "new_unique_increment": (
                    int(not duplicate) if candidate_only else
                    int(
                        decision in {"accepted_hackathon", "accepted_metadata_only", "needs_review"}
                        and not duplicate
                    )
                ),
            },
            duplicate_status=DuplicateStatus.CONFIRMED if duplicate else DuplicateStatus.UNIQUE,
            duplicate_group=("duplicate-" + (duplicate_of or candidate.document_id)) if duplicate else "",
            duplicate_of=duplicate_of,
            duplicate_method=duplicate_reason if duplicate else "",
            uniqueness_score=0 if duplicate else 10,
            body_text=detail_text,
            publication_metadata={
                "published_date": candidate.published_date,
                "parent_document_id": candidate.parent_document_id,
            },
        )
        quality = evaluate_content(
            document.source_title, detail_text,
            extraction_status=document.extraction_status, language=document.language,
        )
        rights = evaluate_rights(
            document.robots_status, document.access_status, document.rights_status,
            document.rights_evidence, policy_status=str(state.get("policy_status", "unknown")),
        )
        score_candidate(document, quality, rights)
        rejection_map = {
            "rejected_duplicate": HardRejectionCode.DUPLICATE_EXACT,
            "rejected_irrelevant": HardRejectionCode.NOT_MOKPO_RELEVANT,
            "rejected_empty": HardRejectionCode.EXTRACTION_FAILED,
            "rejected_quality": HardRejectionCode.EXTRACTION_FAILED
                if not extraction_ok else HardRejectionCode.NO_HISTORICAL_BODY,
            "rejected_access_policy": HardRejectionCode.POLICY_BLOCKED,
        }
        extra_rejection = rejection_map.get(decision)
        if extra_rejection and extra_rejection not in document.rejection_reasons:
            document.rejection_reasons.append(extra_rejection)
        if decision not in {"accepted_hackathon", "accepted_metadata_only", "needs_review"}:
            document.review_status = ReviewStatus.AUTO_REJECTED
            document.acceptance_status = AcceptanceStatus.REJECTED
        return document.to_dict()

    return build


def _readiness_by_source(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if report.get("phase") != "A" or report.get("status") != "PASS":
        raise BatchError("Phase A preflight PASS report is required")
    if int(report.get("collection_network_requests", -1)) != 0:
        raise BatchError("preflight report must have collection_network_requests=0")
    return {str(item["source_id"]): item for item in report.get("sources", [])}


def build_verified_adapters(plan: Sequence[dict[str, Any]], report: Mapping[str, Any],
                            environment: Mapping[str, str]) -> dict[str, PublicSourceAdapter]:
    readiness = _readiness_by_source(report)
    adapters: dict[str, PublicSourceAdapter] = {}
    for item in plan:
        if int(item.get("unique_target", 0)) <= 0:
            continue
        source_id = str(item["source_id"])
        state = readiness.get(source_id)
        if not state or state.get("collection_ready") is not True:
            raise BatchError("source is not verified for Phase A: " + source_id)
        if state.get("robots_status") not in {"allowed", "verified_allowed"}:
            raise BatchError("robots status is not allowed: " + source_id)
        if state.get("policy_status") != "allowed":
            raise BatchError("policy status is not allowed: " + source_id)
        if state.get("public_access_status") != "public":
            raise BatchError("public access is not verified: " + source_id)
        if state.get("rights_metadata_status") not in {"verified", "document_level_required"}:
            raise BatchError("rights metadata is not verified: " + source_id)
        required_delay = float(item.get("minimum_delay_seconds", 1.5))
        reported_delay = float(state.get("crawl_delay_seconds", 0) or 0)
        if reported_delay and reported_delay > required_delay:
            raise BatchError("configured delay is lower than robots Crawl-Delay: " + source_id)
        original = ADAPTERS[source_id]
        spec = original.spec
        if spec.api_key_environment and not environment.get(spec.api_key_environment, "").strip():
            raise BatchError("API_KEY_MISSING:" + spec.api_key_environment)
        if (spec.api_key_environment and spec.api_key_format_environment and
                environment.get(spec.api_key_format_environment, "").strip().lower() not in {"encoding", "decoding"}):
            raise BatchError("API_KEY_FORMAT_MISSING:" + spec.api_key_format_environment)
        if state.get("endpoint_status") not in {"verified", "not_applicable"}:
            raise BatchError("endpoint is not verified: " + source_id)
        verified_spec = replace(spec, robots_status="verified_allowed", policy_status="allowed",
                                endpoint_verification_status="verified", production_enabled=True)
        adapters[source_id] = type(original)(verified_spec)
    return adapters


def _candidate_readiness_by_source(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if report.get("phase") != "A":
        raise BatchError("Phase A candidate readiness report is required")
    if int(report.get("collection_network_requests", -1)) != 0:
        raise BatchError("candidate readiness report must have collection_network_requests=0")
    return {str(item["source_id"]): item for item in report.get("sources", [])}


def build_candidate_ready_adapters(
    plan: Sequence[dict[str, Any]], report: Mapping[str, Any],
    environment: Mapping[str, str], source_ids: Sequence[str],
) -> dict[str, PublicSourceAdapter]:
    """Build opt-in candidate-only adapters without weakening verified collection."""
    readiness = _candidate_readiness_by_source(report)
    plans = {str(item["source_id"]): item for item in plan}
    adapters: dict[str, PublicSourceAdapter] = {}
    for source_id in source_ids:
        item = plans.get(source_id)
        state = readiness.get(source_id)
        if item is None or state is None:
            raise BatchError("candidate source readiness is missing: " + source_id)
        if SourceTier(str(item["source_tier"])) == SourceTier.TIER_4:
            raise BatchError("Tier 4 source is discovery-only: " + source_id)
        if state.get("candidate_collection_ready") is not True:
            raise BatchError("source is not candidate-ready: " + source_id)
        if state.get("verified_collection_ready") is True:
            raise BatchError("candidate-only mode requires verified promotion to remain disabled")
        if state.get("robots_status") not in {"allowed", "verified_allowed"}:
            raise BatchError("robots status is not allowed: " + source_id)
        if state.get("public_access_status") != "public":
            raise BatchError("public access is not verified: " + source_id)
        if state.get("live_extraction_status") not in {"success", "verified"}:
            raise BatchError("live extraction is not verified: " + source_id)
        blockers = {str(value).upper() for value in state.get("blockers", [])}
        if blockers & {
            "LOGIN", "LOGIN_REQUIRED", "CAPTCHA", "PAYWALL", "ACCESS_BARRIER",
            "ROBOTS_BLOCKED", "REDIRECT_OUTSIDE_ALLOWED_HOSTS",
        }:
            raise BatchError("candidate source has an access blocker: " + source_id)
        allowed_hosts = tuple(str(value) for value in item.get("allowed_hosts", []) if str(value))
        if not allowed_hosts:
            raise BatchError("candidate source has no allowed hosts: " + source_id)
        original = ADAPTERS[source_id]
        if original.spec.api_key_environment:
            raise BatchError("candidate-only exact HTML pilot does not accept API sources")
        policy_status = str(state.get("policy_status", "unknown"))
        if policy_status not in {
            "unknown", "needs_human_review", "document_level_required", "allowed"
        }:
            raise BatchError("candidate source policy is explicitly unusable: " + source_id)
        candidate_spec = replace(
            original.spec,
            allowed_hosts=allowed_hosts,
            robots_status="verified_allowed",
            policy_status=policy_status,
            endpoint_verification_status="verified",
            production_enabled=True,
            candidate_only=True,
        )
        adapters[source_id] = type(original)(candidate_spec)
    return adapters


def _assert_candidate_lane(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise BatchError("Phase A outputs must remain under data/history_candidates") from exc
    if "provisional_hackathon" in path.as_posix():
        raise BatchError("provisional_hackathon writes are forbidden")


class PhaseAExecutor:
    def __init__(self, config: Mapping[str, Any], preflight_report: Mapping[str, Any],
                 *, environment: Mapping[str, str] | None = None,
                 transport_factory: Callable[[Sequence[str]], BatchTransport] | None = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.config = config
        self.environment = environment or os.environ
        self.readiness = _readiness_by_source(preflight_report)
        self.source_plan = {
            str(item["source_id"]): item for item in config["phase_a_source_plan"]
        }
        self.adapters = build_verified_adapters(config["phase_a_source_plan"], preflight_report,
                                                self.environment)
        budget = config["phase_a_request_budget"]
        stage_limits = {
            item["source_id"]: {"discovery": int(item["discovery_request_budget"]),
                                "detail": int(item["detail_request_budget"])}
            for item in config["phase_a_source_plan"] if int(item.get("unique_target", 0)) > 0
        }
        source_delays = {
            item["source_id"]: float(item.get("minimum_delay_seconds", budget["delay_seconds"]))
            for item in config["phase_a_source_plan"] if int(item.get("unique_target", 0)) > 0
        }
        kwargs = {}
        if sleep is not None:
            kwargs["sleep"] = sleep
        self.controller = RequestController(
            int(budget["collection_maximum_requests"]), float(budget["delay_seconds"]),
            transport_factory or (lambda hosts: UrllibBatchTransport(hosts, PHASE_A_USER_AGENT)),
            require_source_preflight=True, source_stage_limits=stage_limits,
            source_delay_seconds=source_delays, **kwargs,
        )
        self.pipeline = BatchPipeline(self.adapters, self.controller, phase_a_authorized=True)

    def discover(self, *, acknowledgement: str, batch_id: str, keywords: Sequence[str],
                 output_root: Path, timeout: float, max_bytes: int) -> dict[str, Any]:
        if acknowledgement != EXECUTION_ACKNOWLEDGEMENT:
            raise BatchError("explicit Phase A execution acknowledgement is required")
        candidate_root = output_root / "history_candidates"
        catalog = candidate_root / "manifests" / (batch_id + ".catalog.jsonl")
        report_json = candidate_root / "reports" / "phase_a" / (batch_id + "-discovery.json")
        report_md = candidate_root / "reports" / "phase_a" / (batch_id + "-discovery.md")
        manifest = candidate_root / "manifests" / "candidates.jsonl"
        extracted = candidate_root / "extracted"
        for path in (catalog, report_json, report_md, manifest, extracted):
            _assert_candidate_lane(path, candidate_root)
        limits = {
            "max_accepted": 50, "max_per_source": 20,
            "max_requests": int(self.config["phase_a_request_budget"]["collection_maximum_requests"]),
            "delay_seconds": float(self.config["phase_a_request_budget"]["delay_seconds"]),
            "per_source_limits": {item["source_id"]: int(item["unique_target"])
                                  for item in self.config["phase_a_source_plan"]
                                  if int(item.get("unique_target", 0)) > 0},
        }
        result = self.pipeline.discover(batch_id, list(self.adapters), keywords, catalog,
                                        report_json, report_md, self.environment,
                                        timeout, max_bytes, limits)
        result["preflight_forced"] = self.controller.require_source_preflight
        result["collection_network_requests"] = self.controller.request_count
        result["candidate_lane"] = candidate_root.as_posix()
        return result

    def collect(self, *, acknowledgement: str, batch_id: str, keywords: Sequence[str],
                output_root: Path, timeout: float, max_bytes: int) -> dict[str, Any]:
        discovery = self.discover(acknowledgement=acknowledgement, batch_id=batch_id,
                                  keywords=keywords, output_root=output_root,
                                  timeout=timeout, max_bytes=max_bytes)
        candidate_root = output_root / "history_candidates"
        catalog = candidate_root / "manifests" / (batch_id + ".catalog.jsonl")
        manifest = candidate_root / "manifests" / "candidates.jsonl"
        raw = candidate_root / "raw"
        extracted = candidate_root / "extracted"
        report_json = candidate_root / "reports" / "phase_a" / (batch_id + "-collection.json")
        report_md = candidate_root / "reports" / "phase_a" / (batch_id + "-collection.md")
        for path in (catalog, manifest, raw, extracted, report_json, report_md):
            _assert_candidate_lane(path, candidate_root)
        plan = [item for item in self.config["phase_a_source_plan"] if int(item.get("unique_target", 0)) > 0]
        limits = {
            "max_accepted": 50, "max_per_source": 20,
            "max_requests": int(self.config["phase_a_request_budget"]["collection_maximum_requests"]),
            "delay_seconds": float(self.config["phase_a_request_budget"]["delay_seconds"]),
            "per_source_limits": {item["source_id"]: int(item["unique_target"]) for item in plan},
        }
        collection = self.pipeline.execute(
            batch_id, list(self.adapters), catalog, manifest, extracted,
            report_json, report_md, timeout, max_bytes, limits,
            collected_at=now_iso(), environment=self.environment,
            raw_dir=raw,
            record_builder=phase_a_candidate_record_builder(
                batch_id=batch_id, source_plan=self.source_plan, readiness=self.readiness,
            ),
        )
        return {
            "mode": "phase-a-collection", "discovery": discovery,
            "collection": collection, "preflight_forced": True,
            "collection_network_requests": self.controller.request_count,
            "candidate_lane": candidate_root.as_posix(),
        }


class CandidateOnlyExecutor:
    """Bounded exact-URL writer for candidate evidence; never promotes verified data."""

    def __init__(
        self, config: Mapping[str, Any], readiness_report: Mapping[str, Any],
        *, source_ids: Sequence[str], maximum_total_requests: int,
        environment: Mapping[str, str] | None = None,
        transport_factory: Callable[[Sequence[str]], BatchTransport] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise BatchError("candidate-only mode requires explicit unique sources")
        if maximum_total_requests < 1 or maximum_total_requests > 10:
            raise BatchError("candidate-only request ceiling must be between 1 and 10")
        self.config = config
        self.environment = environment or os.environ
        self.source_ids = list(source_ids)
        self.readiness = _candidate_readiness_by_source(readiness_report)
        self.source_plan = {
            str(item["source_id"]): item for item in config["phase_a_source_plan"]
        }
        self.adapters = build_candidate_ready_adapters(
            config["phase_a_source_plan"], readiness_report, self.environment, self.source_ids,
        )
        budget = config["phase_a_request_budget"]
        global_delay = float(budget["delay_seconds"])
        source_delays = {
            source_id: float(self.source_plan[source_id].get("minimum_delay_seconds", global_delay))
            for source_id in self.source_ids
        }
        kwargs: dict[str, Any] = {}
        if sleep is not None:
            kwargs["sleep"] = sleep
        self.controller = RequestController(
            maximum_total_requests, global_delay,
            transport_factory or (lambda hosts: UrllibBatchTransport(hosts, PHASE_A_USER_AGENT)),
            require_source_preflight=True,
            source_stage_limits={
                source_id: {"detail": maximum_total_requests} for source_id in self.source_ids
            },
            source_delay_seconds=source_delays,
            **kwargs,
        )
        self.pipeline = BatchPipeline(self.adapters, self.controller, phase_a_authorized=True)

    def collect_exact(
        self, *, acknowledgement: str, batch_id: str, exact_seed_catalog: Path,
        baseline_manifest: Path, output_root: Path, max_documents: int,
        timeout: float, max_bytes: int,
    ) -> dict[str, Any]:
        if acknowledgement != EXECUTION_ACKNOWLEDGEMENT:
            raise BatchError("explicit candidate-only execution acknowledgement is required")
        if max_documents < 1 or max_documents > 10:
            raise BatchError("max-documents must be between 1 and 10")
        if self.controller.max_requests != max_documents:
            raise BatchError("exact candidate request ceiling must equal max-documents")
        seed_rows = read_jsonl(exact_seed_catalog)
        if len(seed_rows) != max_documents:
            raise BatchError("exact seed count must equal max-documents")
        baseline_rows = read_jsonl(baseline_manifest)
        baseline_by_url: dict[str, str] = {}
        for row in baseline_rows:
            retained_id = str(
                row.get("document_id") or row.get("candidate_id") or row.get("source_id") or ""
            )
            for name in ("canonical_url", "source_url"):
                value = str(row.get(name, ""))
                if value and retained_id:
                    baseline_by_url[canonicalize_public_url(value)] = retained_id

        required = {
            "source_id", "document_id", "title", "source_url", "canonical_url",
            "institution", "publisher_family",
        }
        candidates = []
        for row in seed_rows:
            missing = sorted(required - {key for key, value in row.items() if str(value).strip()})
            if missing:
                raise BatchError("exact seed is missing fields: " + ",".join(missing))
            source_id = str(row["source_id"])
            if source_id not in self.source_ids:
                raise BatchError("exact seed source was not explicitly selected: " + source_id)
            if str(row["publisher_family"]) != str(self.source_plan[source_id]["publisher_family"]):
                raise BatchError("exact seed publisher_family does not match source plan")
            adapter = self.adapters[source_id]
            source_url = canonicalize_public_url(str(row["source_url"]))
            canonical_url = canonicalize_public_url(str(row["canonical_url"]))
            validate_public_url(source_url, adapter.spec.allowed_hosts)
            validate_public_url(canonical_url, adapter.spec.allowed_hosts)
            retained_id = baseline_by_url.get(canonical_url, "")
            declared_duplicate = str(row.get("duplicate_of", ""))
            if declared_duplicate and declared_duplicate != retained_id:
                raise BatchError("exact seed duplicate_of does not match baseline canonical record")
            candidate = BatchCandidate.from_dict({
                **row,
                "source_url": source_url,
                "canonical_url": canonical_url,
                "discovery_metadata": {
                    "discovery_request_url": "exact-seed:" + exact_seed_catalog.as_posix(),
                    "discovery_response_final_url": "",
                    "discovery_query": "",
                    "baseline_duplicate_of": retained_id,
                },
            })
            adapter.detail_url(candidate, self.environment)
            candidates.append(candidate)

        candidate_root = output_root / "history_candidates"
        catalog = candidate_root / "manifests" / (batch_id + ".exact-seed.jsonl")
        manifest = candidate_root / "manifests" / "candidates.jsonl"
        raw = candidate_root / "raw"
        extracted = candidate_root / "extracted"
        report_json = candidate_root / "reports" / "candidate-only" / (batch_id + ".json")
        report_md = candidate_root / "reports" / "candidate-only" / (batch_id + ".md")
        for path in (catalog, manifest, raw, extracted, report_json, report_md):
            _assert_candidate_lane(path, candidate_root)
        if catalog.exists() or report_json.exists() or report_md.exists():
            raise BatchError("candidate-only batch_id already exists")
        existing_candidate_rows = read_jsonl(manifest)
        existing_ids = {
            str(row.get("candidate_id") or row.get("document_id") or "")
            for row in existing_candidate_rows
        }
        if any(candidate.document_id in existing_ids for candidate in candidates):
            raise BatchError("exact seed candidate_id already exists")

        # The normalized exact-seed catalog is input provenance. It can remain after a
        # transport failure, but it is never counted as a created candidate document.
        catalog_bytes = b"".join(
            (json.dumps(asdict(candidate), ensure_ascii=False) + "\n").encode("utf-8")
            for candidate in candidates
        )
        atomic_write({catalog: catalog_bytes})
        limits = {
            "max_accepted": max_documents,
            "max_per_source": max_documents,
            "max_requests": max_documents,
            "delay_seconds": float(self.config["phase_a_request_budget"]["delay_seconds"]),
            "per_source_limits": {source_id: max_documents for source_id in self.source_ids},
        }
        collection = self.pipeline.execute(
            batch_id, self.source_ids, catalog, manifest, extracted,
            report_json, report_md, timeout, max_bytes, limits,
            collected_at=now_iso(), environment=self.environment, raw_dir=raw,
            record_builder=phase_a_candidate_record_builder(
                batch_id=batch_id, source_plan=self.source_plan, readiness=self.readiness,
                candidate_only=True,
            ),
            external_duplicate_records=baseline_rows,
        )
        batch_records = [
            row for row in read_jsonl(manifest) if str(row.get("batch_id", "")) == batch_id
        ]
        new_unique_increment = sum(
            int(row.get("provenance", {}).get("new_unique_increment", 0))
            for row in batch_records
        )
        collection["counts"]["accepted_unique"] = collection["counts"]["stored"]
        collection["counts"]["stored"] = len(batch_records)
        collection["counts"]["actual_candidates_created"] = len(batch_records)
        collection["counts"]["new_unique_increment"] = new_unique_increment
        atomic_write({
            report_json: (
                json.dumps(collection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            report_md: markdown_report(collection),
        })
        return {
            "mode": "candidate-only-exact-seed",
            "candidate_only": True,
            "verified_collection_ready": False,
            "collection": collection,
            "collection_network_requests": self.controller.request_count,
            "candidate_lane": candidate_root.as_posix(),
            "baseline_manifest": baseline_manifest.as_posix(),
            "stored": len(batch_records),
            "actual_candidates_created": len(batch_records),
            "new_unique_increment": new_unique_increment,
        }
