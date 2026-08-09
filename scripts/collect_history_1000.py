"""Explicit progressive collection entry point; defaults to no-network dry-run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from history_chatbot.history_collection.models import Phase
from history_chatbot.history_collection.phase_a import (
    EXECUTION_ACKNOWLEDGEMENT, CandidateOnlyExecutor, PhaseAExecutor,
)
from history_chatbot.history_collection.pipeline import build_dry_run, load_config
from history_chatbot.history_collection.preflight import PhaseAPreflight, default_preflight_controller


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "history_collection.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phase", choices=[item.value for item in Phase], required=True)
    value.add_argument("--batch-id", default="offline-preview")
    value.add_argument("--target-unique", type=int)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--discover", action="store_true")
    mode.add_argument("--candidate-only", action="store_true")
    value.add_argument("--allow-network", action="store_true")
    value.add_argument("--source", action="append")
    value.add_argument("--preflight-report", type=Path)
    value.add_argument("--maximum-total-requests", type=int)
    value.add_argument("--execute-approved-phase-a", action="store_true")
    value.add_argument("--execute-approved-candidate-pilot", action="store_true")
    value.add_argument("--exact-seed-catalog", type=Path)
    value.add_argument("--baseline-manifest", type=Path,
                       default=ROOT / "data/provisional_hackathon/manifests/sources.jsonl")
    value.add_argument("--max-documents", type=int)
    value.add_argument("--output-root", type=Path, default=ROOT / "data")
    return value


def _require_network_acknowledgements(args, command: str) -> None:
    if not args.allow_network:
        parser().error(command + " requires explicit --allow-network")
    if args.maximum_total_requests is None:
        parser().error(command + " requires explicit --maximum-total-requests")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    phase = Phase(args.phase)
    config = load_config(args.config)
    configured_target = int(config["phase_targets"][phase.value])
    if args.target_unique is not None and args.target_unique != configured_target:
        parser().error("--target-unique must equal the fixed cumulative phase target")
    if not args.preflight and not args.discover and not args.candidate_only:
        result = build_dry_run(phase, config, batch_id=args.batch_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if phase != Phase.A:
        parser().error("Phase B/C/D require a prior PASS checkpoint and separate approval")
    budget = config["phase_a_request_budget"]
    if args.preflight:
        _require_network_acknowledgements(args, "--preflight")
        if not args.source:
            parser().error("--preflight requires at least one explicit --source")
        maximum = int(args.maximum_total_requests)
        if maximum > int(budget["preflight_maximum_requests"]):
            parser().error("preflight request budget exceeds configured ceiling")
        controller = default_preflight_controller(maximum, float(budget["timeout_seconds"]),
                                                  min(int(budget["max_response_bytes"]), 262144))
        report = PhaseAPreflight(controller, os.environ).run(config["phase_a_source_plan"], args.source)
        print(json.dumps(report.to_dict(), ensure_ascii=True, indent=2))
        return 0 if report.status == "PASS" else 2

    if args.candidate_only:
        _require_network_acknowledgements(args, "--candidate-only")
        if not args.execute_approved_candidate_pilot:
            parser().error("--candidate-only requires --execute-approved-candidate-pilot")
        if not args.source or len(args.source) != 1:
            parser().error("--candidate-only requires exactly one explicit --source")
        if args.preflight_report is None:
            parser().error("--candidate-only requires --preflight-report")
        if args.exact_seed_catalog is None:
            parser().error("--candidate-only requires --exact-seed-catalog")
        if args.max_documents is None:
            parser().error("--candidate-only requires --max-documents")
        if int(args.maximum_total_requests) != int(args.max_documents):
            parser().error("candidate-only request ceiling must equal --max-documents")
        readiness_report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        executor = CandidateOnlyExecutor(
            config, readiness_report, source_ids=args.source,
            maximum_total_requests=int(args.maximum_total_requests), environment=os.environ,
        )
        result = executor.collect_exact(
            acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
            batch_id=args.batch_id,
            exact_seed_catalog=args.exact_seed_catalog,
            baseline_manifest=args.baseline_manifest,
            output_root=args.output_root,
            max_documents=int(args.max_documents),
            timeout=float(budget["timeout_seconds"]),
            max_bytes=int(budget["max_response_bytes"]),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _require_network_acknowledgements(args, "--discover")
    if not args.execute_approved_phase_a:
        parser().error("--discover requires --execute-approved-phase-a")
    if args.preflight_report is None:
        parser().error("--discover requires --preflight-report")
    configured_collection_budget = int(budget["collection_maximum_requests"])
    if int(args.maximum_total_requests) != configured_collection_budget:
        parser().error("Phase A collection request budget must equal the reviewed configured ceiling")
    preflight_report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    executor = PhaseAExecutor(config, preflight_report, environment=os.environ)
    result = executor.collect(
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT, batch_id=args.batch_id,
        keywords=config["priority_queries"], output_root=args.output_root,
        timeout=float(budget["timeout_seconds"]), max_bytes=int(budget["max_response_bytes"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
