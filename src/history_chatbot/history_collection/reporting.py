"""Machine-readable and Markdown phase reports with non-overwriting paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from history_chatbot.history_collection.checkpoint import PhaseCheckpoint
from history_chatbot.history_collection.pipeline import atomic_write


def report_paths(root: Path, checkpoint: PhaseCheckpoint, batch_id: str) -> tuple[Path, Path]:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in batch_id)
    directory = root / ("phase_" + checkpoint.phase.value.lower())
    return directory / (safe + ".json"), directory / (safe + ".md")


def render_markdown(value: dict[str, Any]) -> str:
    lines = ["# History collection checkpoint", "",
             "- Phase: `%s`" % value["phase"],
             "- Gate: **%s**" % value["gate_status"],
             "- Unique candidates: %s / %s" % (value["unique_candidates"], value["target_unique"]),
             "- Duplicates: %s" % value["duplicate_count"],
             "- Auto candidates: %s" % value["accepted_candidate"],
             "- Human review: %s" % value["needs_human_review"],
             "- Auto rejected: %s" % value["auto_rejected"], "",
             "## Rates", ""]
    for name in ("extraction_success_rate", "rights_known_rate", "mokpo_relevance_rate",
                 "duplicate_rate", "noise_rate", "largest_publisher_share", "largest_topic_share"):
        lines.append("- %s: %.1f%%" % (name, float(value[name]) * 100))
    lines.extend(["", "## Stop reasons", ""])
    lines.extend("- " + reason for reason in value["stop_reasons"])
    if not value["stop_reasons"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_report(root: Path, checkpoint: PhaseCheckpoint, batch_id: str) -> tuple[Path, Path]:
    json_path, markdown_path = report_paths(root, checkpoint, batch_id)
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("동일 batch report를 덮어쓸 수 없습니다: " + batch_id)
    value = checkpoint.to_dict()
    value["batch_id"] = batch_id
    atomic_write(json_path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    atomic_write(markdown_path, render_markdown(value).encode("utf-8"))
    return json_path, markdown_path
