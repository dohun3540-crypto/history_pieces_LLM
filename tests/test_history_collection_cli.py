import hashlib
import json
from pathlib import Path

import pytest

from scripts.collect_history_1000 import main


ROOT = Path(__file__).parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dry_run_is_default_and_network_free(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--phase", "A", "--batch-id", "test", "--target-unique", "50"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_requests"] == 0 and result["network"] is False
    assert result["source_plan_total"] == 50
    assert not list(tmp_path.iterdir())


def test_discover_requires_preflight_and_makes_zero_requests(capsys):
    with pytest.raises(SystemExit):
        main(["--phase", "A", "--discover", "--allow-network"])
    assert capsys.readouterr().out == ""


def test_hackathon_baseline_is_exact_and_unchanged_by_dry_run(capsys):
    manifest = ROOT / "data/provisional_hackathon/manifests/sources.jsonl"
    chunks = ROOT / "data/provisional_hackathon/processed/chunks.jsonl"
    raw = ROOT / "data/provisional_hackathon/raw"
    before = {manifest: digest(manifest), chunks: digest(chunks)}
    document_ids = {json.loads(line)["document_id"] for line in chunks.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(document_ids) == 48
    assert sum(bool(line.strip()) for line in chunks.read_text(encoding="utf-8").splitlines()) == 239
    assert len(list(raw.glob("*.html"))) == 48
    assert sum(bool(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines()) == 50
    assert main(["--phase", "A", "--dry-run"]) == 0
    capsys.readouterr()
    assert before == {manifest: digest(manifest), chunks: digest(chunks)}
