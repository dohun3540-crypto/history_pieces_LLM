import json

from history_chatbot.ingestion.models import ReviewStatus
from history_chatbot.ingestion.pipeline import IngestionPipeline
from history_chatbot.ingestion.source_registry import SourceRegistry


def test_valid_virtual_txt_runs_through_pipeline(tmp_path, source_factory) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "virtual.txt"
    raw_path.write_text(
        "테스트용 가상 자료입니다. 실제 역사적 사실이 아닙니다.\n\n"
        "  파이프라인   확인용 둘째 문단입니다.  ",
        encoding="utf-8",
    )
    registry = SourceRegistry(tmp_path / "data" / "manifests" / "sources.jsonl")
    source = source_factory(raw_path)
    registry.register(source)

    result = IngestionPipeline(
        registry=registry,
        raw_root=raw_dir,
        extracted_dir=tmp_path / "data" / "extracted",
        processed_dir=tmp_path / "data" / "processed",
    ).process(source.document_id)

    assert result.document.review_status == ReviewStatus.METADATA_ADDED
    assert result.chunks
    assert all(chunk.document_id == source.document_id for chunk in result.chunks)
    output_lines = [
        json.loads(line)
        for line in (tmp_path / "data" / "processed" / f"{source.document_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert output_lines[0]["document_id"] == source.document_id
    assert registry.get(source.document_id).review_status == ReviewStatus.METADATA_ADDED
