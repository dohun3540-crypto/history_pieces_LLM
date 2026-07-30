from history_chatbot.ingestion.source_registry import SourceRegistry


def test_source_registry_registers_and_reads_utf8(tmp_path, source_factory) -> None:
    registry = SourceRegistry(tmp_path / "sources.jsonl")
    source = source_factory(tmp_path / "raw" / "virtual.txt")
    registry.register(source)

    loaded = registry.get(source.document_id)
    assert loaded.title == "테스트용 가상 자료"
    assert loaded.document_id == source.document_id


def test_source_registry_rejects_duplicate_id(tmp_path, source_factory) -> None:
    registry = SourceRegistry(tmp_path / "sources.jsonl")
    source = source_factory(tmp_path / "raw" / "virtual.txt")
    registry.register(source)

    try:
        registry.register(source)
    except ValueError as error:
        assert "이미 등록된" in str(error)
    else:
        raise AssertionError("중복 document_id가 허용되었습니다.")
