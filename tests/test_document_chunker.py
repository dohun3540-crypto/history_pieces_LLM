from history_chatbot.ingestion.chunker import DocumentChunker


def test_long_paragraph_is_split_and_traceable(tmp_path, source_factory) -> None:
    source = source_factory(tmp_path / "virtual.txt")
    text = ("첫 번째 테스트 문장입니다. " * 20).strip()
    chunks = DocumentChunker(max_chars=80, overlap=10).split(text, source)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == source.document_id for chunk in chunks)
    assert chunks[0].chunk_id == f"{source.document_id}::chunk-0000"
    assert all(chunk.text == text[chunk.start_char : chunk.end_char] for chunk in chunks)
