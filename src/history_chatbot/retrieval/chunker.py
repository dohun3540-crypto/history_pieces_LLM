"""벡터 검색기로 전환할 때도 재사용 가능한 문자 기반 청킹."""

from history_chatbot.retrieval.document import Document


def chunk_document(
    document: Document, chunk_size: int = 500, overlap: int = 50
) -> list[Document]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size는 양수이고 overlap은 0 이상 chunk_size 미만이어야 합니다.")
    if len(document.content) <= chunk_size:
        return [document]

    chunks: list[Document] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(document.content), step)):
        content = document.content[start : start + chunk_size]
        if not content:
            break
        chunks.append(
            Document(
                id=f"{document.id}::chunk-{index}",
                title=document.title,
                source=document.source,
                content=content,
                language=document.language,
                metadata={**document.metadata, "parent_id": document.id},
            )
        )
        if start + chunk_size >= len(document.content):
            break
    return chunks
