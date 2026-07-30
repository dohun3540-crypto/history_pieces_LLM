from history_chatbot.retrieval.document import Document
from history_chatbot.retrieval.retriever import KeywordRetriever


def test_keyword_retriever_finds_sample_document() -> None:
    document = Document(
        id="sample",
        title="목포 근대역사 샘플 안내",
        source="프로토타입 내부 테스트",
        content="이 문서는 검색 기능 확인을 위한 샘플 자료입니다.",
    )
    results = KeywordRetriever([document]).search("목포 샘플")
    assert len(results) == 1
    assert results[0].document.title == "목포 근대역사 샘플 안내"
    assert results[0].document.source == "프로토타입 내부 테스트"
