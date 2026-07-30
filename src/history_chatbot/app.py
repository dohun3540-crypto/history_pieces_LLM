"""RAG 구성 요소를 연결하는 애플리케이션 서비스."""

from __future__ import annotations

from history_chatbot.memory.conversation import ConversationMemory, Turn
from history_chatbot.models.base import BaseLLM, GenerationRequest
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.preprocessing.query_rewriter import Query
from history_chatbot.retrieval.loader import load_json_documents
from history_chatbot.retrieval.retriever import BaseRetriever, KeywordRetriever
from history_chatbot.settings import Settings


class HistoryChatbot:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        memory: ConversationMemory,
        top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.memory = memory
        self.top_k = top_k

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "HistoryChatbot":
        resolved = settings or Settings.default()
        documents = load_json_documents(resolved.data_path)
        return cls(
            retriever=KeywordRetriever(documents),
            llm=MockLLM(resolved.fallback_message),
            memory=ConversationMemory(resolved.memory_max_turns),
            top_k=resolved.top_k,
        )

    def ask(self, text: str) -> str:
        query = Query.from_text(text)
        contexts = tuple(self.retriever.search(query.normalized_query, self.top_k))
        answer = self.llm.generate(
            GenerationRequest(
                original_query=query.original_query,
                normalized_query=query.normalized_query,
                contexts=contexts,
            )
        )
        self.memory.add(
            Turn(
                original_query=query.original_query,
                normalized_query=query.normalized_query,
                answer=answer,
            )
        )
        return answer
