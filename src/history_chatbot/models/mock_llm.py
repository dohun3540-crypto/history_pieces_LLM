"""외부 모델이나 API 없이 RAG 흐름을 검증하는 mock 모델."""

from history_chatbot.fallback import fallback_response
from history_chatbot.models.base import BaseLLM, GenerationRequest


class MockLLM(BaseLLM):
    def __init__(self, fallback_message: str) -> None:
        self.fallback_message = fallback_message

    def generate(self, request: GenerationRequest) -> str:
        if not request.contexts:
            return fallback_response(self.fallback_message)

        evidence = "\n".join(
            f"- {result.document.content}\n"
            f"  자료: {result.document.title} | 출처: {result.document.source}"
            for result in request.contexts
        )
        return f"검색된 자료를 바탕으로 안내합니다.\n{evidence}"
