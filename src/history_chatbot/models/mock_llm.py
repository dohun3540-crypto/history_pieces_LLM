"""외부 모델이나 API 없이 RAG 흐름을 검증하는 mock 모델."""

from collections.abc import Iterator

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

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Local/Remote streaming backend와 같은 기본 스트리밍 경계."""
        for token in self.generate(request).split(" "):
            yield token + " "

    def generate_grounded(
        self,
        *,
        prompt: str,
        evidence: tuple[str, ...],
        is_fixture: bool,
    ) -> str:
        del prompt
        if not evidence:
            return "확인 가능한 자료가 부족합니다."
        prefix = "[테스트용 응답] " if is_fixture else ""
        return f"{prefix}제공된 검색 근거 안에서만 안내합니다. " + " ".join(evidence)

    def stream_grounded(
        self,
        *,
        prompt: str,
        evidence: tuple[str, ...],
        is_fixture: bool,
    ) -> Iterator[str]:
        answer = self.generate_grounded(
            prompt=prompt, evidence=evidence, is_fixture=is_fixture
        )
        for token in answer.split(" "):
            yield token + " "
