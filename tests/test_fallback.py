from history_chatbot.models.base import GenerationRequest
from history_chatbot.models.mock_llm import MockLLM
from history_chatbot.settings import DEFAULT_FALLBACK


def test_mock_llm_returns_fallback_without_context() -> None:
    llm = MockLLM(DEFAULT_FALLBACK)
    request = GenerationRequest("없는 질문", "없는 질문", ())
    assert llm.generate(request) == "제공된 목포 근대역사 자료에서 확인할 수 없습니다."
