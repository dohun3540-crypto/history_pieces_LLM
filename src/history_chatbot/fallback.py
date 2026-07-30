"""근거 부족 시 일관된 응답을 제공한다."""

from history_chatbot.settings import DEFAULT_FALLBACK


def fallback_response(message: str = DEFAULT_FALLBACK) -> str:
    return message
