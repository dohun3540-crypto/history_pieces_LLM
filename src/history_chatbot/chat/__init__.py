"""대화형 RAG의 서비스 경계."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from history_chatbot.chat.service import HistoryChatService

__all__ = ["HistoryChatService"]


def __getattr__(name: str):
    if name == "HistoryChatService":
        from history_chatbot.chat.service import HistoryChatService

        return HistoryChatService
    raise AttributeError(name)
