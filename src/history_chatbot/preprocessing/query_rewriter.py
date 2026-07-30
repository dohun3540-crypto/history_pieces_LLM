"""원문 질문을 보존하는 질의 전처리."""

from dataclasses import dataclass

from history_chatbot.preprocessing.normalize_korean import normalize_korean


@dataclass(frozen=True, slots=True)
class Query:
    original_query: str
    normalized_query: str

    @classmethod
    def from_text(cls, text: str) -> "Query":
        return cls(original_query=text, normalized_query=normalize_korean(text))
