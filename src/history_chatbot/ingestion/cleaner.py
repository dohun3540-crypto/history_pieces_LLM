"""원문의 의미를 바꾸지 않는 보수적 텍스트 정제."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod

from history_chatbot.ingestion.models import CleanedText


class RepeatedMarginRemover(ABC):
    """반복 머리말·꼬리말 제거 구현을 위한 확장점."""

    @abstractmethod
    def remove(self, text: str) -> tuple[str, str | None]:
        """정제문과 적용 로그를 반환한다."""


class TextCleaner:
    def __init__(
        self,
        remove_page_number_lines: bool = True,
        margin_removers: tuple[RepeatedMarginRemover, ...] = (),
    ) -> None:
        self.remove_page_number_lines = remove_page_number_lines
        self.margin_removers = margin_removers

    def clean(self, original_text: str) -> CleanedText:
        log: list[str] = []
        text = unicodedata.normalize("NFC", original_text)
        if text != original_text:
            log.append("Unicode NFC 정규화")

        normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized_newlines != text:
            log.append("줄바꿈을 LF로 통일")
        text = normalized_newlines

        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        compact_spaces = "\n".join(lines)
        if compact_spaces != text:
            log.append("앞뒤 및 연속 공백 정리")
        text = compact_spaces

        if self.remove_page_number_lines:
            filtered = "\n".join(
                line for line in text.split("\n") if not re.fullmatch(r"\d{1,4}", line)
            )
            if filtered != text:
                log.append("독립 숫자 페이지 줄 제거")
            text = filtered

        for remover in self.margin_removers:
            text, entry = remover.remove(text)
            if entry:
                log.append(entry)

        compact_blank_lines = re.sub(r"\n{3,}", "\n\n", text).strip()
        if compact_blank_lines != text:
            log.append("과도한 빈 줄 및 문서 가장자리 공백 정리")
        text = compact_blank_lines
        return CleanedText(original_text, text, tuple(log))
