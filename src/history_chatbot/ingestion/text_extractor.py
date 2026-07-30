"""로컬 원문에서 텍스트를 추출한다. 네트워크 접근은 하지 않는다."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from history_chatbot.ingestion.models import ExtractionResult


class TextExtractionError(RuntimeError):
    pass


class BaseTextExtractor(ABC):
    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:
        """파일에서 원문 텍스트를 추출한다."""


class PlainTextExtractor(BaseTextExtractor):
    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}

    def extract(self, path: Path) -> ExtractionResult:
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise TextExtractionError(f"지원하지 않는 텍스트 형식입니다: {path.suffix}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise TextExtractionError("원문은 UTF-8 TXT 또는 Markdown이어야 합니다.") from error
        return ExtractionResult(text, "plain_text_utf8", str(path))


class PdfTextExtractor(BaseTextExtractor):
    def extract(self, path: Path) -> ExtractionResult:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as error:
            raise TextExtractionError(
                "PDF 추출에는 선택적 의존성 'pypdf'가 필요합니다. "
                "현재 기본 설치에는 포함되지 않습니다."
            ) from error
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        return ExtractionResult("\n\n".join(pages), "pypdf", str(path))


def extract_text(path: Path) -> ExtractionResult:
    suffix = path.suffix.lower()
    if suffix in PlainTextExtractor.SUPPORTED_SUFFIXES:
        return PlainTextExtractor().extract(path)
    if suffix == ".pdf":
        return PdfTextExtractor().extract(path)
    if suffix in {".html", ".htm"}:
        raise TextExtractionError(
            "HTML 및 웹페이지 수집은 아직 구현되지 않았습니다. "
            "향후 BaseTextExtractor 구현으로 확장할 수 있습니다."
        )
    raise TextExtractionError(f"지원하지 않는 원문 형식입니다: {suffix}")
