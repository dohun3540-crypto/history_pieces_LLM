"""수동 메타데이터를 기본으로 하는 확장 인터페이스."""

from abc import ABC, abstractmethod

from history_chatbot.ingestion.models import SourceDocument


class MetadataEnricher(ABC):
    """자동 추출 결과는 사실 확정이 아니라 검수 후보만 반환해야 한다."""

    @abstractmethod
    def suggest(self, text: str, document: SourceDocument) -> dict[str, object]:
        """사람의 검수를 거쳐야 하는 메타데이터 후보를 반환한다."""


class ManualMetadataEnricher(MetadataEnricher):
    def suggest(self, text: str, document: SourceDocument) -> dict[str, object]:
        return {}
