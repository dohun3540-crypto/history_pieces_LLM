"""개발·테스트·운영 실행 모드와 데이터 안전 경계."""

from __future__ import annotations

from enum import StrEnum


FIXTURE_NOTICE = "테스트용 가상 자료이며 실제 역사 사실이 아님"


class RuntimeMode(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    HACKATHON = "hackathon"
    PRODUCTION = "production"

    @classmethod
    def parse(cls, value: str) -> "RuntimeMode":
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError(
                "실행 모드는 development, test, hackathon, production 중 하나여야 합니다."
            ) from error

    @property
    def allows_fixtures(self) -> bool:
        return self in {self.DEVELOPMENT, self.TEST}


class ProductionNotReadyError(RuntimeError):
    """운영에 사용할 실제 검수 자료가 아직 없을 때 발생한다."""


class ProvisionalDataDetectedError(RuntimeError):
    """운영 경로에 임시 해커톤 자료가 탐지되었을 때 발생한다."""

    code = "provisional_data_detected"


class ProvisionalIndexDetectedError(RuntimeError):
    """운영 경로에서 해커톤 전용 인덱스가 탐지되었을 때 발생한다."""

    code = "provisional_index_detected"
