"""외부 API 연결 준비 상태."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApiStatus:
    source_id: str
    status: str
    network_allowed: bool
    message: str


def tour_api_status(environ: dict[str, str] | None = None) -> ApiStatus:
    values = os.environ if environ is None else environ
    if not values.get("TOUR_API_SERVICE_KEY", "").strip():
        return ApiStatus(
            "tour_api",
            "pending_credentials",
            False,
            "TOUR_API_SERVICE_KEY 설정 전에는 실제 네트워크 호출을 수행할 수 없습니다.",
        )
    return ApiStatus("tour_api", "ready_for_dry_run", True, "dry-run 실행 준비 완료")
