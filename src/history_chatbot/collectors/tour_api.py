"""한국관광공사 국문 관광정보 API의 보수적인 파일럿 수집기."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from history_chatbot.collectors.base import CollectedCandidate
from history_chatbot.collectors.status import tour_api_status
from history_chatbot.ingestion.source_registry import SourceRegistry


DEFAULT_BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
PILOT_KEYWORDS = (
    "목포 근대역사",
    "목포 개항",
    "목포 근대역사관",
    "구 목포 일본영사관",
    "동양척식주식회사 목포지점",
    "목포 독립운동",
)
MAX_TOTAL_RESULTS = 20
MAX_RESULTS_PER_KEYWORD = 5


class TourApiError(RuntimeError):
    """비밀정보를 포함하지 않는 사용자용 API 오류."""


class TourApiTransport(Protocol):
    def get_json(self, endpoint: str, params: dict[str, str], *, timeout: float) -> dict:
        """지정한 API endpoint를 호출한다."""


class UrllibTourApiTransport:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_json(self, endpoint: str, params: dict[str, str], *, timeout: float) -> dict:
        # URL은 이 메서드 밖으로 노출하지 않으며 예외에도 서비스 키를 포함하지 않는다.
        url = f"{self.base_url}/{endpoint}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "MokpoHistoryRAGCollector/0.1",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise TourApiError(f"Tour API {endpoint} 요청에 실패했습니다.") from error


@dataclass(frozen=True, slots=True)
class TourApiItem:
    keyword: str
    content_id: str
    title: str
    source_url: str
    modified_time: str
    overview: str


@dataclass(frozen=True, slots=True)
class TourApiCollection:
    candidates: tuple[CollectedCandidate, ...]
    excluded: tuple[str, ...]


class TourApiCollector:
    def __init__(
        self,
        service_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: TourApiTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not service_key.strip():
            raise TourApiError(
                "TOUR_API_SERVICE_KEY가 없습니다. 실제 키를 .env 또는 현재 셸 환경변수에 "
                "설정한 뒤 다시 실행하세요."
            )
        self._service_key = service_key
        self.base_url = base_url.rstrip("/")
        self.transport = transport or UrllibTourApiTransport(self.base_url)
        self.timeout = timeout

    @classmethod
    def from_environment(cls, **kwargs) -> "TourApiCollector":
        status = tour_api_status()
        if not status.network_allowed:
            raise TourApiError(status.message)
        return cls(
            os.environ.get("TOUR_API_SERVICE_KEY", ""),
            base_url=os.environ.get("TOUR_API_BASE_URL", DEFAULT_BASE_URL),
            **kwargs,
        )

    def dry_run(self, keywords: tuple[str, ...] = PILOT_KEYWORDS) -> tuple[TourApiItem, ...]:
        selected: list[TourApiItem] = []
        seen: set[str] = set()
        for keyword in keywords:
            if len(selected) >= MAX_TOTAL_RESULTS:
                break
            for result in self.search(keyword)[:MAX_RESULTS_PER_KEYWORD]:
                content_id = str(result.get("contentid", "")).strip()
                if not content_id or content_id in seen:
                    continue
                detail = self.fetch_detail(content_id)
                overview = str(detail.get("overview", "")).strip()
                if not overview:
                    continue
                selected.append(
                    TourApiItem(
                        keyword=keyword,
                        content_id=content_id,
                        title=str(detail.get("title") or result.get("title") or "").strip(),
                        source_url=_source_url(detail.get("homepage"), content_id),
                        modified_time=str(
                            detail.get("modifiedtime") or result.get("modifiedtime") or ""
                        ).strip(),
                        overview=overview,
                    )
                )
                seen.add(content_id)
                if len(selected) >= MAX_TOTAL_RESULTS:
                    break
        return tuple(selected)

    def search(self, keyword: str) -> list[dict]:
        payload = self.transport.get_json(
            "searchKeyword2",
            self._common_params(
                keyword=keyword,
                numOfRows=str(MAX_RESULTS_PER_KEYWORD),
                pageNo="1",
                arrange="A",
            ),
            timeout=self.timeout,
        )
        return _items(payload)

    def fetch_detail(self, content_id: str) -> dict:
        payload = self.transport.get_json(
            "detailCommon2",
            self._common_params(
                contentId=content_id,
                defaultYN="Y",
                firstImageYN="N",
                areacodeYN="N",
                catcodeYN="N",
                addrinfoYN="N",
                mapinfoYN="N",
                overviewYN="Y",
            ),
            timeout=self.timeout,
        )
        items = _items(payload)
        return items[0] if items else {}

    def collect(
        self,
        *,
        raw_dir: Path,
        extracted_dir: Path,
        catalog_path: Path,
        manifest_path: Path,
        prepared_items: tuple[TourApiItem, ...] | None = None,
    ) -> TourApiCollection:
        items = prepared_items if prepared_items is not None else self.dry_run()
        raw_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        registry = SourceRegistry(manifest_path)
        existing_ids = {document.document_id for document in registry.list()}
        candidates: list[CollectedCandidate] = []
        excluded: list[str] = []

        for item in items:
            document_id = f"tour-api-{item.content_id}"
            if document_id in existing_ids:
                excluded.append(f"{document_id}: manifest에 이미 존재")
                continue
            raw_path = raw_dir / f"{document_id}.json"
            extracted_path = extracted_dir / f"{document_id}.txt"
            raw_payload = json.dumps(asdict(item), ensure_ascii=False, indent=2)
            raw_path.write_text(raw_payload, encoding="utf-8", newline="\n")
            extracted_path.write_text(item.overview, encoding="utf-8", newline="\n")
            candidate = CollectedCandidate(
                document_id=document_id,
                source_id="tour_api",
                source_url=item.source_url,
                title=item.title,
                publisher="한국관광공사",
                published_date="",
                accessed_date=date.today().isoformat(),
                language="ko",
                license_name="",
                license_url="https://www.data.go.kr/data/15101578/openapi.do",
                copyright_status="unknown",
                allowed_for_rag=False,
                allowed_for_training=False,
                redistribution_allowed=False,
                trust_grade="A",
                rag_priority_candidate=False,
                review_status="draft",
                raw_path=str(raw_path),
                extracted_path=str(extracted_path),
                ocr_path="",
                content_sha256=hashlib.sha256(item.overview.encode("utf-8")).hexdigest(),
                notes=f"Tour API contentid={item.content_id}; 자료별 이용조건 검증 대기",
            )
            with catalog_path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
            registry.register(candidate.to_source_document())
            existing_ids.add(document_id)
            candidates.append(candidate)
        return TourApiCollection(tuple(candidates), tuple(excluded))

    def _common_params(self, **params: str) -> dict[str, str]:
        return {
            "serviceKey": self._service_key,
            "MobileOS": "ETC",
            "MobileApp": "MokpoHistoryRAG",
            "_type": "json",
            **params,
        }


def _items(payload: dict) -> list[dict]:
    try:
        header = payload["response"]["header"]
        if str(header.get("resultCode", "0000")) != "0000":
            raise TourApiError(
                f"Tour API가 오류를 반환했습니다: {header.get('resultMsg', '상세 불명')}"
            )
    except (KeyError, TypeError):
        pass
    try:
        value = payload["response"]["body"]["items"]["item"]
    except (KeyError, TypeError):
        return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _source_url(homepage: object, content_id: str) -> str:
    value = str(homepage or "").strip()
    match = re.search(r"""href=["']([^"']+)""", value, flags=re.IGNORECASE)
    if match:
        value = match.group(1)
    if value.startswith(("https://", "http://")):
        return value
    # 개별 URL을 추측하지 않고 공식 API 명세 페이지로 추적한다.
    return "https://www.data.go.kr/data/15101578/openapi.do"
