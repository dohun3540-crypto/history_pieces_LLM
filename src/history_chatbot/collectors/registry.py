"""수집기 선택과 URL·내용·유사 제목 중복 제거."""

from __future__ import annotations

import json
import re
from pathlib import Path

from history_chatbot.collectors.base import CollectedCandidate, CollectorConfig, canonicalize_url
from history_chatbot.collectors.heritage_portal import HeritagePortalCollector
from history_chatbot.collectors.history_database import HistoryDatabaseCollector
from history_chatbot.collectors.mokpo_city import MokpoCityCollector
from history_chatbot.collectors.national_archives import NationalArchivesCollector
from history_chatbot.collectors.open_access import OpenAccessCollector
from history_chatbot.collectors.public_nuri import PublicNuriCollector


COLLECTOR_TYPES = {
    "heritage_portal": HeritagePortalCollector,
    "history_database": HistoryDatabaseCollector,
    "mokpo_city": MokpoCityCollector,
    "national_archives": NationalArchivesCollector,
    "public_nuri": PublicNuriCollector,
    "open_access": OpenAccessCollector,
}


def build_collector(config: CollectorConfig, **kwargs):
    try:
        collector_class = COLLECTOR_TYPES[config.collector_type]
    except KeyError as error:
        raise ValueError(f"알 수 없는 collector_type입니다: {config.collector_type}") from error
    return collector_class(config, **kwargs)


class CandidateRegistry:
    def __init__(self, path: Path, similarity_threshold: float = 0.9) -> None:
        self.path = path
        self.similarity_threshold = similarity_threshold

    def list(self) -> list[CollectedCandidate]:
        if not self.path.exists():
            return []
        candidates: list[CollectedCandidate] = []
        with self.path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    candidates.append(CollectedCandidate(**json.loads(line)))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"후보 catalog {number}번째 줄 오류: {error}") from error
        return candidates

    def add_new(self, candidates: tuple[CollectedCandidate, ...]) -> list[CollectedCandidate]:
        existing = self.list()
        accepted: list[CollectedCandidate] = []
        for candidate in candidates:
            if self._is_duplicate(candidate, [*existing, *accepted]):
                continue
            accepted.append(candidate)
        if accepted:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                for candidate in accepted:
                    file.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
        return accepted

    def _is_duplicate(
        self, candidate: CollectedCandidate, others: list[CollectedCandidate]
    ) -> bool:
        candidate_url = canonicalize_url(candidate.source_url)
        for other in others:
            if candidate_url == canonicalize_url(other.source_url):
                return True
            if candidate.content_sha256 and candidate.content_sha256 == other.content_sha256:
                return True
            if self._title_similarity(candidate.title, other.title) >= self.similarity_threshold:
                return True
        return False

    @staticmethod
    def _title_similarity(left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", left.lower()))
        right_tokens = set(re.findall(r"[0-9A-Za-z가-힣]+", right.lower()))
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0
