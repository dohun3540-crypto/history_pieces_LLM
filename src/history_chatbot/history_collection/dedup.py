"""Indexed document deduplication without all-pairs text comparison."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from history_chatbot.history_collection.models import CandidateDocument, DuplicateStatus
from history_chatbot.history_collection.quality import normalize_body


TRACKING_KEYS = {"fbclid", "gclid", "ref", "source", "utm_campaign", "utm_medium", "utm_source"}


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode(sorted((key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                             if key.lower() not in TRACKING_KEYS))
    host = (parts.hostname or "").lower()
    port = "" if parts.port in (None, 80, 443) else ":%d" % parts.port
    return urlunsplit((parts.scheme.lower(), host + port, parts.path or "/", query, ""))


def normalized_body_hash(text: str) -> str:
    compact = normalize_body(text).lower()
    return hashlib.sha256(compact.encode("utf-8")).hexdigest() if compact else ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣]+", normalize_body(text).lower())


def simhash64(text: str) -> int:
    vector = [0] * 64
    tokens = _tokens(text)
    for index in range(max(1, len(tokens) - 2)):
        feature = " ".join(tokens[index:index + 3])
        digest = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    return sum((1 << bit) for bit, value in enumerate(vector) if value >= 0)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def metadata_key(candidate: CandidateDocument) -> str:
    title = " ".join(_tokens(candidate.source_title))
    published = candidate.publication_metadata.get("published_date", "")
    record_id = candidate.publication_metadata.get("document_number", "")
    if not (published or record_id):
        return ""
    return "\0".join((candidate.publisher_family.strip().lower(), title, published, record_id))


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    status: DuplicateStatus
    method: str = ""
    duplicate_of: str = ""
    group: str = ""
    similarity: float = 0.0


class DuplicateIndex:
    """Exact indexes plus banded SimHash candidate generation."""

    def __init__(self, *, near_distance: int = 8) -> None:
        self.near_distance = near_distance
        self.urls: dict[str, str] = {}
        self.raw_hashes: dict[str, str] = {}
        self.body_hashes: dict[str, str] = {}
        self.metadata: dict[str, str] = {}
        self.fingerprints: dict[str, int] = {}
        self.bands: dict[tuple[int, int], set[str]] = {}
        self.records: dict[str, CandidateDocument] = {}

    @staticmethod
    def _group(method: str, retained: str) -> str:
        return "dup-" + hashlib.sha256((method + "\0" + retained).encode("utf-8")).hexdigest()[:16]

    def check(self, candidate: CandidateDocument) -> DuplicateDecision:
        exact_checks = (
            ("canonical_url", canonicalize_url(candidate.canonical_url or candidate.source_url), self.urls),
            ("raw_sha256", candidate.raw_sha256, self.raw_hashes),
            ("normalized_body_sha256", normalized_body_hash(candidate.body_text), self.body_hashes),
            ("publication_metadata", metadata_key(candidate), self.metadata),
        )
        for method, value, index in exact_checks:
            if value and value in index:
                retained = index[value]
                return DuplicateDecision(DuplicateStatus.CONFIRMED, method, retained,
                                         self._group(method, retained), 1.0)

        fingerprint = simhash64(candidate.body_text)
        nearby: set[str] = set()
        for band in range(4):
            nearby.update(self.bands.get((band, (fingerprint >> (band * 16)) & 0xFFFF), set()))
        for other_id in sorted(nearby):
            distance = hamming_distance(fingerprint, self.fingerprints[other_id])
            if distance <= self.near_distance:
                other = self.records[other_id]
                work_id = candidate.provenance.get("canonical_work_id")
                mirrored = bool(work_id and work_id == other.provenance.get("canonical_work_id") and
                                candidate.publisher_family != other.publisher_family)
                method = "mirror" if mirrored else "simhash_near"
                status = DuplicateStatus.CONFIRMED if mirrored else DuplicateStatus.SUSPECTED
                return DuplicateDecision(status, method, other_id, self._group(method, other_id),
                                         round(1 - distance / 64, 4))
        return DuplicateDecision(DuplicateStatus.UNIQUE)

    def add(self, candidate: CandidateDocument) -> DuplicateDecision:
        decision = self.check(candidate)
        candidate.duplicate_status = decision.status
        candidate.duplicate_method = decision.method
        candidate.duplicate_of = decision.duplicate_of
        candidate.duplicate_group = decision.group
        candidate.uniqueness_score = 0 if decision.status == DuplicateStatus.CONFIRMED else 5 if decision.status == DuplicateStatus.SUSPECTED else 10
        if decision.status == DuplicateStatus.CONFIRMED:
            return decision
        candidate_id = candidate.candidate_id
        self.records[candidate_id] = candidate
        url = canonicalize_url(candidate.canonical_url or candidate.source_url)
        if url:
            self.urls[url] = candidate_id
        if candidate.raw_sha256:
            self.raw_hashes[candidate.raw_sha256] = candidate_id
        body_hash = normalized_body_hash(candidate.body_text)
        if body_hash:
            self.body_hashes[body_hash] = candidate_id
        key = metadata_key(candidate)
        if key:
            self.metadata[key] = candidate_id
        fingerprint = simhash64(candidate.body_text)
        self.fingerprints[candidate_id] = fingerprint
        for band in range(4):
            self.bands.setdefault((band, (fingerprint >> (band * 16)) & 0xFFFF), set()).add(candidate_id)
        return decision
