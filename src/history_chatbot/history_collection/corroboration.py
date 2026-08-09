"""Claim-level corroboration that preserves unresolved historical conflicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class FactCandidate:
    subject: str
    predicate: str
    value: str
    normalized_value: str
    date_precision: str
    place: str
    evidence_span: str
    candidate_id: str
    publisher_family: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FactAssessment:
    subject: str
    predicate: str
    place: str
    status: str
    corroborated: bool
    fact_conflict: bool
    conflicting_values: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    supporting_sources: tuple[str, ...] = field(default_factory=tuple)
    unresolved_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["conflicting_values"] = list(self.conflicting_values)
        value["supporting_sources"] = list(self.supporting_sources)
        return value


def assess_facts(facts: Iterable[FactCandidate]) -> list[FactAssessment]:
    groups: dict[tuple[str, str, str], list[FactCandidate]] = {}
    for fact in facts:
        groups.setdefault((fact.subject.strip(), fact.predicate.strip(), fact.place.strip()), []).append(fact)
    assessments: list[FactAssessment] = []
    for (subject, predicate, place), members in sorted(groups.items()):
        by_value: dict[str, dict[str, FactCandidate]] = {}
        for fact in members:
            by_value.setdefault(fact.normalized_value, {})[fact.publisher_family] = fact
        values = sorted(by_value)
        independent = sorted({fact.publisher_family for fact in members})
        if len(values) > 1:
            conflicts = tuple({"value": value, "supporting_sources": sorted(by_value[value])} for value in values)
            assessments.append(FactAssessment(subject, predicate, place, "conflict", False, True,
                                               conflicts, tuple(independent),
                                               "독립 출처가 상충하여 자동 확정하지 않음"))
        else:
            corroborated = len(independent) >= 2
            assessments.append(FactAssessment(subject, predicate, place,
                                               "corroborated" if corroborated else "single_source",
                                               corroborated, False, (), tuple(independent),
                                               "" if corroborated else "독립 publisher family가 2개 미만"))
    return assessments
