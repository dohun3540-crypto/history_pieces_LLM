"""tokenizer가 없을 때도 동작하는 보수적 컨텍스트 예산 관리자."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


class ConservativeCharacterEstimator:
    """한국어·다국어 혼합 입력을 문자 2개당 약 1 token으로 넉넉히 추정한다."""

    def estimate(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 2)) if text else 0


@dataclass(frozen=True, slots=True)
class BudgetResult:
    system_prompt: str
    user_prompt: str
    evidence: tuple[str, ...]
    conversation: tuple[str, ...]
    estimated_input_tokens: int
    reserved_output_tokens: int
    trimmed_evidence: int
    trimmed_conversation: int


class ContextBudgetManager:
    def __init__(
        self,
        context_window: int,
        *,
        estimator: TokenEstimator | None = None,
    ) -> None:
        if context_window < 256:
            raise ValueError("context_window는 최소 256 token이어야 합니다.")
        self.context_window = context_window
        self.estimator = estimator or ConservativeCharacterEstimator()

    def fit(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence: Sequence[str],
        conversation: Sequence[str],
        max_new_tokens: int,
    ) -> BudgetResult:
        available = self.context_window - max_new_tokens
        required = self.estimator.estimate(system_prompt) + self.estimator.estimate(user_prompt)
        if available <= required:
            raise ValueError("context_length_exceeded: 시스템 지침과 현재 질문을 유지할 수 없습니다.")

        used = required
        kept_conversation: list[str] = []
        # The immediately previous exchange is needed to resolve ellipsis and
        # answer-transformation requests. Reserve it before evidence, then keep
        # high-score evidence ahead of older chat history.
        recent = conversation[-1:] if conversation else ()
        for item in recent:
            cost = self.estimator.estimate(item)
            if used + cost <= available:
                kept_conversation.append(item)
                used += cost

        kept_evidence: list[str] = []
        for item in evidence:  # 이미 검색 점수 순
            cost = self.estimator.estimate(item)
            if used + cost <= available:
                kept_evidence.append(item)
                used += cost

        older_kept: list[str] = []
        for item in reversed(conversation[:-1]):  # 최근 대화 우선
            cost = self.estimator.estimate(item)
            if used + cost <= available:
                older_kept.append(item)
                used += cost
        kept_conversation = list(reversed(older_kept)) + kept_conversation
        return BudgetResult(
            system_prompt,
            user_prompt,
            tuple(kept_evidence),
            tuple(kept_conversation),
            used,
            max_new_tokens,
            len(evidence) - len(kept_evidence),
            len(conversation) - len(kept_conversation),
        )
