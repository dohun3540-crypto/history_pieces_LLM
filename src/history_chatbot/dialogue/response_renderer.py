"""Domain-aware response rendering without changing preserved seed originals."""

from __future__ import annotations

from dataclasses import dataclass

from history_chatbot.dialogue.persona import (
    ConversationStage, GiroksaeStyleGuard, OutputDomain, SpeechLevel,
    speech_level_for,
)
from history_chatbot.dialogue.situation_models import SituationId


@dataclass(frozen=True, slots=True)
class RenderedDialogue:
    text: str
    output_domain: OutputDomain
    speech_level: SpeechLevel


class GiroksaeResponseRenderer:
    def __init__(self, guard: GiroksaeStyleGuard | None = None) -> None:
        self.guard = guard or GiroksaeStyleGuard()

    def render(
        self, text: str, *, domain: OutputDomain, situation: SituationId,
        stage: ConversationStage, locale: str = "ko",
        citations: tuple[dict[str, object], ...] = (),
    ) -> RenderedDialogue:
        normalized = " ".join(text.split())
        self.guard.ensure(
            normalized, domain=domain, situation=situation, stage=stage,
            locale=locale, citations=citations,
        )
        return RenderedDialogue(normalized, domain, speech_level_for(domain))

    def journey_caption(
        self, user_words: str, *, storage_capability: bool = False,
        user_consent: bool = False,
    ) -> RenderedDialogue:
        """Use only supplied user words; this does not infer or persist sentiment."""
        text = " ".join(user_words.split()).strip()
        if not text:
            raise ValueError("여정필름 캡션에는 사용자가 실제로 표현한 내용이 필요합니다.")
        if not (storage_capability and user_consent):
            text = text.replace("저장했습니다", "이번 여정에서 표현한 장면")
        return RenderedDialogue(
            text, OutputDomain.JOURNEY_FILM_CAPTION,
            SpeechLevel.NEUTRAL_CAPTION,
        )
