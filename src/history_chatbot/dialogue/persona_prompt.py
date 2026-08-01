"""Compatibility exports for the canonical final Giroksae persona."""

from history_chatbot.dialogue.persona import (
    CHARACTER_KO_PROMPT as GIROKSAE_PERSONA,
    PERSONA_ID,
    PERSONA_SOURCE,
)

DEFAULT_SPEECH_LEVEL = "banmal"
ZH_CN_POLICY_HOOK = "configs/giroksae_zh_cn_terms.json"

__all__ = [
    "DEFAULT_SPEECH_LEVEL", "GIROKSAE_PERSONA", "PERSONA_ID",
    "PERSONA_SOURCE", "ZH_CN_POLICY_HOOK",
]
