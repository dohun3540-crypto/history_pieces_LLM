"""Legacy RAG helper backed by the canonical final Giroksae persona."""

from history_chatbot.dialogue.modes import ConversationMode
from history_chatbot.dialogue.persona import OutputDomain, build_persona_prompt
from history_chatbot.dialogue.situation_models import SituationId

SYSTEM_PROMPT_KO = build_persona_prompt(
    domain=OutputDomain.CHARACTER_DIALOGUE,
    locale="ko",
    mode=ConversationMode.FREE_CHAT,
    situation=SituationId.HISTORY_FACT_QUESTION,
)


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    joined = "\n\n".join(contexts)
    return f"{SYSTEM_PROMPT_KO}\n\n[문맥]\n{joined}\n\n[질문]\n{query}"
