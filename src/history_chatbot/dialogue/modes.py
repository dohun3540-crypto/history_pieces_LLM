from enum import StrEnum

from history_chatbot.dialogue.situation_models import SituationId


class ConversationMode(StrEnum):
    PIECE_CHAT = "piece_chat"
    FREE_CHAT = "free_chat"


PIECE_CHAT_MAX_TURNS = 3

# Both surfaces may receive operational or safety requests. Keeping this explicit
# prevents a screen from silently falling back to a history situation.
PIECE_CHAT_ALLOWED_SITUATIONS = frozenset(SituationId)
FREE_CHAT_ALLOWED_SITUATIONS = frozenset(SituationId)
