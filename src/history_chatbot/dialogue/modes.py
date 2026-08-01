from enum import StrEnum


class ConversationMode(StrEnum):
    PIECE_CHAT = "piece_chat"
    FREE_CHAT = "free_chat"


PIECE_CHAT_MAX_TURNS = 3
