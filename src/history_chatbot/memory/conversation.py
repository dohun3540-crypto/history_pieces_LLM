"""모델 가중치와 분리된 세션 대화 메모리."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Turn:
    original_query: str
    normalized_query: str
    answer: str


class ConversationMemory:
    """현재 프로세스 안에서만 대화 기록을 보관한다.

    이 클래스는 모델을 학습하거나 대화를 학습 데이터에 추가하지 않는다.
    """

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns는 양수여야 합니다.")
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)

    def history(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def clear(self) -> None:
        self._turns.clear()
