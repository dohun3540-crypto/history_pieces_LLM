"""애플리케이션 설정 객체."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_FALLBACK = "제공된 목포 근대역사 자료에서 확인할 수 없습니다."


@dataclass(frozen=True, slots=True)
class Settings:
    data_path: Path
    top_k: int = 3
    fallback_message: str = DEFAULT_FALLBACK
    memory_max_turns: int = 20

    @classmethod
    def default(cls, project_root: Path | None = None) -> "Settings":
        root = project_root or Path(__file__).resolve().parents[2]
        return cls(data_path=root / "data" / "sample" / "mokpo_history_sample.json")
