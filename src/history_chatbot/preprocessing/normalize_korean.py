"""한국어를 포함한 Unicode 입력 정규화."""

import re
import unicodedata


def normalize_korean(text: str) -> str:
    """NFC 정규화 후 앞뒤 및 연속 공백을 정리한다."""
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()
