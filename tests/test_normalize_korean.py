import unicodedata

from history_chatbot.preprocessing.normalize_korean import normalize_korean
from history_chatbot.preprocessing.query_rewriter import Query


def test_normalizes_korean_whitespace_and_unicode_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "목포")
    assert normalize_korean(f"  {decomposed}\t 근대  역사\n") == "목포 근대 역사"


def test_query_preserves_original_query() -> None:
    query = Query.from_text("  목포   역사  ")
    assert query.original_query == "  목포   역사  "
    assert query.normalized_query == "목포 역사"
