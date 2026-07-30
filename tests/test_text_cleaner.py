import unicodedata

from history_chatbot.ingestion.cleaner import TextCleaner


def test_cleaner_normalizes_unicode_spaces_and_page_lines() -> None:
    decomposed = unicodedata.normalize("NFD", "목포")
    original = f"  {decomposed}   테스트\r\n\r\n12\r\n\r\n\r\n  둘째   문단  "
    result = TextCleaner(remove_page_number_lines=True).clean(original)

    assert result.original_text == original
    assert result.cleaned_text == "목포 테스트\n\n둘째 문단"
    assert "독립 숫자 페이지 줄 제거" in result.cleaning_log
