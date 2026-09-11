from __future__ import annotations

import pytest

from typobenchpl import RuleId, analyze, check


@pytest.mark.parametrize(
    ("text", "rule_id"),
    [
        ("", RuleId.EMPTY_TEXT),
        ("Tekst\x00test", RuleId.FORBIDDEN_CONTROL),
        ("Tekst\ud800test", RuleId.UNPAIRED_SURROGATE),
        ("To jest,, błąd.", RuleId.REPEATED_COMMA),
        ("To jest;; błąd.", RuleId.REPEATED_SEMICOLON),
        ("To jest.. błąd.", RuleId.INVALID_DOT_RUN),
        ("To jest , błąd.", RuleId.SPACE_BEFORE_COMMA),
        ("Pierwsza część ; druga.", RuleId.SPACE_BEFORE_SEMICOLON),
        ("Powiedział : tak.", RuleId.SPACE_BEFORE_COLON),
        ("Koniec .", RuleId.SPACE_BEFORE_PERIOD),
        ("Czy to działa ?", RuleId.SPACE_BEFORE_QUESTION),
        ("Uwaga !", RuleId.SPACE_BEFORE_EXCLAMATION),
        ("Ala,ma kota.", RuleId.MISSING_SPACE_AFTER_COMMA),
        ("Ala ma kota;Jan ma psa.", RuleId.MISSING_SPACE_AFTER_SEMICOLON),
        ("Powiedział:tak.", RuleId.MISSING_SPACE_AFTER_COLON),
        ("Naprawdę?Tak.", RuleId.MISSING_SPACE_AFTER_QUESTION_OR_EXCLAMATION),
        ("To jest ([błąd)] tutaj.", RuleId.INVALID_BRACKET_ORDER),
        ("To jest (błąd.", RuleId.UNMATCHED_BRACKET),
        ("To jest błąd).", RuleId.UNMATCHED_BRACKET),
    ],
)
def test_rejects_known_errors(text: str, rule_id: RuleId) -> None:
    result = analyze(text)

    assert result.value == 0
    assert result.issue is not None
    assert result.issue.rule_id is rule_id


@pytest.mark.parametrize(
    "text",
    [
        "To jest poprawne zdanie.",
        "Temperatura wynosi 3,14°C.",
        "Spotkanie jest o 12:30.",
        "Wersja 1.2.3 działa.",
        "Serwer ma adres 192.168.1.1.",
        "Naprawdę?!",
        "Nie!!",
        "To było... interesujące.",
        "Powiedział: „Tak!”",
        "To jest ([poprawny]) przykład.",
        "Adres to https://example.com/a,,b.",
        "Napisz na test@example.com.",
    ],
)
def test_accepts_supported_prose(text: str) -> None:
    assert check(text) == 1


def test_invalid_utf8_uses_byte_offsets() -> None:
    result = analyze(b"tekst\xfftest")

    assert result.value == 0
    assert result.issue is not None
    assert result.issue.rule_id is RuleId.INVALID_UTF8
    assert (result.issue.start, result.issue.end, result.issue.offset_unit) == (5, 6, "byte")


def test_issue_offsets_use_unicode_codepoints() -> None:
    result = analyze("Zażółć,, tekst.")

    assert result.issue is not None
    assert result.issue.rule_id is RuleId.REPEATED_COMMA
    assert (result.issue.start, result.issue.end) == (6, 8)


def test_result_serialization() -> None:
    assert analyze("Poprawnie.").as_dict() == {
        "result": 1,
        "status": "PASS",
        "issues": [],
    }
