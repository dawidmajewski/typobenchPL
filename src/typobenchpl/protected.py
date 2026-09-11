from __future__ import annotations

import re

ProtectedRange = tuple[int, int]

_PROTECTED_PATTERN = re.compile(
    r"""
    (?P<url>https?://[^\s<>"']+)
    |
    (?P<email>(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-]))
    |
    (?P<dotted_number>(?<!\w)\d+(?:\.\d+)+(?!\w))
    |
    (?P<decimal_comma>(?<!\w)\d+,\d+(?!\w))
    |
    (?P<number_pair>(?<!\w)\d+:\d+(?!\w))
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TRAILING_URL_PUNCTUATION = ".,;:!?"
_CLOSING_DELIMITERS = {")": "(", "]": "[", "}": "{"}


def protected_ranges(text: str) -> tuple[ProtectedRange, ...]:
    ranges: list[ProtectedRange] = []

    for match in _PROTECTED_PATTERN.finditer(text):
        start, end = match.span()
        if match.lastgroup == "url":
            end = _trim_url(text, start, end)
        if start < end:
            ranges.append((start, end))

    if not ranges:
        return ()

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start < previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _trim_url(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _TRAILING_URL_PUNCTUATION:
        end -= 1

    while end > start and text[end - 1] in _CLOSING_DELIMITERS:
        closing = text[end - 1]
        opening = _CLOSING_DELIMITERS[closing]
        value = text[start:end]
        if value.count(closing) <= value.count(opening):
            break
        end -= 1

    return end
