from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from typobenchpl.protected import protected_ranges


class RuleId(StrEnum):
    EMPTY_TEXT = "G000"
    INVALID_UTF8 = "G001"
    FORBIDDEN_CONTROL = "G002"
    UNPAIRED_SURROGATE = "G003"

    REPEATED_COMMA = "G010"
    REPEATED_SEMICOLON = "G011"
    INVALID_DOT_RUN = "G012"

    SPACE_BEFORE_COMMA = "G020"
    SPACE_BEFORE_SEMICOLON = "G021"
    SPACE_BEFORE_COLON = "G022"
    SPACE_BEFORE_PERIOD = "G023"
    SPACE_BEFORE_QUESTION = "G024"
    SPACE_BEFORE_EXCLAMATION = "G025"

    MISSING_SPACE_AFTER_COMMA = "G030"
    MISSING_SPACE_AFTER_SEMICOLON = "G031"
    MISSING_SPACE_AFTER_COLON = "G032"
    MISSING_SPACE_AFTER_QUESTION_OR_EXCLAMATION = "G033"

    INVALID_BRACKET_ORDER = "G040"
    UNMATCHED_BRACKET = "G041"


RULE_MESSAGES: dict[RuleId, str] = {
    RuleId.EMPTY_TEXT: "text is empty",
    RuleId.INVALID_UTF8: "input is not valid UTF-8",
    RuleId.FORBIDDEN_CONTROL: "text contains a forbidden control character",
    RuleId.UNPAIRED_SURROGATE: "text contains an unpaired Unicode surrogate",
    RuleId.REPEATED_COMMA: "text contains repeated commas",
    RuleId.REPEATED_SEMICOLON: "text contains repeated semicolons",
    RuleId.INVALID_DOT_RUN: "a dot run must contain one or three dots",
    RuleId.SPACE_BEFORE_COMMA: "whitespace before a comma is not allowed",
    RuleId.SPACE_BEFORE_SEMICOLON: "whitespace before a semicolon is not allowed",
    RuleId.SPACE_BEFORE_COLON: "whitespace before a colon is not allowed",
    RuleId.SPACE_BEFORE_PERIOD: "whitespace before a period is not allowed",
    RuleId.SPACE_BEFORE_QUESTION: "whitespace before a question mark is not allowed",
    RuleId.SPACE_BEFORE_EXCLAMATION: "whitespace before an exclamation mark is not allowed",
    RuleId.MISSING_SPACE_AFTER_COMMA: "a comma between words must be followed by whitespace",
    RuleId.MISSING_SPACE_AFTER_SEMICOLON: (
        "a semicolon between words must be followed by whitespace"
    ),
    RuleId.MISSING_SPACE_AFTER_COLON: "a colon between words must be followed by whitespace",
    RuleId.MISSING_SPACE_AFTER_QUESTION_OR_EXCLAMATION: (
        "a question or exclamation mark between words must be followed by whitespace"
    ),
    RuleId.INVALID_BRACKET_ORDER: "brackets are closed in an invalid order",
    RuleId.UNMATCHED_BRACKET: "a bracket has no matching pair",
}


@dataclass(frozen=True, slots=True)
class Issue:
    rule_id: RuleId
    start: int
    end: int
    offset_unit: Literal["codepoint", "byte"] = "codepoint"

    def as_dict(self) -> dict[str, str | int]:
        return {
            "rule_id": self.rule_id.value,
            "message": RULE_MESSAGES[self.rule_id],
            "start": self.start,
            "end": self.end,
            "offset_unit": self.offset_unit,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    issues: tuple[Issue, ...] = ()

    @property
    def issue(self) -> Issue | None:
        return self.issues[0] if self.issues else None

    @property
    def passed(self) -> bool:
        return self.issue is None

    @property
    def value(self) -> int:
        return int(self.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "result": self.value,
            "status": "PASS" if self.passed else "FAIL",
            "issues": [issue.as_dict() for issue in self.issues],
        }


_OPENING_BRACKETS = "([{"
_CLOSING_BRACKETS = {")": "(", "]": "[", "}": "{"}
_ALLOWED_CONTROLS = "\t\n\r"


def check(value: str | bytes) -> int:
    return analyze(value).value


def analyze(value: str | bytes) -> GateResult:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            return _failed(
                RuleId.INVALID_UTF8,
                error.start,
                error.end,
                offset_unit="byte",
            )
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError("value must be str or bytes")

    if not text or text.isspace():
        return _failed(RuleId.EMPTY_TEXT, 0, len(text))

    ranges = protected_ranges(text)
    range_index = 0
    stack: list[tuple[str, int]] = []
    issues: list[Issue] = []
    index = 0

    while index < len(text):
        while range_index < len(ranges) and ranges[range_index][1] <= index:
            range_index += 1

        character = text[index]
        category = unicodedata.category(character)
        if category == "Cc" and character not in _ALLOWED_CONTROLS:
            issues.append(Issue(RuleId.FORBIDDEN_CONTROL, index, index + 1))
            index += 1
            continue
        if category == "Cs":
            issues.append(Issue(RuleId.UNPAIRED_SURROGATE, index, index + 1))
            index += 1
            continue

        if _is_protected(index, ranges, range_index):
            index += 1
            continue

        if character == ",":
            run_end = _run_end(text, index, character)
            if run_end - index > 1:
                issues.append(Issue(RuleId.REPEATED_COMMA, index, run_end))
                index = run_end
                continue
            issue = _space_before(text, index, RuleId.SPACE_BEFORE_COMMA)
            if issue:
                issues.append(issue)
            if _between_letters_without_space(text, index):
                issues.append(Issue(RuleId.MISSING_SPACE_AFTER_COMMA, index, index + 1))

        elif character == ";":
            run_end = _run_end(text, index, character)
            if run_end - index > 1:
                issues.append(Issue(RuleId.REPEATED_SEMICOLON, index, run_end))
                index = run_end
                continue
            issue = _space_before(text, index, RuleId.SPACE_BEFORE_SEMICOLON)
            if issue:
                issues.append(issue)
            if _between_letters_without_space(text, index):
                issues.append(Issue(RuleId.MISSING_SPACE_AFTER_SEMICOLON, index, index + 1))

        elif character == ".":
            run_end = _run_end(text, index, character)
            run_length = run_end - index
            if run_length not in {1, 3}:
                issues.append(Issue(RuleId.INVALID_DOT_RUN, index, run_end))
            if run_length == 1:
                issue = _space_before(text, index, RuleId.SPACE_BEFORE_PERIOD)
                if issue:
                    issues.append(issue)
            index = run_end - 1

        elif character == ":":
            issue = _space_before(text, index, RuleId.SPACE_BEFORE_COLON)
            if issue:
                issues.append(issue)
            if _between_letters_without_space(text, index):
                issues.append(Issue(RuleId.MISSING_SPACE_AFTER_COLON, index, index + 1))

        elif character == "?":
            issue = _space_before(text, index, RuleId.SPACE_BEFORE_QUESTION)
            if issue:
                issues.append(issue)
            if _ends_punctuation_run_before_letter(text, index):
                issues.append(
                    Issue(
                        RuleId.MISSING_SPACE_AFTER_QUESTION_OR_EXCLAMATION,
                        index,
                        index + 1,
                    )
                )

        elif character == "!":
            issue = _space_before(text, index, RuleId.SPACE_BEFORE_EXCLAMATION)
            if issue:
                issues.append(issue)
            if _ends_punctuation_run_before_letter(text, index):
                issues.append(
                    Issue(
                        RuleId.MISSING_SPACE_AFTER_QUESTION_OR_EXCLAMATION,
                        index,
                        index + 1,
                    )
                )

        elif character in _OPENING_BRACKETS:
            stack.append((character, index))

        elif character in _CLOSING_BRACKETS:
            if not stack:
                issues.append(Issue(RuleId.UNMATCHED_BRACKET, index, index + 1))
            elif stack[-1][0] == _CLOSING_BRACKETS[character]:
                stack.pop()
            else:
                issues.append(Issue(RuleId.INVALID_BRACKET_ORDER, index, index + 1))
                expected = _CLOSING_BRACKETS[character]
                matching_index = next(
                    (
                        position
                        for position in range(len(stack) - 1, -1, -1)
                        if stack[position][0] == expected
                    ),
                    None,
                )
                if matching_index is not None:
                    del stack[matching_index]

        index += 1

    issues.extend(
        Issue(RuleId.UNMATCHED_BRACKET, opening_index, opening_index + 1)
        for _, opening_index in stack
    )

    return GateResult(tuple(sorted(issues, key=lambda issue: (issue.start, issue.end))))


def _failed(
    rule_id: RuleId,
    start: int,
    end: int,
    offset_unit: Literal["codepoint", "byte"] = "codepoint",
) -> GateResult:
    return GateResult((Issue(rule_id, start, end, offset_unit),))


def _is_protected(index: int, ranges: tuple[tuple[int, int], ...], range_index: int) -> bool:
    if range_index >= len(ranges):
        return False
    start, end = ranges[range_index]
    return start <= index < end


def _run_end(text: str, start: int, character: str) -> int:
    end = start + 1
    while end < len(text) and text[end] == character:
        end += 1
    return end


def _space_before(text: str, index: int, rule_id: RuleId) -> Issue | None:
    if index == 0 or not text[index - 1].isspace():
        return None

    start = index - 1
    while start > 0 and text[start - 1].isspace():
        start -= 1
    return Issue(rule_id, start, index + 1)


def _between_letters_without_space(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1].isalpha()
        and text[index + 1].isalpha()
    )


def _ends_punctuation_run_before_letter(text: str, index: int) -> bool:
    if index + 1 >= len(text) or not text[index + 1].isalpha():
        return False
    if index == 0:
        return False
    punctuation_context = "?!)]}\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}\"'"
    return text[index - 1].isalpha() or text[index - 1] in punctuation_context
