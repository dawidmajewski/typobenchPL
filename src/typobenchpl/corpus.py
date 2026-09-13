from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import IO, Any

from typobenchpl.gate import analyze


class CorpusError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    runs: int
    passed: int
    failed: int
    issues_by_rule: dict[str, int]
    generated_chars: int | None = None
    evaluated_chars: int = 0
    total_issues: int = 0
    invalid_outputs: int = 0
    truncated_outputs: int = 0

    @property
    def clean_output_rate(self) -> float:
        return self.passed / self.runs * 100

    @property
    def issues_per_10k_chars(self) -> float | None:
        if self.evaluated_chars == 0:
            return None
        return self.total_issues / self.evaluated_chars * 10_000

    def as_dict(self) -> dict[str, object]:
        generated_chars = (
            self.evaluated_chars if self.generated_chars is None else self.generated_chars
        )
        return {
            "clean_output_rate": round(self.clean_output_rate, 2),
            "runs": self.runs,
            "passed": self.passed,
            "failed": self.failed,
            "generated_chars": generated_chars,
            "evaluated_chars": self.evaluated_chars,
            "discarded_chars": generated_chars - self.evaluated_chars,
            "total_issues": self.total_issues,
            "issues_per_10k_chars": (
                None if self.issues_per_10k_chars is None else round(self.issues_per_10k_chars, 3)
            ),
            "invalid_outputs": self.invalid_outputs,
            "truncated_outputs": self.truncated_outputs,
            "issues_by_rule": self.issues_by_rule,
        }


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    total: int
    passed: int
    failures: tuple[str, ...]

    @property
    def successful(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, object]:
        return {
            "successful": self.successful,
            "total": self.total,
            "passed": self.passed,
            "failed": len(self.failures),
            "failures": list(self.failures),
        }


def score_stream(stream: IO[str], runs: int = 1000) -> ScoreSummary:
    if runs <= 0:
        raise CorpusError("runs must be greater than zero")

    passed = 0
    processed = 0
    generated_chars = 0
    evaluated_chars = 0
    total_issues = 0
    invalid_outputs = 0
    truncated_outputs = 0
    issues: Counter[str] = Counter()

    records = iter(_records(stream))
    while processed < runs:
        try:
            line_number, record = next(records)
        except StopIteration:
            break
        text: str | bytes
        text_bytes_hex = record.get("text_bytes_hex")
        if text_bytes_hex is not None:
            if not isinstance(text_bytes_hex, str):
                raise CorpusError(f"line {line_number}: 'text_bytes_hex' must be a string")
            try:
                text = bytes.fromhex(text_bytes_hex)
            except ValueError as error:
                raise CorpusError(f"line {line_number}: invalid 'text_bytes_hex'") from error
        else:
            text = record.get("text")
            if not isinstance(text, str):
                raise CorpusError(f"line {line_number}: 'text' must be a string")

        result = analyze(text)
        passed += result.value
        processed += 1
        record_chars = record.get("evaluated_chars", len(text) if isinstance(text, str) else 0)
        if not isinstance(record_chars, int) or isinstance(record_chars, bool) or record_chars < 0:
            raise CorpusError(
                f"line {line_number}: 'evaluated_chars' must be a non-negative integer"
            )
        record_generated_chars = record.get("generated_chars", record_chars)
        if (
            not isinstance(record_generated_chars, int)
            or isinstance(record_generated_chars, bool)
            or record_generated_chars < record_chars
        ):
            raise CorpusError(
                f"line {line_number}: 'generated_chars' must be an integer >= evaluated_chars"
            )
        generated_chars += record_generated_chars
        evaluated_chars += record_chars
        total_issues += len(result.issues)
        invalid_outputs += any(issue.rule_id.value == "G001" for issue in result.issues)
        truncated_outputs += record.get("truncated") is True
        for issue in result.issues:
            issues[issue.rule_id.value] += 1

    if processed < runs:
        raise CorpusError(f"expected {runs} texts, found {processed}")

    return ScoreSummary(
        runs=runs,
        passed=passed,
        failed=runs - passed,
        issues_by_rule=dict(sorted(issues.items())),
        generated_chars=generated_chars,
        evaluated_chars=evaluated_chars,
        total_issues=total_issues,
        invalid_outputs=invalid_outputs,
        truncated_outputs=truncated_outputs,
    )


def verify_cases(case_directory: Path | None = None) -> VerificationSummary:
    sources: Iterable[tuple[str, IO[str]]]
    opened_streams: list[IO[str]] = []

    if case_directory is None:
        package_cases = files("typobenchpl").joinpath("cases")
        sources = [
            (name, package_cases.joinpath(name).open("r", encoding="utf-8"))
            for name in ("must_pass.jsonl", "must_fail.jsonl")
        ]
    else:
        sources = []
        for name in ("must_pass.jsonl", "must_fail.jsonl"):
            path = case_directory / name
            if not path.is_file():
                raise CorpusError(f"missing case file: {path}")
            stream = path.open("r", encoding="utf-8")
            opened_streams.append(stream)
            sources.append((name, stream))

    failures: list[str] = []
    seen_ids: set[str] = set()
    total = 0

    try:
        for source_name, stream in sources:
            for line_number, record in _records(stream):
                total += 1
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    raise CorpusError(f"{source_name}:{line_number}: invalid or missing 'id'")
                if case_id in seen_ids:
                    raise CorpusError(f"duplicate case id: {case_id}")
                seen_ids.add(case_id)

                expected = record.get("expected")
                if expected not in {0, 1} or isinstance(expected, bool):
                    raise CorpusError(f"{source_name}:{line_number}: 'expected' must be 0 or 1")

                value = _case_value(record, source_name, line_number)
                result = analyze(value)
                actual_rule = result.issue.rule_id.value if result.issue else None
                expected_rule = record.get("rule_id")
                expected_start = record.get("start")
                expected_end = record.get("end")

                reasons: list[str] = []
                if result.value != expected:
                    reasons.append(f"expected {expected}, got {result.value}")
                if expected_rule is not None and actual_rule != expected_rule:
                    reasons.append(f"expected rule {expected_rule}, got {actual_rule}")
                if expected_start is not None and (
                    result.issue is None or result.issue.start != expected_start
                ):
                    actual_start = result.issue.start if result.issue else None
                    reasons.append(f"expected start {expected_start}, got {actual_start}")
                if expected_end is not None and (
                    result.issue is None or result.issue.end != expected_end
                ):
                    actual_end = result.issue.end if result.issue else None
                    reasons.append(f"expected end {expected_end}, got {actual_end}")

                if reasons:
                    failures.append(f"{case_id}: {'; '.join(reasons)}")
    finally:
        for _, stream in sources:
            stream.close()
        for stream in opened_streams:
            if not stream.closed:
                stream.close()

    return VerificationSummary(total, total - len(failures), tuple(failures))


def _records(stream: IO[str]) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_number, raw_line in enumerate(stream, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusError(f"line {line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(record, dict):
            raise CorpusError(f"line {line_number}: each record must be an object")
        yield line_number, record


def _case_value(record: dict[str, Any], source_name: str, line_number: int) -> str | bytes:
    has_text = "text" in record
    has_bytes = "bytes_hex" in record
    if has_text == has_bytes:
        raise CorpusError(
            f"{source_name}:{line_number}: provide exactly one of 'text' or 'bytes_hex'"
        )

    if has_text:
        text = record["text"]
        if not isinstance(text, str):
            raise CorpusError(f"{source_name}:{line_number}: 'text' must be a string")
        return text

    bytes_hex = record["bytes_hex"]
    if not isinstance(bytes_hex, str):
        raise CorpusError(f"{source_name}:{line_number}: 'bytes_hex' must be a string")
    try:
        return bytes.fromhex(bytes_hex)
    except ValueError as error:
        raise CorpusError(f"{source_name}:{line_number}: invalid hexadecimal input") from error
