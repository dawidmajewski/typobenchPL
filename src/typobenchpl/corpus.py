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

    @property
    def score(self) -> float:
        return self.passed / self.runs * 100

    def as_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 2),
            "runs": self.runs,
            "passed": self.passed,
            "failed": self.failed,
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
    issues: Counter[str] = Counter()

    records = iter(_records(stream))
    while processed < runs:
        try:
            line_number, record = next(records)
        except StopIteration:
            break
        text = record.get("text")
        if not isinstance(text, str):
            raise CorpusError(f"line {line_number}: 'text' must be a string")

        result = analyze(text)
        passed += result.value
        processed += 1
        if result.issue is not None:
            issues[result.issue.rule_id.value] += 1

    if processed < runs:
        raise CorpusError(f"expected {runs} texts, found {processed}")

    return ScoreSummary(
        runs=runs,
        passed=passed,
        failed=runs - passed,
        issues_by_rule=dict(sorted(issues.items())),
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
