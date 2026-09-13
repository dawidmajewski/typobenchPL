from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from typobenchpl.byte_transformer import ByteTransformerGenerator, is_byte_transformer_model
from typobenchpl.corpus import ScoreSummary
from typobenchpl.gate import analyze
from typobenchpl.hf import GeneratedText, HuggingFaceCausalGenerator
from typobenchpl.suite import BenchmarkSuite, GenerationSettings, load_suite


class RunnerError(ValueError):
    pass


class TextGenerator(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText: ...


@dataclass(frozen=True, slots=True)
class RunSummary:
    score: ScoreSummary
    output_directory: Path
    duration_seconds: float
    chars_per_prompt: int = 0
    max_attempts_per_prompt: int = 0
    prompt_coverage: dict[str, int] | None = None

    @property
    def coverage_complete(self) -> bool:
        return self.prompt_coverage is None or all(
            characters >= self.chars_per_prompt for characters in self.prompt_coverage.values()
        )

    def as_dict(self) -> dict[str, object]:
        result = self.score.as_dict()
        result["attempts"] = result.pop("runs")
        result["passed_outputs"] = result.pop("passed")
        result["failed_outputs"] = result.pop("failed")
        return {
            **result,
            "chars_per_prompt": self.chars_per_prompt,
            "max_attempts_per_prompt": self.max_attempts_per_prompt,
            "target_chars": self.chars_per_prompt * len(self.prompt_coverage or {}),
            "coverage_complete": self.coverage_complete,
            "prompt_coverage": self.prompt_coverage or {},
            "duration_seconds": round(self.duration_seconds, 3),
            "output_directory": str(self.output_directory),
        }


def run_benchmark(
    *,
    model_reference: str,
    suite_reference: str | Path = "polish-prose-v1",
    chars_per_prompt: int = 3000,
    max_attempts_per_prompt: int = 100,
    output_directory: Path | None = None,
    revision: str | None = None,
    device: str = "auto",
    generator: TextGenerator | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> RunSummary:
    if chars_per_prompt <= 0:
        raise RunnerError("chars_per_prompt must be greater than zero")
    if max_attempts_per_prompt <= 0:
        raise RunnerError("max_attempts_per_prompt must be greater than zero")

    suite = load_suite(suite_reference)
    invalid_prompts = [prompt.id for prompt in suite.prompts if not analyze(prompt.text).passed]
    if invalid_prompts:
        raise RunnerError(f"suite contains invalid prompts: {', '.join(invalid_prompts)}")
    if generator is None:
        if is_byte_transformer_model(model_reference):
            if revision is not None:
                raise RunnerError("--revision is not supported for local byte-transformer models")
            generator = ByteTransformerGenerator(model_reference, device=device)
        else:
            generator = HuggingFaceCausalGenerator(
                model_reference,
                revision=revision,
                device=device,
            )

    destination = output_directory or _default_output_directory(model_reference, suite)
    if destination.exists():
        raise RunnerError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)

    started_at = datetime.now(UTC)
    manifest = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "chars_per_prompt": chars_per_prompt,
        "max_attempts_per_prompt": max_attempts_per_prompt,
        "model": generator.metadata,
        "suite": suite.as_dict(),
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)

    passed = 0
    attempts = 0
    generated_chars = 0
    evaluated_chars = 0
    invalid_outputs = 0
    truncated_outputs = 0
    issues: Counter[str] = Counter()
    prompt_coverage = {prompt.id: 0 for prompt in suite.prompts}
    target_chars = chars_per_prompt * len(suite.prompts)
    credited_chars = 0
    started = time.perf_counter()

    try:
        with (destination / "outputs.jsonl").open("w", encoding="utf-8") as output_stream:
            for prompt_index, prompt in enumerate(suite.prompts):
                for prompt_attempt in range(max_attempts_per_prompt):
                    if prompt_coverage[prompt.id] >= chars_per_prompt:
                        break

                    attempts += 1
                    seed = suite.settings.seed + prompt_index + prompt_attempt * len(suite.prompts)
                    generation_started = time.perf_counter()
                    try:
                        generated = generator.generate(prompt.text, suite.settings, seed)
                    except Exception as error:
                        raise RunnerError(
                            f"generation {attempts} ({prompt.id}, attempt {prompt_attempt + 1}) "
                            f"failed: {error}"
                        ) from error

                    raw_completion = generated.completion
                    completion: str | None
                    full_text: str | None
                    evaluated_text: str | None
                    text_bytes_hex: str | None = None
                    completion_bytes_hex: str | None = None
                    completion_chars = 0
                    truncated = False

                    if isinstance(raw_completion, bytes):
                        full_bytes = prompt.text.encode("utf-8") + raw_completion
                        try:
                            completion = raw_completion.decode("utf-8", errors="strict")
                        except UnicodeDecodeError:
                            completion = None
                            full_text = None
                            evaluated_text = None
                            completion_bytes_hex = raw_completion.hex()
                            text_bytes_hex = full_bytes.hex()
                            result = analyze(full_bytes)
                        else:
                            full_text = prompt.text + completion
                            evaluated_text, completion_chars, truncated = _evaluation_span(
                                prompt.text,
                                completion,
                                generated.finish_reason,
                            )
                            result = analyze(evaluated_text)
                    else:
                        completion = raw_completion
                        full_text = prompt.text + completion
                        evaluated_text, completion_chars, truncated = _evaluation_span(
                            prompt.text,
                            completion,
                            generated.finish_reason,
                        )
                        result = analyze(evaluated_text)

                    passed += result.value
                    generated_chars += len(completion) if completion is not None else 0
                    evaluated_chars += completion_chars
                    invalid_outputs += any(issue.rule_id.value == "G001" for issue in result.issues)
                    truncated_outputs += truncated
                    for issue in result.issues:
                        issues[issue.rule_id.value] += 1

                    previous_coverage = prompt_coverage[prompt.id]
                    prompt_coverage[prompt.id] += completion_chars
                    credited_chars += min(prompt_coverage[prompt.id], chars_per_prompt) - min(
                        previous_coverage,
                        chars_per_prompt,
                    )

                    record = {
                        "id": f"run-{attempts:06d}",
                        "prompt_id": prompt.id,
                        "prompt_attempt": prompt_attempt + 1,
                        "category": prompt.category,
                        "prompt": prompt.text,
                        "completion": completion,
                        "completion_bytes_hex": completion_bytes_hex,
                        "full_text": full_text,
                        "text": evaluated_text,
                        "text_bytes_hex": text_bytes_hex,
                        "generated_chars": len(completion) if completion is not None else 0,
                        "evaluated_chars": completion_chars,
                        "seed": seed,
                        "prompt_tokens": generated.prompt_tokens,
                        "generated_tokens": generated.generated_tokens,
                        "finish_reason": generated.finish_reason,
                        "truncated": truncated,
                        "generation_seconds": round(time.perf_counter() - generation_started, 6),
                        "result": result.value,
                        "issue": result.issue.as_dict() if result.issue else None,
                        "issues": [issue.as_dict() for issue in result.issues],
                    }
                    output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_stream.flush()
                    if progress is not None and completion_chars:
                        progress(credited_chars, target_chars, time.perf_counter() - started)

        duration = time.perf_counter() - started
        score = ScoreSummary(
            runs=attempts,
            passed=passed,
            failed=attempts - passed,
            issues_by_rule=dict(sorted(issues.items())),
            generated_chars=generated_chars,
            evaluated_chars=evaluated_chars,
            total_issues=sum(issues.values()),
            invalid_outputs=invalid_outputs,
            truncated_outputs=truncated_outputs,
        )
        summary = RunSummary(
            score,
            destination,
            duration,
            chars_per_prompt,
            max_attempts_per_prompt,
            prompt_coverage,
        )
        _write_json(destination / "summary.json", summary.as_dict())

        manifest.update(
            {
                "status": "completed" if summary.coverage_complete else "incomplete",
                "finished_at": datetime.now(UTC).isoformat(),
                "summary": summary.as_dict(),
            }
        )
        _write_json(manifest_path, manifest)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _default_output_directory(model_reference: str, suite: BenchmarkSuite) -> Path:
    model_name = Path(model_reference).name or model_reference
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("runs") / model_slug / f"{suite.name}-{timestamp}"


def _evaluation_span(
    prompt: str,
    completion: str,
    finish_reason: str,
) -> tuple[str, int, bool]:
    if not completion.strip():
        return "", 0, finish_reason == "max_tokens"

    full_text = prompt + completion
    if finish_reason != "max_tokens":
        return full_text, len(completion), False

    boundary = _last_sentence_boundary(full_text)
    if boundary <= len(prompt):
        return "", 0, True
    if not full_text[boundary:].strip():
        boundary = len(full_text)
    return full_text[:boundary], boundary - len(prompt), boundary < len(full_text)


def _last_sentence_boundary(text: str) -> int:
    closing_characters = "\"')]}\N{RIGHT DOUBLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}»"
    boundary = 0
    for index, character in enumerate(text):
        if character not in ".?!\N{HORIZONTAL ELLIPSIS}":
            continue
        end = index + 1
        while end < len(text) and text[end] in closing_characters:
            end += 1
        if end == len(text) or text[end].isspace():
            boundary = end
    return boundary


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
