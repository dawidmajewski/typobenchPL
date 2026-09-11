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

    def as_dict(self) -> dict[str, object]:
        return {
            **self.score.as_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
            "output_directory": str(self.output_directory),
        }


def run_benchmark(
    *,
    model_reference: str,
    suite_reference: str | Path = "polish-prose-v1",
    runs: int = 1000,
    output_directory: Path | None = None,
    revision: str | None = None,
    device: str = "auto",
    generator: TextGenerator | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> RunSummary:
    if runs <= 0:
        raise RunnerError("runs must be greater than zero")

    suite = load_suite(suite_reference)
    if generator is None:
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
        "runs": runs,
        "model": generator.metadata,
        "suite": suite.as_dict(),
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)

    passed = 0
    issues: Counter[str] = Counter()
    started = time.perf_counter()

    try:
        with (destination / "outputs.jsonl").open("w", encoding="utf-8") as output_stream:
            for run_index in range(runs):
                prompt = suite.prompts[run_index % len(suite.prompts)]
                seed = suite.settings.seed + run_index
                generation_started = time.perf_counter()
                try:
                    generated = generator.generate(prompt.text, suite.settings, seed)
                except Exception as error:
                    raise RunnerError(
                        f"generation {run_index + 1} ({prompt.id}) failed: {error}"
                    ) from error

                full_text = prompt.text + generated.completion
                evaluated_text = full_text if generated.completion.strip() else ""
                result = analyze(evaluated_text)
                passed += result.value
                if result.issue is not None:
                    issues[result.issue.rule_id.value] += 1

                record = {
                    "id": f"run-{run_index + 1:06d}",
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "prompt": prompt.text,
                    "completion": generated.completion,
                    "full_text": full_text,
                    "text": evaluated_text,
                    "seed": seed,
                    "prompt_tokens": generated.prompt_tokens,
                    "generated_tokens": generated.generated_tokens,
                    "generation_seconds": round(time.perf_counter() - generation_started, 6),
                    "result": result.value,
                    "issue": result.issue.as_dict() if result.issue else None,
                }
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_stream.flush()
                if progress is not None:
                    progress(run_index + 1, runs, time.perf_counter() - started)

        duration = time.perf_counter() - started
        score = ScoreSummary(
            runs=runs,
            passed=passed,
            failed=runs - passed,
            issues_by_rule=dict(sorted(issues.items())),
        )
        summary = RunSummary(score, destination, duration)
        _write_json(destination / "summary.json", summary.as_dict())

        manifest.update(
            {
                "status": "completed",
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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
