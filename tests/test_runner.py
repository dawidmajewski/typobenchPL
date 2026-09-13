from __future__ import annotations

import json

import pytest

from typobenchpl.corpus import score_stream
from typobenchpl.hf import GeneratedText
from typobenchpl.runner import RunnerError, run_benchmark
from typobenchpl.suite import GenerationSettings


class FakeGenerator:
    def __init__(self, completions: tuple[str | bytes, ...] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.completions = iter(completions or (" a.", " b.", ",, x.", "", " ok."))

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "fake",
            "model": "fake-model",
            "device": "cpu",
        }

    def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText:
        self.calls.append((prompt, seed))
        completion = next(self.completions)
        return GeneratedText(completion, prompt_tokens=3, generated_tokens=2 if completion else 0)


def write_suite(tmp_path, prompt_count: int = 2):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "suite.toml").write_text(
        """name = "test-suite"
version = "1.0.0"
description = "Test suite"
mode = "completion"
evaluation_scope = "prompt_and_completion"

[generation]
max_new_tokens = 80
do_sample = true
temperature = 0.7
top_k = 40
repetition_penalty = 1.3
seed = 42
prepend_bos = "auto"
""",
        encoding="utf-8",
    )
    prompts = [
        {"id": f"prompt-{index + 1}", "prompt": f"Prompt {index + 1}", "category": "test"}
        for index in range(prompt_count)
    ]
    (suite / "prompts.jsonl").write_text(
        "\n".join(json.dumps(prompt, ensure_ascii=False) for prompt in prompts) + "\n",
        encoding="utf-8",
    )
    return suite


def test_run_writes_reproducible_artifacts(tmp_path) -> None:
    generator = FakeGenerator()
    output_directory = tmp_path / "run"
    suite = write_suite(tmp_path)

    summary = run_benchmark(
        model_reference="fake-model",
        suite_reference=suite,
        chars_per_prompt=6,
        max_attempts_per_prompt=3,
        output_directory=output_directory,
        generator=generator,
    )

    assert summary.coverage_complete
    assert summary.score.clean_output_rate == 60
    assert summary.score.passed == 3
    assert summary.score.issues_by_rule == {"G000": 1, "G010": 1}
    assert summary.score.generated_chars == 15
    assert summary.score.evaluated_chars == 15
    assert summary.score.total_issues == 2
    assert summary.score.issues_per_10k_chars == pytest.approx(1333.333)
    assert summary.score.truncated_outputs == 1
    assert summary.prompt_coverage == {"prompt-1": 6, "prompt-2": 9}
    assert [seed for _, seed in generator.calls] == [42, 44, 43, 45, 47]

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    persisted_summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output_directory / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["status"] == "completed"
    assert manifest["suite"]["digest"]
    assert persisted_summary["issues_per_10k_chars"] == 1333.333
    assert persisted_summary["attempts"] == 5
    assert persisted_summary["target_chars"] == 12
    assert records[0]["result"] == 1
    assert records[2]["issues"][0]["rule_id"] == "G010"
    assert records[3]["text"] == ""
    assert records[3]["full_text"] == records[3]["prompt"]

    with (output_directory / "outputs.jsonl").open("r", encoding="utf-8") as stream:
        rescored = score_stream(stream, runs=5)
    assert rescored.as_dict() == summary.score.as_dict()


def test_run_refuses_to_overwrite_output_directory(tmp_path) -> None:
    with pytest.raises(RunnerError, match="already exists"):
        run_benchmark(
            model_reference="fake-model",
            output_directory=tmp_path,
            generator=FakeGenerator(),
        )


def test_failed_run_updates_manifest(tmp_path) -> None:
    class FailingGenerator(FakeGenerator):
        def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText:
            raise RuntimeError("generation failed")

    output_directory = tmp_path / "failed"
    with pytest.raises(RunnerError, match="generation 1"):
        run_benchmark(
            model_reference="fake-model",
            chars_per_prompt=1,
            output_directory=output_directory,
            generator=FailingGenerator(),
        )

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RunnerError"


def test_interrupted_run_updates_manifest(tmp_path) -> None:
    class InterruptedGenerator(FakeGenerator):
        def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText:
            raise KeyboardInterrupt

    output_directory = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(
            model_reference="fake-model",
            chars_per_prompt=1,
            output_directory=output_directory,
            generator=InterruptedGenerator(),
        )

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["error"]["type"] == "KeyboardInterrupt"


def test_invalid_utf8_does_not_contribute_to_coverage(tmp_path) -> None:
    generator = FakeGenerator((b"\xff", b" ok."))
    output_directory = tmp_path / "run"

    summary = run_benchmark(
        model_reference="fake-model",
        suite_reference=write_suite(tmp_path, prompt_count=1),
        chars_per_prompt=4,
        max_attempts_per_prompt=2,
        output_directory=output_directory,
        generator=generator,
    )

    records = [
        json.loads(line)
        for line in (output_directory / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary.score.invalid_outputs == 1
    assert summary.score.generated_chars == 4
    assert summary.score.evaluated_chars == 4
    assert summary.score.issues_by_rule == {"G001": 1}
    assert records[0]["completion"] is None
    assert records[0]["completion_bytes_hex"] == "ff"
    assert "�" not in json.dumps(records[0], ensure_ascii=False)
    with (output_directory / "outputs.jsonl").open("r", encoding="utf-8") as stream:
        assert score_stream(stream, runs=2).as_dict() == summary.score.as_dict()


def test_reports_incomplete_coverage_after_attempt_limit(tmp_path) -> None:
    summary = run_benchmark(
        model_reference="fake-model",
        suite_reference=write_suite(tmp_path, prompt_count=1),
        chars_per_prompt=1,
        max_attempts_per_prompt=2,
        output_directory=tmp_path / "run",
        generator=FakeGenerator(("", "")),
    )

    assert not summary.coverage_complete
    assert summary.prompt_coverage == {"prompt-1": 0}
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"


def test_hard_limit_excludes_incomplete_final_sentence(tmp_path) -> None:
    completion = " Pełne zdanie. (urwany fragment"
    summary = run_benchmark(
        model_reference="fake-model",
        suite_reference=write_suite(tmp_path, prompt_count=1),
        chars_per_prompt=14,
        max_attempts_per_prompt=1,
        output_directory=tmp_path / "run",
        generator=FakeGenerator((completion,)),
    )

    record = json.loads((tmp_path / "run" / "outputs.jsonl").read_text(encoding="utf-8"))
    assert summary.coverage_complete
    assert summary.score.total_issues == 0
    assert summary.score.truncated_outputs == 1
    assert summary.score.generated_chars == len(completion)
    assert summary.score.evaluated_chars == 14
    assert record["text"] == "Prompt 1 Pełne zdanie."
    assert record["evaluated_chars"] == 14
    assert record["truncated"] is True
