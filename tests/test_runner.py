from __future__ import annotations

import json

import pytest

from typobenchpl.corpus import score_stream
from typobenchpl.hf import GeneratedText
from typobenchpl.runner import RunnerError, run_benchmark
from typobenchpl.suite import GenerationSettings


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.completions = iter((" ciąg dalszy.", ",, błąd.", ""))

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


def test_run_writes_reproducible_artifacts(tmp_path) -> None:
    generator = FakeGenerator()
    output_directory = tmp_path / "run"

    summary = run_benchmark(
        model_reference="fake-model",
        runs=3,
        output_directory=output_directory,
        generator=generator,
    )

    assert summary.score.score == pytest.approx(100 / 3)
    assert summary.score.passed == 1
    assert summary.score.issues_by_rule == {"G000": 1, "G010": 1}
    assert [seed for _, seed in generator.calls] == [42, 43, 44]

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    persisted_summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (output_directory / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["status"] == "completed"
    assert manifest["suite"]["digest"]
    assert persisted_summary["score"] == 33.33
    assert records[0]["result"] == 1
    assert records[1]["issue"]["rule_id"] == "G010"
    assert records[2]["text"] == ""
    assert records[2]["full_text"] == records[2]["prompt"]

    with (output_directory / "outputs.jsonl").open("r", encoding="utf-8") as stream:
        rescored = score_stream(stream, runs=3)
    assert rescored.as_dict() == summary.score.as_dict()


def test_run_refuses_to_overwrite_output_directory(tmp_path) -> None:
    with pytest.raises(RunnerError, match="already exists"):
        run_benchmark(
            model_reference="fake-model",
            runs=1,
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
            runs=1,
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
            runs=1,
            output_directory=output_directory,
            generator=InterruptedGenerator(),
        )

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["error"]["type"] == "KeyboardInterrupt"
