from __future__ import annotations

import json

from typobenchpl.cli import main
from typobenchpl.corpus import ScoreSummary
from typobenchpl.runner import RunSummary


def test_check_prints_one_for_pass(capsys) -> None:
    exit_code = main(["check", "--text", "To jest poprawne."])

    assert exit_code == 0
    assert capsys.readouterr().out == "1\n"


def test_check_prints_zero_for_fail(capsys) -> None:
    exit_code = main(["check", "--text", "To jest,, błędne."])

    assert exit_code == 1
    assert capsys.readouterr().out == "0\n"


def test_score_prints_issues_per_10k_chars(tmp_path, capsys) -> None:
    path = tmp_path / "outputs.jsonl"
    records = [
        {"text": "Poprawnie."},
        {"text": "Źle,, tutaj."},
        {"text": "Również poprawnie."},
        {"text": "Źle;; tutaj."},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    exit_code = main(["score", str(path), "-n", "4"])

    assert exit_code == 0
    assert capsys.readouterr().out == "384.615\n"


def test_verify_prints_summary(capsys) -> None:
    exit_code = main(["verify"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert "tests passed" in output.out
    assert output.err == ""


def test_run_prints_score_and_artifact_path(tmp_path, monkeypatch, capsys) -> None:
    output_directory = tmp_path / "run"
    captured: dict[str, object] = {}

    def fake_run_benchmark(**options):
        captured.update(options)
        progress = options["progress"]
        progress(1, 4, 0.25)
        progress(4, 4, 1.25)
        return RunSummary(
            ScoreSummary(
                runs=4,
                passed=3,
                failed=1,
                issues_by_rule={"G010": 1},
                evaluated_chars=1000,
                total_issues=1,
            ),
            output_directory,
            1.25,
        )

    monkeypatch.setattr("typobenchpl.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(
        [
            "run",
            "--model",
            "models/example",
            "--suite",
            "polish-prose-v1",
            "--chars-per-prompt",
            "4",
            "--max-attempts-per-prompt",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_directory),
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert output.out == "10.000\n"
    assert output.err == (
        "coverage: 1/4 chars (25.0%) elapsed 00:00:00 eta 00:00:00\n"
        "coverage: 4/4 chars (100.0%) elapsed 00:00:01 eta 00:00:00\n"
        f"results: {output_directory}\n"
    )
    assert callable(captured.pop("progress"))
    assert captured == {
        "model_reference": "models/example",
        "suite_reference": "polish-prose-v1",
        "chars_per_prompt": 4,
        "max_attempts_per_prompt": 2,
        "output_directory": output_directory,
        "revision": None,
        "device": "cpu",
    }


def test_run_handles_keyboard_interrupt(monkeypatch, capsys) -> None:
    def interrupt(**options):
        raise KeyboardInterrupt

    monkeypatch.setattr("typobenchpl.cli.run_benchmark", interrupt)

    exit_code = main(["run", "--model", "models/example"])

    output = capsys.readouterr()
    assert exit_code == 130
    assert output.out == ""
    assert output.err == "interrupted\n"


def test_run_reports_incomplete_character_coverage(tmp_path, monkeypatch, capsys) -> None:
    output_directory = tmp_path / "run"

    def fake_run_benchmark(**options):
        return RunSummary(
            ScoreSummary(runs=1, passed=0, failed=1, issues_by_rule={"G000": 1}),
            output_directory,
            1.0,
            chars_per_prompt=10,
            max_attempts_per_prompt=1,
            prompt_coverage={"prompt-1": 0},
        )

    monkeypatch.setattr("typobenchpl.cli.run_benchmark", fake_run_benchmark)

    exit_code = main(["run", "--model", "models/example"])

    output = capsys.readouterr()
    assert exit_code == 2
    assert output.out == "n/a\n"
    assert output.err == (
        f"results: {output_directory}\nincomplete: 1 prompt did not reach the character quota\n"
    )
