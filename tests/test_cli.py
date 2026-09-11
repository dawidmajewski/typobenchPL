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


def test_score_prints_percentage(tmp_path, capsys) -> None:
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
    assert capsys.readouterr().out == "50.00\n"


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
            ScoreSummary(runs=4, passed=3, failed=1, issues_by_rule={"G010": 1}),
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
            "-n",
            "4",
            "--device",
            "cpu",
            "--output-dir",
            str(output_directory),
        ]
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert output.out == "75.00\n"
    assert output.err == (
        "progress: 1/4 (25.0%) elapsed 00:00:00 eta 00:00:00\n"
        "progress: 4/4 (100.0%) elapsed 00:00:01 eta 00:00:00\n"
        f"results: {output_directory}\n"
    )
    assert callable(captured.pop("progress"))
    assert captured == {
        "model_reference": "models/example",
        "suite_reference": "polish-prose-v1",
        "runs": 4,
        "output_directory": output_directory,
        "revision": None,
        "device": "cpu",
    }
