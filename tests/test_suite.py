from __future__ import annotations

import json

import pytest

from typobenchpl import analyze
from typobenchpl.suite import SuiteError, load_suite


def test_loads_builtin_suite() -> None:
    suite = load_suite("polish-prose-v1")

    assert suite.name == "polish-prose-v1"
    assert suite.version == "1.0.0"
    assert len(suite.prompts) == 100
    assert suite.settings.max_new_tokens == 80
    assert suite.settings.seed == 42
    assert suite.settings.prepend_bos == "auto"
    assert len(suite.digest) == 64
    assert all(analyze(prompt.text).passed for prompt in suite.prompts)


def test_loads_suite_from_directory(tmp_path) -> None:
    (tmp_path / "suite.toml").write_text(
        """
name = "custom"
version = "1"
description = "Custom suite"
mode = "completion"
evaluation_scope = "prompt_and_completion"

[generation]
max_new_tokens = 10
do_sample = false
temperature = 1.0
top_k = 0
repetition_penalty = 1.0
seed = 7
prepend_bos = "never"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"id": "one", "prompt": "Początek"}),
        encoding="utf-8",
    )

    suite = load_suite(tmp_path)

    assert suite.name == "custom"
    assert suite.prompts[0].text == "Początek"
    assert suite.settings.do_sample is False


def test_rejects_duplicate_prompt_ids(tmp_path) -> None:
    (tmp_path / "suite.toml").write_text(
        """
name = "custom"
version = "1"
description = "Custom suite"
mode = "completion"
evaluation_scope = "prompt_and_completion"

[generation]
max_new_tokens = 10
do_sample = true
temperature = 0.7
top_k = 40
repetition_penalty = 1.2
seed = 1
prepend_bos = "auto"
""".strip(),
        encoding="utf-8",
    )
    duplicate = json.dumps({"id": "same", "prompt": "Początek"})
    (tmp_path / "prompts.jsonl").write_text(
        f"{duplicate}\n{duplicate}\n",
        encoding="utf-8",
    )

    with pytest.raises(SuiteError, match="duplicate prompt id"):
        load_suite(tmp_path)
