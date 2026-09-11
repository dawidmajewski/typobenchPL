from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal


class SuiteError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_k: int
    repetition_penalty: float
    seed: int
    prepend_bos: Literal["auto", "always", "never"]

    def as_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "prepend_bos": self.prepend_bos,
        }


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    text: str
    category: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    name: str
    version: str
    description: str
    mode: Literal["completion"]
    evaluation_scope: Literal["prompt_and_completion"]
    settings: GenerationSettings
    prompts: tuple[Prompt, ...]
    digest: str
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "mode": self.mode,
            "evaluation_scope": self.evaluation_scope,
            "prompt_count": len(self.prompts),
            "digest": self.digest,
            "source": self.source,
            "generation": self.settings.as_dict(),
        }


def load_suite(reference: str | Path) -> BenchmarkSuite:
    path = Path(reference)
    if path.is_dir():
        root: Traversable = path
        source = str(path.resolve())
    elif path.exists():
        raise SuiteError(f"suite path is not a directory: {path}")
    else:
        root = files("typobenchpl").joinpath("suites", str(reference))
        source = f"builtin:{reference}"

    config_path = root.joinpath("suite.toml")
    prompts_path = root.joinpath("prompts.jsonl")
    if not config_path.is_file() or not prompts_path.is_file():
        raise SuiteError(f"unknown or incomplete suite: {reference}")

    config_bytes = config_path.read_bytes()
    prompts_bytes = prompts_path.read_bytes()
    try:
        config = tomllib.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SuiteError(f"invalid suite.toml in {reference}: {error}") from error

    prompts = _load_prompts(prompts_bytes, reference)
    settings = _load_settings(config)
    name = _required_string(config, "name")
    version = _required_string(config, "version")
    description = _required_string(config, "description")
    mode = config.get("mode")
    if mode != "completion":
        raise SuiteError("suite 'mode' must be 'completion'")
    evaluation_scope = config.get("evaluation_scope")
    if evaluation_scope != "prompt_and_completion":
        raise SuiteError("suite 'evaluation_scope' must be 'prompt_and_completion'")

    digest = hashlib.sha256(config_bytes + b"\0" + prompts_bytes).hexdigest()
    return BenchmarkSuite(
        name=name,
        version=version,
        description=description,
        mode=mode,
        evaluation_scope=evaluation_scope,
        settings=settings,
        prompts=prompts,
        digest=digest,
        source=source,
    )


def _load_settings(config: dict[str, Any]) -> GenerationSettings:
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise SuiteError("suite.toml must contain a [generation] table")

    max_new_tokens = _required_integer(generation, "max_new_tokens", minimum=1)
    do_sample = generation.get("do_sample")
    if not isinstance(do_sample, bool):
        raise SuiteError("generation.do_sample must be a boolean")
    temperature = _required_number(generation, "temperature", minimum_exclusive=0)
    top_k = _required_integer(generation, "top_k", minimum=0)
    repetition_penalty = _required_number(generation, "repetition_penalty", minimum_exclusive=0)
    seed = _required_integer(generation, "seed", minimum=0)
    prepend_bos = generation.get("prepend_bos")
    if prepend_bos not in {"auto", "always", "never"}:
        raise SuiteError("generation.prepend_bos must be auto, always, or never")

    return GenerationSettings(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        prepend_bos=prepend_bos,
    )


def _load_prompts(data: bytes, reference: str | Path) -> tuple[Prompt, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SuiteError(f"prompts.jsonl in {reference} is not valid UTF-8") from error

    prompts: list[Prompt] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SuiteError(
                f"prompts.jsonl line {line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise SuiteError(f"prompts.jsonl line {line_number}: record must be an object")

        prompt_id = record.get("id")
        prompt_text = record.get("prompt")
        category = record.get("category")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SuiteError(f"prompts.jsonl line {line_number}: invalid or missing 'id'")
        if prompt_id in seen_ids:
            raise SuiteError(f"duplicate prompt id: {prompt_id}")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise SuiteError(f"prompts.jsonl line {line_number}: invalid or missing 'prompt'")
        if category is not None and not isinstance(category, str):
            raise SuiteError(f"prompts.jsonl line {line_number}: 'category' must be a string")

        seen_ids.add(prompt_id)
        prompts.append(Prompt(prompt_id, prompt_text, category))

    if not prompts:
        raise SuiteError("suite must contain at least one prompt")
    return tuple(prompts)


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SuiteError(f"suite.{key} must be a non-empty string")
    return value


def _required_integer(config: dict[str, Any], key: str, minimum: int) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SuiteError(f"generation.{key} must be an integer >= {minimum}")
    return value


def _required_number(config: dict[str, Any], key: str, minimum_exclusive: float) -> float:
    value = config.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= minimum_exclusive:
        raise SuiteError(f"generation.{key} must be greater than {minimum_exclusive}")
    return float(value)
