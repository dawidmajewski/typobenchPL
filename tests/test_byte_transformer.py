from __future__ import annotations

import json

import pytest

from typobenchpl.byte_transformer import (
    ByteTransformerError,
    ByteTransformerGenerator,
    is_byte_transformer_model,
)


def test_detects_byte_transformer_model_by_capabilities(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "format": "byte-transformer-v1",
                "tokenizer": "utf8-byte",
                "vocab_size": 256,
                "architecture": {},
            }
        ),
        encoding="utf-8",
    )

    assert is_byte_transformer_model(str(tmp_path))


def test_rejects_other_or_invalid_configs(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    assert not is_byte_transformer_model(str(tmp_path))

    config_path.write_text("not json", encoding="utf-8")
    assert not is_byte_transformer_model(str(tmp_path))


def test_byte_transformer_initialization_wraps_model_load_errors(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "format": "byte-transformer-v1",
                "tokenizer": "utf8-byte",
                "vocab_size": 256,
                "architecture": {
                    "width": 16,
                    "layers": 1,
                    "heads": 2,
                    "context_length": 8,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ByteTransformerError, match=r"could not load byte-transformer model"):
        ByteTransformerGenerator(str(tmp_path), device="cpu")
