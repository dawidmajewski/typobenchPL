from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typobenchpl.hf import GeneratedText, resolve_device
from typobenchpl.suite import GenerationSettings


class ByteTransformerError(ValueError):
    pass


class ByteTransformerGenerator:
    def __init__(self, model_reference: str, *, device: str = "auto") -> None:
        try:
            import safetensors
            import torch
            from safetensors.torch import load_file
        except ImportError as error:
            raise ByteTransformerError(
                "byte-transformer model support is not installed; run 'uv sync --extra hf'"
            ) from error

        root = Path(model_reference)
        config = _load_config(root)
        architecture = config["architecture"]
        context_length = architecture["context_length"]
        self.context_length = context_length
        self._torch = torch
        self._safetensors_version = safetensors.__version__
        self.device = resolve_device(torch, device)

        class TinyTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = architecture["width"]
                self.tokens = torch.nn.Embedding(256, width)
                self.positions = torch.nn.Embedding(context_length, width)
                layer = torch.nn.TransformerEncoderLayer(
                    width,
                    architecture["heads"],
                    width * 4,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                self.blocks = torch.nn.TransformerEncoder(
                    layer,
                    architecture["layers"],
                    enable_nested_tensor=False,
                )
                self.norm = torch.nn.LayerNorm(width)
                self.head = torch.nn.Linear(width, 256, bias=False)

            def forward(self, token_ids: Any) -> Any:
                length = token_ids.shape[1]
                positions = torch.arange(length, device=token_ids.device)
                hidden = self.tokens(token_ids) + self.positions(positions)
                mask = torch.ones(
                    length,
                    length,
                    dtype=torch.bool,
                    device=token_ids.device,
                ).triu(1)
                return self.head(self.norm(self.blocks(hidden, mask=mask)))

        self.model = TinyTransformer()
        try:
            state = load_file(str(root / "model.safetensors"))
            self.model.load_state_dict(state)
        except Exception as error:
            raise ByteTransformerError(
                f"could not load byte-transformer model '{model_reference}': {error}"
            ) from error

        self.model.eval()
        self.model.to(self.device)
        self.model_reference = model_reference

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": "byte-transformer",
            "model": self.model_reference,
            "revision_requested": None,
            "revision_resolved": None,
            "model_type": "byte-transformer-v1",
            "tokenizer": "utf8-byte",
            "device": str(self.device),
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "torch_version": self._torch.__version__,
            "safetensors_version": self._safetensors_version,
        }

    def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText:
        if settings.prepend_bos == "always":
            raise ByteTransformerError(
                "prepend_bos=always but the UTF-8 byte tokenizer has no BOS token"
            )

        torch = self._torch
        prompt_tokens = list(prompt.encode("utf-8"))
        tokens = prompt_tokens.copy() or [32]
        torch.manual_seed(seed)

        with torch.inference_mode():
            for _ in range(settings.max_new_tokens):
                input_ids = torch.tensor(
                    [tokens[-self.context_length :]],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(input_ids)[0, -1]
                logits = _apply_repetition_penalty(
                    torch,
                    logits,
                    tokens,
                    settings.repetition_penalty,
                )

                if settings.do_sample:
                    logits = logits / settings.temperature
                    if 0 < settings.top_k < logits.shape[-1]:
                        threshold = torch.topk(logits, settings.top_k).values[-1]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
                else:
                    next_token = int(torch.argmax(logits).item())
                tokens.append(next_token)

        generated_tokens = tokens[len(prompt_tokens) :]
        return GeneratedText(
            completion=bytes(generated_tokens),
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated_tokens),
            finish_reason="max_tokens",
        )


def is_byte_transformer_model(model_reference: str) -> bool:
    config_path = Path(model_reference) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return _is_byte_transformer_config(config)


def _load_config(root: Path) -> dict[str, Any]:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ByteTransformerError(
            f"could not read byte-transformer config in '{root}': {error}"
        ) from error

    if not isinstance(config, dict):
        raise ByteTransformerError("byte-transformer config must be a JSON object")
    if config.get("tokenizer") != "utf8-byte" or config.get("vocab_size") != 256:
        raise ByteTransformerError(
            "byte-transformer model must use the 256-token UTF-8 byte tokenizer"
        )

    architecture = config.get("architecture")
    required = ("width", "layers", "heads", "context_length")
    if not isinstance(architecture, dict) or any(
        not isinstance(architecture.get(key), int) or architecture[key] <= 0 for key in required
    ):
        raise ByteTransformerError("byte-transformer config has an invalid architecture")
    if architecture["width"] % architecture["heads"] != 0:
        raise ByteTransformerError("byte-transformer architecture width must be divisible by heads")
    return config


def _is_byte_transformer_config(config: object) -> bool:
    return (
        isinstance(config, dict)
        and config.get("tokenizer") == "utf8-byte"
        and config.get("vocab_size") == 256
        and isinstance(config.get("architecture"), dict)
    )


def _apply_repetition_penalty(
    torch: Any,
    logits: Any,
    tokens: list[int],
    penalty: float,
) -> Any:
    repeated = torch.tensor(sorted(set(tokens)), dtype=torch.long, device=logits.device)
    scores = logits[repeated]
    logits[repeated] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits
