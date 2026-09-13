from __future__ import annotations

from typing import Any, Literal

from typobenchpl.suite import GenerationSettings


class HuggingFaceError(ValueError):
    pass


class HuggingFaceCausalGenerator:
    def __init__(
        self,
        model_reference: str,
        *,
        revision: str | None = None,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as error:
            raise HuggingFaceError(
                "Hugging Face support is not installed; run 'uv sync --extra hf'"
            ) from error

        self._torch = torch
        self._set_seed = set_seed
        self._transformers_version = transformers.__version__
        self.device = resolve_device(torch, device)

        load_options = {"revision": revision} if revision is not None else {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_reference, **load_options)
            self.model = AutoModelForCausalLM.from_pretrained(model_reference, **load_options)
        except Exception as error:
            raise HuggingFaceError(f"could not load model '{model_reference}': {error}") from error

        self.model.eval()
        self.model.to(self.device)
        self.model_reference = model_reference
        self.revision_requested = revision

    @property
    def metadata(self) -> dict[str, object]:
        config = self.model.config
        return {
            "backend": "transformers-causal",
            "model": self.model_reference,
            "revision_requested": self.revision_requested,
            "revision_resolved": getattr(config, "_commit_hash", None),
            "model_type": getattr(config, "model_type", None),
            "device": str(self.device),
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "torch_version": self._torch.__version__,
            "transformers_version": self._transformers_version,
        }

    def generate(self, prompt: str, settings: GenerationSettings, seed: int) -> GeneratedText:
        torch = self._torch
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        input_ids, attention_mask = self._apply_bos(input_ids, attention_mask, settings.prepend_bos)

        input_length = int(input_ids.shape[-1])
        self._validate_context(input_length, settings.max_new_tokens)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self._set_seed(seed)
        generation_options: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "repetition_penalty": settings.repetition_penalty,
        }
        if settings.do_sample:
            generation_options["temperature"] = settings.temperature
            generation_options["top_k"] = settings.top_k

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.model.config, "eos_token_id", None)
        if pad_token_id is not None:
            generation_options["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_options,
            )

        generated_ids = output[0, input_length:].detach().cpu().tolist()
        completion = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        eos_token_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or ())
        finished_with_eos = bool(generated_ids and generated_ids[-1] in eos_token_ids)
        return GeneratedText(
            completion=completion,
            prompt_tokens=input_length,
            generated_tokens=len(generated_ids),
            finish_reason=(
                "eos"
                if finished_with_eos or len(generated_ids) < settings.max_new_tokens
                else "max_tokens"
            ),
        )

    def _apply_bos(self, input_ids: Any, attention_mask: Any, mode: str) -> tuple[Any, Any]:
        bos_token_id = self.tokenizer.bos_token_id
        if mode == "never" or bos_token_id is None:
            if mode == "always" and bos_token_id is None:
                raise HuggingFaceError("prepend_bos=always but tokenizer has no BOS token")
            return input_ids, attention_mask

        already_present = input_ids.shape[-1] > 0 and int(input_ids[0, 0]) == bos_token_id
        configured = getattr(self.model.config, "bos_token_id", None) is not None
        should_prepend = not already_present and (mode == "always" or configured)
        if not should_prepend:
            return input_ids, attention_mask

        bos = self._torch.full(
            (input_ids.shape[0], 1),
            bos_token_id,
            dtype=input_ids.dtype,
        )
        input_ids = self._torch.cat((bos, input_ids), dim=1)
        if attention_mask is not None:
            prefix = self._torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
            )
            attention_mask = self._torch.cat((prefix, attention_mask), dim=1)
        return input_ids, attention_mask

    def _validate_context(self, prompt_tokens: int, max_new_tokens: int) -> None:
        config = self.model.config
        context_size = getattr(config, "max_position_embeddings", None)
        if context_size is None:
            context_size = getattr(config, "n_positions", None)
        if isinstance(context_size, int) and prompt_tokens + max_new_tokens > context_size:
            raise HuggingFaceError(
                f"prompt ({prompt_tokens} tokens) plus generation ({max_new_tokens} tokens) "
                f"exceeds model context ({context_size} tokens)"
            )


class GeneratedText:
    __slots__ = ("completion", "finish_reason", "generated_tokens", "prompt_tokens")

    def __init__(
        self,
        completion: str | bytes,
        prompt_tokens: int,
        generated_tokens: int,
        finish_reason: Literal["eos", "max_tokens"] = "max_tokens",
    ) -> None:
        self.completion = completion
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.finish_reason = finish_reason


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise HuggingFaceError(f"invalid device: {requested}") from error

    if device.type == "cuda" and not torch.cuda.is_available():
        raise HuggingFaceError("CUDA was requested but is not available")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise HuggingFaceError("MPS was requested but is not available")
    return device
