from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class TeacherOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor | None
    selected_hidden_states: torch.Tensor | None
    past_key_values: Any | None = None


def infer_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_dtype(dtype_name: str | None, device: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class TransformersTeacherRunner:
    def __init__(
        self,
        model_path: str,
        *,
        quantization: str = "none",
        revision: str | None = None,
        device: str | None = None,
        dtype_name: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.quantization = quantization
        self.revision = revision
        self.device = infer_device(device)
        self.dtype = infer_dtype(dtype_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> Any:
        quantization = self._resolve_quantization()

        if quantization == "awq":
            return self._load_awq_model()

        if quantization == "w8a8":
            return self._load_compressed_tensors_model()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        return model.to(self.device)

    def _resolve_quantization(self) -> str:
        if self.quantization != "none":
            return self.quantization

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.model_path,
                revision=self.revision,
                trust_remote_code=False,
            )
            quant_config = getattr(config, "quantization_config", None)
            if isinstance(quant_config, dict):
                method = quant_config.get("quant_method")
                if method == "awq":
                    self.quantization = "awq"
                    return "awq"
                if method == "compressed-tensors":
                    self.quantization = "w8a8"
                    return "w8a8"
        except Exception:
            pass

        return self.quantization

    def _load_compressed_tensors_model(self) -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        return model.to(self.device)

    def _load_awq_model(self) -> Any:
        # Disable autoawq's Triton GEMM backend before importing AutoAWQForCausalLM.
        # Triton JIT-compiles CUDA utilities on first use, which requires Python.h
        # (Python development headers).  These are often absent on HPC compute nodes.
        # Setting AWQ_TRITON_AVAILABLE=False before the linear modules are imported
        # forces autoawq to use its pre-compiled CUDA extension (awq_ext) instead.
        try:
            import awq.modules.triton.gemm as _awq_triton_mod
            _awq_triton_mod.AWQ_TRITON_AVAILABLE = False
        except Exception:
            pass


        try:
            from awq import AutoAWQForCausalLM
        except ImportError as exc:
            raise ImportError(
                "AWQ support requires `pip install -e .[quant]` or `pip install autoawq`."
            ) from exc

        model = AutoAWQForCausalLM.from_quantized(
            self.model_path,
            fuse_layers=False,
        )
        return model.to(self.device)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def tokenizer_revision(self) -> str | None:
        return getattr(self.tokenizer, "_commit_hash", None)

    @property
    def model_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    def encode_text(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def forward_prompt(
        self,
        text: str,
        *,
        layer_indices: list[int] | None = None,
        max_length: int | None = None,
        logits_mode: str = "full",
    ) -> TeacherOutput:
        encoded = self.encode_text(text, max_length=max_length)
        return self.forward_ids(
            encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            layer_indices=layer_indices,
            logits_mode=logits_mode,
        )

    def forward_ids(
        self,
        input_ids: torch.Tensor | list[int],
        *,
        attention_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
        logits_mode: str = "full",
        past_key_values: Any | None = None,
        use_cache: bool = False,
        hidden_states_on_device: bool = False,
    ) -> TeacherOutput:
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to(self.device)
        else:
            input_ids = input_ids.to(self.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": layer_indices is not None,
            "return_dict": True,
        }
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
        if use_cache:
            model_kwargs["use_cache"] = True

        with torch.no_grad():
            outputs = self.model(**model_kwargs)

        logits: torch.Tensor | None
        if logits_mode == "none":
            logits = None
        elif logits_mode == "last":
            logits = outputs.logits[:, -1:, :].detach()
        elif logits_mode == "full" or logits_mode == "all":
            # `all` is an alias of `full`; kept for readability at parallel-speculative call sites.
            logits = outputs.logits.detach()
        else:
            raise ValueError(f"unsupported logits_mode: {logits_mode}")

        selected_hidden_states: torch.Tensor | None = None
        if layer_indices is not None:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("teacher model did not return hidden states")
            selected_hidden_states = torch.stack(
                [
                    hidden_states[layer_index + 1][0].detach().to(self.dtype)
                    for layer_index in layer_indices
                ],
                dim=0,
            )
            if not hidden_states_on_device:
                selected_hidden_states = selected_hidden_states.cpu()

        returned_cache = getattr(outputs, "past_key_values", None) if use_cache else None

        return TeacherOutput(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            logits=logits,
            selected_hidden_states=selected_hidden_states,
            past_key_values=returned_cache,
        )

    def get_output_projection(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            weight = lm_head.weight
            if hasattr(weight, "data"):
                weight = weight.data
            weight = weight.detach().cpu().to(torch.float16)
            bias = getattr(lm_head, "bias", None)
            return weight, None if bias is None else bias.detach().cpu().to(torch.float16)

        embeddings = self.model.get_input_embeddings()
        if embeddings is None or not hasattr(embeddings, "weight"):
            raise RuntimeError("unable to locate output projection or input embeddings")
        return embeddings.weight.detach().cpu().to(torch.float16), None

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def trim_kv_cache(cache: Any, target_length: int) -> Any:
    """Trim a HuggingFace KV cache to exactly ``target_length`` positions.

    Supports the two cache container shapes that ``transformers`` produces
    across supported versions:

    * Legacy tuple-of-tuples: ``((k0, v0), (k1, v1), ...)`` where each tensor
      has shape ``[batch, heads, seq, head_dim]``.
    * ``DynamicCache`` (object with ``key_cache``/``value_cache`` lists or a
      ``crop`` method).

    Raises :class:`RuntimeError` if the cache type is unrecognised so that
    parallel speculative decoding fails loudly rather than silently producing
    wrong outputs.
    """
    if cache is None:
        return None
    if target_length < 0:
        raise ValueError(f"target_length must be non-negative, got {target_length}")

    # Preferred: DynamicCache-style object with an explicit crop hook.
    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(target_length)
        return cache

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        for index, key_tensor in enumerate(key_cache):
            if key_tensor is None:
                continue
            key_cache[index] = key_tensor[..., :target_length, :]
        for index, value_tensor in enumerate(value_cache):
            if value_tensor is None:
                continue
            value_cache[index] = value_tensor[..., :target_length, :]
        seen_tokens = getattr(cache, "_seen_tokens", None)
        if seen_tokens is not None:
            cache._seen_tokens = min(int(seen_tokens), target_length)
        return cache

    if isinstance(cache, tuple):
        return tuple(
            (key[..., :target_length, :], value[..., :target_length, :])
            for key, value in cache
        )

    raise RuntimeError(f"unsupported past_key_values container: {type(cache)!r}")
