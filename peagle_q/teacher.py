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
        if self.quantization == "awq":
            try:
                from awq import AutoAWQForCausalLM
            except ImportError as exc:
                raise ImportError(
                    "AWQ support requires `pip install -e .[quant]` or `pip install autoawq`."
                ) from exc

            return AutoAWQForCausalLM.from_quantized(
                self.model_path,
                fuse_layers=False,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.revision,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=layer_indices is not None,
                return_dict=True,
            )

        logits: torch.Tensor | None
        if logits_mode == "none":
            logits = None
        elif logits_mode == "last":
            logits = outputs.logits[:, -1:, :].detach()
        elif logits_mode == "full":
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
                    hidden_states[layer_index + 1][0].detach().to(torch.float16).cpu()
                    for layer_index in layer_indices
                ],
                dim=0,
            )

        return TeacherOutput(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            logits=logits,
            selected_hidden_states=selected_hidden_states,
        )

    def get_output_projection(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight"):
            weight = self.model.lm_head.weight.detach().cpu().to(torch.float16)
            bias = getattr(self.model.lm_head, "bias", None)
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
