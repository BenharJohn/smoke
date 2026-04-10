from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def choose_attention_heads(hidden_size: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


@dataclass(slots=True)
class DraftHeadConfig:
    hidden_size: int
    num_tap_layers: int
    vocab_size: int
    layer_indices: list[int]
    teacher_model_path: str
    teacher_quantization: str
    projection_model_path: str | None = None
    projection_quantization: str = "none"
    dropout: float = 0.0
    attention_heads: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Eagle3StyleDraftHead(nn.Module):
    def __init__(self, config: DraftHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        input_size = config.num_tap_layers * hidden_size
        heads = config.attention_heads or choose_attention_heads(hidden_size)

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.mid_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.register_buffer("output_weight", torch.empty(0), persistent=False)
        self.register_buffer("output_bias", torch.empty(0), persistent=False)

    def attach_projection(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        self.output_weight = weight
        self.output_bias = torch.empty(0) if bias is None else bias

    def encode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(
            device=self.input_proj.weight.device,
            dtype=self.input_proj.weight.dtype,
        )
        batch_size, num_taps, seq_len, hidden_size = hidden_states.shape
        if num_taps != self.config.num_tap_layers:
            raise ValueError(
                f"expected {self.config.num_tap_layers} taps, received {num_taps}"
            )
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"expected hidden size {self.config.hidden_size}, received {hidden_size}"
            )

        fused = (
            hidden_states.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, num_taps * hidden_size)
        )
        x = self.input_proj(fused)
        x = self.input_norm(x)
        x = self.block(
            x,
            src_mask=causal_mask(seq_len, x.device),
            src_key_padding_mask=attention_mask == 0,
        )
        x = x + self.mlp(self.mid_norm(x))
        return self.output_norm(x)

    def project_logits(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.output_weight.numel() == 0:
            raise RuntimeError("output projection has not been attached")
        bias = None if self.output_bias.numel() == 0 else self.output_bias
        return F.linear(
            encoded,
            self.output_weight.to(device=encoded.device, dtype=encoded.dtype),
            None if bias is None else bias.to(device=encoded.device, dtype=encoded.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(hidden_states, attention_mask)
        return self.project_logits(encoded)

    def predict_next_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        temperature: float = 0.0,
    ) -> int:
        logits = self.forward(hidden_states, attention_mask)[0, -1]
        if temperature and temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            return int(torch.multinomial(probabilities, num_samples=1).item())
        return int(torch.argmax(logits).item())
