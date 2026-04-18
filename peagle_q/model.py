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

    def propose_sequence(
        self,
        initial_hidden: torch.Tensor,
        num_proposals: int,
        *,
        temperature: float = 0.0,
    ) -> tuple[list[int], torch.Tensor]:
        """Autoregressive self-fed draft proposals (no verifier call per step).

        This enables **parallel-verification speculative decoding** — the
        caller can gather ``num_proposals`` tokens with a single draft-side
        loop, then verify them in one verifier forward pass with KV cache.

        Step 0 consumes the real verifier hidden at the commit tip (shape
        ``[1, num_tap_layers, 1, H]``). Steps 1..K-1 self-feed: the draft's
        own output hidden is broadcast across the ``num_tap_layers`` axis
        and recycled as input. This is an approximation — the tap axis is
        supposed to carry multi-scale features from different verifier
        layers, but at inference we only have the draft's single-stream
        output. Acceptance may be slightly lower than the sequential
        verifier-fed path; that quality gap is the cost of real parallel
        speedup without a retraining-based EAGLE-2 recurrence head.

        Returns
        -------
        proposals:
            List of exactly ``num_proposals`` token ids (CPU Python ints).
        emitted_hiddens:
            Tensor ``[num_proposals, H]`` of draft output hiddens, one per
            proposal. Exposed for downstream diagnostics (e.g., measuring
            the self-feed drift between draft-hidden and real verifier hidden).
        """
        if num_proposals <= 0:
            hidden_size = self.config.hidden_size
            device = self.input_proj.weight.device
            return [], torch.zeros(0, hidden_size, device=device)

        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype
        hidden = initial_hidden.to(device=device, dtype=dtype)
        if hidden.dim() != 4:
            raise ValueError(
                f"initial_hidden must be [B, num_taps, seq, H], got shape {tuple(hidden.shape)}"
            )
        if hidden.shape[1] != self.config.num_tap_layers:
            raise ValueError(
                f"initial_hidden tap axis {hidden.shape[1]} does not match "
                f"config.num_tap_layers={self.config.num_tap_layers}"
            )

        proposals: list[int] = []
        emitted: list[torch.Tensor] = []
        mask = torch.ones(hidden.shape[0], 1, dtype=torch.long, device=device)
        num_taps = self.config.num_tap_layers

        for _ in range(num_proposals):
            encoded = self.encode(hidden, mask)  # [B, 1, H]
            logits = self.project_logits(encoded[:, -1:, :])  # [B, 1, V]
            last_logits = logits[0, 0]
            if temperature and temperature > 0:
                probabilities = torch.softmax(last_logits / temperature, dim=-1)
                token = int(torch.multinomial(probabilities, num_samples=1).item())
            else:
                token = int(torch.argmax(last_logits).item())
            proposals.append(token)
            emitted.append(encoded[0, -1].detach())
            # Self-feed: reshape [B, 1, H] -> [B, num_taps, 1, H] via broadcast.
            hidden = encoded.unsqueeze(1).expand(-1, num_taps, -1, -1).contiguous()

        return proposals, torch.stack(emitted, dim=0)
