from __future__ import annotations

from collections.abc import Sequence

DEFAULT_LAYER_RATIOS = (0.125, 0.5, 0.875)


def parse_layer_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",")]
    parsed = [int(part) for part in values if part]
    return parsed or None


def resolve_layer_indices(
    num_hidden_layers: int,
    explicit: Sequence[int] | None = None,
) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    if explicit is not None:
        resolved = sorted({int(index) for index in explicit})
        if not resolved:
            raise ValueError("explicit layer selection cannot be empty")
        for index in resolved:
            if index < 0 or index >= num_hidden_layers:
                raise ValueError(
                    f"layer index {index} is out of range for {num_hidden_layers} layers"
                )
        return resolved

    resolved = sorted(
        {
            min(num_hidden_layers - 1, max(0, int(num_hidden_layers * ratio)))
            for ratio in DEFAULT_LAYER_RATIOS
        }
    )

    if len(resolved) == 3:
        return resolved

    # Very shallow models can collapse multiple ratio picks into the same layer.
    # In that case, spread the taps across the depth as evenly as possible.
    step = max(1, num_hidden_layers // 3)
    fallback = [0, min(num_hidden_layers - 1, step), num_hidden_layers - 1]
    return sorted(set(fallback))
