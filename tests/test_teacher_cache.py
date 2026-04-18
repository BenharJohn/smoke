from __future__ import annotations

import pytest
import torch

from peagle_q.teacher import trim_kv_cache


def _make_legacy_tuple_cache(
    num_layers: int, heads: int, seq: int, head_dim: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            torch.randn(1, heads, seq, head_dim),
            torch.randn(1, heads, seq, head_dim),
        )
        for _ in range(num_layers)
    )


class _FakeDynamicCache:
    """Minimal stand-in that mirrors the ``key_cache``/``value_cache`` attribute
    contract of newer HuggingFace ``DynamicCache`` instances."""

    def __init__(self, num_layers: int, heads: int, seq: int, head_dim: int) -> None:
        self.key_cache = [
            torch.randn(1, heads, seq, head_dim) for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.randn(1, heads, seq, head_dim) for _ in range(num_layers)
        ]
        self._seen_tokens = seq


def test_trim_kv_cache_handles_legacy_tuple() -> None:
    cache = _make_legacy_tuple_cache(num_layers=2, heads=4, seq=10, head_dim=8)
    trimmed = trim_kv_cache(cache, target_length=6)
    assert isinstance(trimmed, tuple)
    for key, value in trimmed:
        assert key.shape == (1, 4, 6, 8)
        assert value.shape == (1, 4, 6, 8)


def test_trim_kv_cache_handles_dynamic_cache_fallback() -> None:
    cache = _FakeDynamicCache(num_layers=2, heads=4, seq=10, head_dim=8)
    trimmed = trim_kv_cache(cache, target_length=3)
    assert trimmed is cache  # mutated in place
    for key_tensor in cache.key_cache:
        assert key_tensor.shape == (1, 4, 3, 8)
    for value_tensor in cache.value_cache:
        assert value_tensor.shape == (1, 4, 3, 8)
    assert cache._seen_tokens == 3


def test_trim_kv_cache_passes_through_none() -> None:
    assert trim_kv_cache(None, target_length=5) is None


def test_trim_kv_cache_rejects_negative_length() -> None:
    cache = _make_legacy_tuple_cache(num_layers=1, heads=2, seq=4, head_dim=2)
    with pytest.raises(ValueError):
        trim_kv_cache(cache, target_length=-1)


def test_trim_kv_cache_raises_for_unknown_container() -> None:
    with pytest.raises(RuntimeError):
        trim_kv_cache(object(), target_length=1)
