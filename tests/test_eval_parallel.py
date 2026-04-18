"""End-to-end smoke tests for the parallel-verification speculative path.

These tests do not load a real verifier model; they build a fake that exposes
just enough of ``TransformersTeacherRunner`` for ``speculative_generate_parallel``
and ``speculative_generate_local`` to drive a short generation loop. The goal
is to catch regressions in the cache bookkeeping, mode dispatch, and
``_ModeAggregator`` plumbing without requiring GPU access.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from peagle_q.eval import (
    _ModeAggregator,
    _run_speculative,
    speculative_generate_parallel,
)
from peagle_q.model import DraftHeadConfig, Eagle3StyleDraftHead
from peagle_q.teacher import TeacherOutput


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = None

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        token_ids = [ord(ch) % 16 for ch in text][:10]
        if not token_ids:
            token_ids = [0]
        tensor = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(t) % 128) for t in token_ids)


class _FakeCache:
    """Toy cache that remembers its length and implements the ``crop`` hook
    expected by :func:`peagle_q.teacher.trim_kv_cache`.
    """

    __slots__ = ("length",)

    def __init__(self, length: int) -> None:
        self.length = length

    def crop(self, target_length: int) -> None:
        self.length = min(self.length, target_length)


class _FakeVerifier:
    """Stand-in for ``TransformersTeacherRunner`` that returns deterministic
    logits and hidden states from a counter, with a toy cache that tracks its
    running length so ``trim_kv_cache`` can be exercised.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_taps: int) -> None:
        self.tokenizer = _FakeTokenizer()
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_taps = num_taps
        self.calls: list[dict] = []

    def encode_text(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    def forward_ids(
        self,
        input_ids,
        *,
        attention_mask=None,
        layer_indices=None,
        logits_mode: str = "full",
        past_key_values=None,
        use_cache: bool = False,
        hidden_states_on_device: bool = False,
    ) -> TeacherOutput:
        if isinstance(input_ids, list):
            new_length = len(input_ids)
        else:
            new_length = int(input_ids.shape[-1])

        previous = past_key_values.length if isinstance(past_key_values, _FakeCache) else 0
        total_length = previous + new_length
        self.calls.append(
            {"new_length": new_length, "previous": previous, "logits_mode": logits_mode}
        )

        # Deterministic "argmax cycles through the vocab" — makes the
        # accept/reject decisions predictable for the draft under test.
        logits = torch.zeros(1, new_length, self.vocab_size)
        for index in range(new_length):
            position = previous + index
            logits[0, index, position % self.vocab_size] = 10.0

        if logits_mode == "last":
            logits = logits[:, -1:, :]

        hidden = torch.zeros(self.num_taps, new_length, self.hidden_size)

        return TeacherOutput(
            input_ids=torch.zeros(1, new_length, dtype=torch.long),
            attention_mask=torch.ones(1, total_length, dtype=torch.long),
            logits=logits,
            selected_hidden_states=hidden,
            past_key_values=_FakeCache(length=total_length),
        )


def _make_draft(vocab_size: int, hidden_size: int, num_taps: int) -> Eagle3StyleDraftHead:
    config = DraftHeadConfig(
        hidden_size=hidden_size,
        num_tap_layers=num_taps,
        vocab_size=vocab_size,
        layer_indices=[0, 1, 2][:num_taps],
        teacher_model_path="fake",
        teacher_quantization="none",
    )
    model = Eagle3StyleDraftHead(config).float().eval()
    model.attach_projection(torch.randn(vocab_size, hidden_size), None)
    return model


def test_parallel_path_runs_and_produces_well_formed_output() -> None:
    verifier = _FakeVerifier(vocab_size=16, hidden_size=8, num_taps=3)
    draft = _make_draft(vocab_size=16, hidden_size=8, num_taps=3)

    with torch.no_grad():
        result = speculative_generate_parallel(
            verifier,
            draft,
            "abc",
            draft_length=3,
            max_new_tokens=8,
            temperature=0.0,
        )

    assert result["speculation_mode"] == "parallel"
    assert 0 <= result["tau"] <= 3
    assert len(result["alpha"]) == 3
    assert result["verifier_forward_calls"] >= 2  # prefill + at least one verify
    assert isinstance(result["text"], str)
    # We should have generated at least one committed token per loop iteration.
    assert len(result["generated_ids"]) >= 1


def test_parallel_path_rejects_temperature_sampling() -> None:
    verifier = _FakeVerifier(vocab_size=8, hidden_size=4, num_taps=2)
    draft = _make_draft(vocab_size=8, hidden_size=4, num_taps=2)

    try:
        speculative_generate_parallel(
            verifier,
            draft,
            "x",
            draft_length=2,
            max_new_tokens=2,
            temperature=0.7,
        )
    except ValueError as exc:
        assert "greedy" in str(exc)
    else:
        raise AssertionError("non-zero temperature must raise in the parallel path")


def test_run_speculative_dispatches_to_parallel() -> None:
    verifier = _FakeVerifier(vocab_size=16, hidden_size=8, num_taps=3)
    draft = _make_draft(vocab_size=16, hidden_size=8, num_taps=3)
    with torch.no_grad():
        result = _run_speculative(
            "parallel",
            verifier,
            draft,
            "hi",
            draft_length=2,
            max_new_tokens=4,
            temperature=0.0,
        )
    assert result["speculation_mode"] == "parallel"


def test_mode_aggregator_records_and_summarises() -> None:
    aggregator = _ModeAggregator.create("parallel", draft_length=3)
    fake_result = {
        "tau": 2.0,
        "tokens_per_second": 12.0,
        "ttft_seconds": 0.1,
        "itl_seconds": 0.05,
        "verifier_forward_calls": 5,
        "position_attempts": [3, 3, 3],
        "position_accepts": [3, 2, 1],
    }
    speedup = aggregator.observe(fake_result, ar_tokens_per_second=4.0)
    assert speedup == 3.0
    summary = aggregator.summarise()
    assert summary["tau"] == 2.0
    assert summary["theoretical_speedup_ideal"] == 3.0
    assert summary["alpha"] == [1.0, 2 / 3, 1 / 3]
    assert summary["speedup_vs_ar"] == 3.0
