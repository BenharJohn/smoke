from __future__ import annotations

import sys
import types

import pytest
import torch

from peagle_q.teacher import TransformersTeacherRunner


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2
    eos_token = "</s>"
    pad_token = None
    _commit_hash = "tok-rev"


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            _commit_hash="model-rev",
        )

    def eval(self):
        return self

    def to(self, device):
        self._device = device
        return self


def test_awq_uses_autoawq_from_quantized(monkeypatch) -> None:
    """AWQ models must route through AutoAWQForCausalLM.from_quantized.

    Commit 3df8474 reverted the earlier "native transformers loader" approach
    because transformers' meta-init path broke on Sol. The current code calls
    ``AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=False)``; this
    test pins that contract so we do not silently regress.
    """
    captured: dict[str, object] = {}

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        return FakeTokenizer()

    class FakeAutoAWQ:
        @staticmethod
        def from_quantized(model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)
            return FakeModel()

    monkeypatch.setattr(
        "peagle_q.teacher.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    # Inject a fake `awq` module so the `from awq import AutoAWQForCausalLM`
    # line inside _load_awq_model resolves to our stub without touching the
    # real autoawq package (which would try to hit the HF hub).
    fake_awq_module = types.ModuleType("awq")
    fake_awq_module.AutoAWQForCausalLM = FakeAutoAWQ
    monkeypatch.setitem(sys.modules, "awq", fake_awq_module)

    runner = TransformersTeacherRunner(
        "fake-awq-model",
        quantization="awq",
        device="cuda",
        dtype_name="float16",
    )

    assert captured["model_path"] == "fake-awq-model"
    assert captured["fuse_layers"] is False
    runner.close()
