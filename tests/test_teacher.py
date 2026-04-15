from __future__ import annotations

import types

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


def test_awq_uses_transformers_native_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        return FakeTokenizer()

    def fake_model_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        "peagle_q.teacher.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "peagle_q.teacher.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )

    runner = TransformersTeacherRunner(
        "fake-awq-model",
        quantization="awq",
        device="cuda",
        dtype_name="float16",
    )

    assert captured["device_map"] == "cuda:0"
    assert captured["torch_dtype"] == torch.float16
    runner.close()
