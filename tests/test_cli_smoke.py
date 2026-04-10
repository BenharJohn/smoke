from __future__ import annotations

import argparse
import json
import sys
import types

from peagle_q import cli


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = [f"{item['role']}:{item['content']}" for item in messages]
        if add_generation_prompt:
            rendered.append("assistant:")
        return " | ".join(rendered)


class FakeTeacherRunner:
    def __init__(self, *args, **kwargs):
        self.tokenizer = FakeTokenizer()
        self.num_hidden_layers = 32

    def forward_prompt(self, text, *, layer_indices=None, max_length=None):
        assert text == "user:hello world | assistant:"
        return type(
            "FakeOutput",
            (),
            {
                "input_ids": type("FakeIds", (), {"shape": (1, 6)})(),
                "selected_hidden_states": type("FakeHidden", (), {"shape": (3, 6, 16)})(),
            },
        )()

    def close(self):
        return None


def test_run_smoke_uses_chat_template_and_reports_generation(monkeypatch, capsys) -> None:
    fake_benchmarks = types.SimpleNamespace(
        render_generation_prompt=lambda tokenizer, messages: tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    fake_eval = types.SimpleNamespace(
        _load_draft_model=lambda *args, **kwargs: None,
        greedy_autoregressive_generate=lambda *args, **kwargs: {
            "text": "speculative decoding drafts tokens",
            "latency_seconds": 1.5,
            "ttft_seconds": 0.5,
            "itl_seconds": 0.25,
            "tokens_per_second": 8.0,
        },
        speculative_generate_local=lambda *args, **kwargs: {
            "tau": 2.0,
            "alpha": [1.0, 1.0, 0.0],
            "text": "draft output",
        },
    )
    fake_teacher = types.SimpleNamespace(
        TransformersTeacherRunner=FakeTeacherRunner,
        infer_device=lambda device=None: device or "cpu",
    )

    monkeypatch.setitem(sys.modules, "peagle_q.benchmarks", fake_benchmarks)
    monkeypatch.setitem(sys.modules, "peagle_q.eval", fake_eval)
    monkeypatch.setitem(sys.modules, "peagle_q.teacher", fake_teacher)

    namespace = argparse.Namespace(
        command="smoke",
        config=None,
        teacher_model_path="fake/model",
        teacher_quantization="none",
        teacher_revision=None,
        prompt="hello world",
        layer_indices=None,
        draft_checkpoint=None,
        max_new_tokens=8,
        device="cpu",
    )

    cli.run_smoke(namespace)
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["device"] == "cpu"
    assert payload["teacher_model_path"] == "fake/model"
    assert payload["rendered_prompt_preview"] == "user:hello world | assistant:"
    assert payload["hidden_state_shape"] == [3, 6, 16]
    assert payload["autoregressive"]["text"] == "speculative decoding drafts tokens"
