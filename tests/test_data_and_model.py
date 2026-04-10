import torch

from peagle_q.data import build_training_prompt, normalize_sharegpt_messages
from peagle_q.model import DraftHeadConfig, Eagle3StyleDraftHead


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        rendered = [f"{item['role']}:{item['content']}" for item in messages]
        if add_generation_prompt:
            rendered.append("assistant:")
        return " | ".join(rendered)


def test_sharegpt_normalization_and_prompt_rendering() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ]
    }
    messages = normalize_sharegpt_messages(record)
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    prompt = build_training_prompt(record, FakeTokenizer())
    assert prompt == "user:hello | assistant:hi"


def test_draft_head_accepts_half_hidden_states_with_float32_weights() -> None:
    config = DraftHeadConfig(
        hidden_size=8,
        num_tap_layers=3,
        vocab_size=16,
        layer_indices=[1, 3, 5],
        teacher_model_path="teacher",
        teacher_quantization="none",
    )
    model = Eagle3StyleDraftHead(config).float()
    model.attach_projection(torch.randn(16, 8, dtype=torch.float16), None)

    hidden_states = torch.randn(1, 3, 4, 8, dtype=torch.float16)
    attention_mask = torch.ones(1, 4, dtype=torch.long)

    logits = model(hidden_states, attention_mask)
    assert list(logits.shape) == [1, 4, 16]
