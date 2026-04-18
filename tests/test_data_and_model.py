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


def test_propose_sequence_emits_tokens_and_hiddens_without_verifier() -> None:
    num_taps = 3
    hidden_size = 8
    vocab_size = 16
    config = DraftHeadConfig(
        hidden_size=hidden_size,
        num_tap_layers=num_taps,
        vocab_size=vocab_size,
        layer_indices=[1, 3, 5],
        teacher_model_path="teacher",
        teacher_quantization="none",
    )
    model = Eagle3StyleDraftHead(config).float().eval()
    model.attach_projection(torch.randn(vocab_size, hidden_size, dtype=torch.float16), None)

    initial_hidden = torch.randn(1, num_taps, 1, hidden_size, dtype=torch.float16)
    with torch.no_grad():
        proposals, emitted = model.propose_sequence(initial_hidden, num_proposals=5)

    assert len(proposals) == 5
    for token in proposals:
        assert isinstance(token, int)
        assert 0 <= token < vocab_size
    assert tuple(emitted.shape) == (5, hidden_size)


def test_propose_sequence_rejects_wrong_tap_axis() -> None:
    config = DraftHeadConfig(
        hidden_size=4,
        num_tap_layers=3,
        vocab_size=5,
        layer_indices=[0, 1, 2],
        teacher_model_path="teacher",
        teacher_quantization="none",
    )
    model = Eagle3StyleDraftHead(config).float().eval()
    model.attach_projection(torch.randn(5, 4), None)

    wrong_hidden = torch.randn(1, 2, 1, 4)  # tap axis has 2 instead of 3
    try:
        model.propose_sequence(wrong_hidden, num_proposals=1)
    except ValueError as exc:
        assert "tap axis" in str(exc)
    else:
        raise AssertionError("propose_sequence should reject mismatched tap axis")


def test_propose_sequence_handles_zero_proposals() -> None:
    config = DraftHeadConfig(
        hidden_size=4,
        num_tap_layers=2,
        vocab_size=5,
        layer_indices=[0, 1],
        teacher_model_path="teacher",
        teacher_quantization="none",
    )
    model = Eagle3StyleDraftHead(config).float().eval()
    model.attach_projection(torch.randn(5, 4), None)

    initial_hidden = torch.randn(1, 2, 1, 4)
    proposals, emitted = model.propose_sequence(initial_hidden, num_proposals=0)
    assert proposals == []
    assert tuple(emitted.shape) == (0, 4)
