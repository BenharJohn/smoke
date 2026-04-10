from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from peagle_q.manifest import TensorManifestEntry, load_manifest


class HiddenStateDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_examples: int | None = None,
        max_length: int = 2048,
    ) -> None:
        entries = load_manifest(manifest_path)
        if max_examples is not None:
            entries = entries[:max_examples]
        self.entries = entries
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry: TensorManifestEntry = self.entries[index]
        record = torch.load(entry.tensor_path, map_location="cpu")
        input_ids = record["input_ids"].view(-1)[: self.max_length]
        attention_mask = record["attention_mask"].view(-1)[: self.max_length]
        hidden_states = record["hidden_states"][:, : self.max_length, :]
        return {
            "prompt_id": entry.prompt_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
        }


def hidden_state_collate(
    batch: list[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("batch cannot be empty")

    prompt_ids = [str(item["prompt_id"]) for item in batch]
    num_taps = int(batch[0]["hidden_states"].shape[0])
    hidden_size = int(batch[0]["hidden_states"].shape[-1])
    max_seq = max(int(item["input_ids"].shape[0]) for item in batch)

    input_ids = torch.zeros(len(batch), max_seq, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_seq, dtype=torch.long)
    hidden_states = torch.zeros(
        len(batch),
        num_taps,
        max_seq,
        hidden_size,
        dtype=torch.float16,
    )

    for row_index, item in enumerate(batch):
        seq_len = int(item["input_ids"].shape[0])
        input_ids[row_index, :seq_len] = item["input_ids"]
        attention_mask[row_index, :seq_len] = item["attention_mask"]
        hidden_states[row_index, :, :seq_len, :] = item["hidden_states"]

    return {
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "hidden_states": hidden_states,
    }
