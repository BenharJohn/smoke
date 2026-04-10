from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TensorManifestEntry:
    prompt_id: str
    dataset_index: int
    prompt_sha256: str
    teacher_model_path: str
    teacher_revision: str | None
    tokenizer_path: str
    tokenizer_revision: str | None
    quantization: str
    layer_indices: list[int]
    tensor_path: str
    prompt_length: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TensorManifestEntry":
        return cls(**payload)


def append_manifest_entry(manifest_path: str | Path, entry: TensorManifestEntry) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def load_manifest(manifest_path: str | Path) -> list[TensorManifestEntry]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(path)

    entries: list[TensorManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(TensorManifestEntry.from_dict(json.loads(line)))
    return entries
