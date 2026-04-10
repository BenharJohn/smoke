from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import torch

from peagle_q.data import build_training_prompt, iter_jsonl, prompt_sha256, record_prompt_id
from peagle_q.layers import resolve_layer_indices
from peagle_q.manifest import TensorManifestEntry, append_manifest_entry, load_manifest
from peagle_q.teacher import TransformersTeacherRunner


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned[:120] or "sample"


def _tensor_filename(prompt_id: str, dataset_index: int, prompt_hash: str) -> str:
    return f"{dataset_index:06d}_{_safe_filename(prompt_id)}_{prompt_hash[:12]}.pt"


def _resolve_resume_manifest(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "manifest.jsonl"
    return candidate


@dataclass(slots=True)
class ExtractRunConfig:
    dataset: str
    teacher_model_path: str
    output_dir: str
    teacher_quantization: str = "none"
    teacher_revision: str | None = None
    max_examples: int | None = None
    max_length: int = 2048
    layer_indices: list[int] | None = None
    resume_from: str | None = None
    dtype: str | None = None
    device: str | None = None
    extractor_backend: str = "transformers"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_hidden_states(config: ExtractRunConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    tensor_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    resume_manifest = _resolve_resume_manifest(config.resume_from)
    completed_samples: set[tuple[int, str]] = set()
    if resume_manifest and resume_manifest.exists():
        completed_samples = {
            (entry.dataset_index, entry.prompt_id) for entry in load_manifest(resume_manifest)
        }
        if not manifest_path.exists() and resume_manifest != manifest_path:
            manifest_path.write_text(resume_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    elif manifest_path.exists():
        completed_samples = {
            (entry.dataset_index, entry.prompt_id) for entry in load_manifest(manifest_path)
        }

    teacher = TransformersTeacherRunner(
        config.teacher_model_path,
        quantization=config.teacher_quantization,
        revision=config.teacher_revision,
        device=config.device,
        dtype_name=config.dtype,
    )
    layer_indices = resolve_layer_indices(
        teacher.num_hidden_layers,
        explicit=config.layer_indices,
    )

    processed = 0
    skipped = 0
    for dataset_index, record in enumerate(iter_jsonl(config.dataset)):
        prompt_id = record_prompt_id(record, dataset_index)
        sample_key = (dataset_index, prompt_id)
        if sample_key in completed_samples:
            skipped += 1
            continue
        if config.max_examples is not None and processed >= config.max_examples:
            break

        prompt_text = build_training_prompt(record, teacher.tokenizer)
        prompt_hash = prompt_sha256(prompt_text)
        teacher_output = teacher.forward_prompt(
            prompt_text,
            layer_indices=layer_indices,
            max_length=config.max_length,
            logits_mode="none",
        )
        if teacher_output.selected_hidden_states is None:
            raise RuntimeError("hidden-state extraction did not return selected layers")

        tensor_path = tensor_dir / _tensor_filename(prompt_id, dataset_index, prompt_hash)
        torch.save(
            {
                "input_ids": teacher_output.input_ids,
                "attention_mask": teacher_output.attention_mask,
                "hidden_states": teacher_output.selected_hidden_states,
            },
            tensor_path,
        )

        entry = TensorManifestEntry(
            prompt_id=prompt_id,
            dataset_index=dataset_index,
            prompt_sha256=prompt_hash,
            teacher_model_path=config.teacher_model_path,
            teacher_revision=teacher.model_revision,
            tokenizer_path=config.teacher_model_path,
            tokenizer_revision=teacher.tokenizer_revision,
            quantization=config.teacher_quantization,
            layer_indices=layer_indices,
            tensor_path=str(tensor_path),
            prompt_length=int(teacher_output.input_ids.shape[-1]),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        append_manifest_entry(manifest_path, entry)
        completed_samples.add(sample_key)
        processed += 1

    teacher.close()

    run_summary = {
        "dataset": config.dataset,
        "teacher_model_path": config.teacher_model_path,
        "teacher_quantization": config.teacher_quantization,
        "layer_indices": layer_indices,
        "processed_examples": processed,
        "skipped_examples": skipped,
        "manifest_path": str(manifest_path),
        "tensor_dir": str(tensor_dir),
        "extractor_backend": config.extractor_backend,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps({"config": config.to_dict(), "summary": run_summary}, indent=2),
        encoding="utf-8",
    )
    return run_summary
