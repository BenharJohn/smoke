from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from peagle_q.dataset import HiddenStateDataset, hidden_state_collate
from peagle_q.manifest import load_manifest
from peagle_q.model import DraftHeadConfig, Eagle3StyleDraftHead
from peagle_q.teacher import TransformersTeacherRunner, infer_device, save_json


@dataclass(slots=True)
class TrainRunConfig:
    manifest: str
    output_dir: str
    projection_model_path: str | None = None
    projection_quantization: str = "none"
    projection_revision: str | None = None
    resume_from: str | None = None
    epochs: int = 3
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.015
    gradient_clip: float = 0.5
    max_examples: int | None = None
    max_length: int = 2048
    checkpoint_every: int = 1000
    num_workers: int = 0
    device: str | None = None
    dtype: str = "float16"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_next_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous().bool()
    active_logits = shifted_logits[shifted_mask]
    active_labels = shifted_labels[shifted_mask]
    return F.cross_entropy(active_logits, active_labels)


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def _load_projection_weights(
    projection_model_path: str,
    *,
    quantization: str,
    revision: str | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    runner = TransformersTeacherRunner(
        projection_model_path,
        quantization=quantization,
        revision=revision,
        device="cpu",
        dtype_name="float32",
    )
    try:
        return runner.get_output_projection()
    finally:
        runner.close()


def _save_checkpoint(
    path: Path,
    *,
    model: Eagle3StyleDraftHead,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    global_step: int,
    epoch: int,
    best_loss: float,
    config: TrainRunConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_loss": best_loss,
            "train_config": config.to_dict(),
            "draft_config": model.config.to_dict(),
        },
        path,
    )


def train_draft_head(config: TrainRunConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    dataset = HiddenStateDataset(
        config.manifest,
        max_examples=config.max_examples,
        max_length=config.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=hidden_state_collate,
    )
    if len(dataset) == 0:
        raise ValueError("manifest did not produce any training examples")

    manifest_entries = load_manifest(config.manifest)
    first_entry = manifest_entries[0]
    projection_model_path = config.projection_model_path or first_entry.teacher_model_path
    projection_weight, projection_bias = _load_projection_weights(
        projection_model_path,
        quantization=config.projection_quantization,
        revision=config.projection_revision,
    )

    sample = dataset[0]
    draft_config = DraftHeadConfig(
        hidden_size=int(sample["hidden_states"].shape[-1]),
        num_tap_layers=int(sample["hidden_states"].shape[0]),
        vocab_size=int(projection_weight.shape[0]),
        layer_indices=list(first_entry.layer_indices),
        teacher_model_path=first_entry.teacher_model_path,
        teacher_quantization=first_entry.quantization,
        projection_model_path=projection_model_path,
        projection_quantization=config.projection_quantization,
    )
    model = Eagle3StyleDraftHead(draft_config)
    model.attach_projection(projection_weight, projection_bias)
    device = infer_device(config.device)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    total_optimizer_steps = max(
        1,
        (len(dataloader) * config.epochs + config.grad_accum_steps - 1)
        // config.grad_accum_steps,
    )
    scheduler = build_lr_scheduler(optimizer, total_optimizer_steps, config.warmup_ratio)

    global_step = 0
    optimizer_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if config.resume_from:
        checkpoint = torch.load(config.resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        model.attach_projection(projection_weight, projection_bias)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        global_step = int(checkpoint["global_step"])
        optimizer_step = global_step // max(1, config.grad_accum_steps)
        start_epoch = int(checkpoint["epoch"])
        best_loss = float(checkpoint["best_loss"])

    save_json(output_dir / "train_config.json", {"config": config.to_dict(), "draft_config": draft_config.to_dict()})

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden_states = batch["hidden_states"].to(device)

            logits = model(hidden_states, attention_mask)
            loss = compute_next_token_loss(logits, input_ids, attention_mask)
            (loss / config.grad_accum_steps).backward()
            running_loss += float(loss.item())
            global_step += 1

            should_step = global_step % config.grad_accum_steps == 0 or batch_index == len(dataloader) - 1
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                average_loss = running_loss / max(1, config.grad_accum_steps)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "optimizer_step": optimizer_step,
                                "loss": average_loss,
                                "learning_rate": scheduler.get_last_lr()[0],
                            }
                        )
                        + "\n"
                    )
                running_loss = 0.0

                if average_loss < best_loss:
                    best_loss = average_loss
                    _save_checkpoint(
                        checkpoints_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        best_loss=best_loss,
                        config=config,
                    )

                if optimizer_step % config.checkpoint_every == 0:
                    _save_checkpoint(
                        checkpoints_dir / f"step_{optimizer_step:06d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        best_loss=best_loss,
                        config=config,
                    )
                    _save_checkpoint(
                        checkpoints_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        best_loss=best_loss,
                        config=config,
                    )

        _save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch + 1,
            best_loss=best_loss,
            config=config,
        )

    summary = {
        "manifest": config.manifest,
        "output_dir": str(output_dir),
        "examples": len(dataset),
        "epochs": config.epochs,
        "best_loss": best_loss,
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "last_checkpoint": str(checkpoints_dir / "last.pt"),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
