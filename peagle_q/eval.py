from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any

import torch

from peagle_q.benchmarks import load_mt_bench, render_generation_prompt
from peagle_q.model import DraftHeadConfig, Eagle3StyleDraftHead
from peagle_q.teacher import TransformersTeacherRunner, infer_device, save_json


@dataclass(slots=True)
class EvaluationConfig:
    benchmark_path: str
    verifier_model_path: str
    draft_checkpoint: str
    output_path: str
    verifier_quantization: str = "none"
    verifier_revision: str | None = None
    max_questions: int | None = None
    draft_length: int = 5
    max_new_tokens: int = 128
    temperature: float = 0.0
    device: str | None = None


def _pick_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature and temperature > 0:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())
    return int(torch.argmax(logits).item())


def _load_draft_model(
    checkpoint_path: str,
    *,
    device: str,
) -> Eagle3StyleDraftHead:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    draft_config = DraftHeadConfig(**checkpoint["draft_config"])
    model = Eagle3StyleDraftHead(draft_config)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    projection_model_path = draft_config.projection_model_path or draft_config.teacher_model_path
    projection_runner = TransformersTeacherRunner(
        projection_model_path,
        quantization=draft_config.projection_quantization,
        device="cpu",
        dtype_name="float32",
    )
    try:
        weight, bias = projection_runner.get_output_projection()
    finally:
        projection_runner.close()

    model.attach_projection(weight, bias)
    model = model.to(device)
    model.eval()
    return model


def greedy_autoregressive_generate(
    verifier: TransformersTeacherRunner,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    encoded = verifier.encode_text(prompt_text)
    prefix_ids = encoded["input_ids"][0].tolist()
    eos_token_id = verifier.tokenizer.eos_token_id

    generated: list[int] = []
    start = time.perf_counter()
    first_token_elapsed: float | None = None

    for _ in range(max_new_tokens):
        output = verifier.forward_ids(
            prefix_ids + generated,
            layer_indices=None,
            logits_mode="last",
        )
        if output.logits is None:
            raise RuntimeError("verifier did not return logits for autoregressive generation")
        next_token = _pick_token(output.logits[0, -1], temperature)
        generated.append(next_token)
        if first_token_elapsed is None:
            first_token_elapsed = time.perf_counter() - start
        if eos_token_id is not None and next_token == eos_token_id:
            break

    total_time = time.perf_counter() - start
    token_count = len(generated)
    return {
        "generated_ids": generated,
        "text": verifier.tokenizer.decode(generated, skip_special_tokens=True),
        "latency_seconds": total_time,
        "ttft_seconds": first_token_elapsed or total_time,
        "itl_seconds": 0.0 if token_count <= 1 else (total_time - (first_token_elapsed or 0.0)) / (token_count - 1),
        "tokens_per_second": 0.0 if total_time == 0 else token_count / total_time,
    }


def speculative_generate_local(
    verifier: TransformersTeacherRunner,
    draft: Eagle3StyleDraftHead,
    prompt_text: str,
    *,
    draft_length: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    if temperature and temperature > 0:
        raise ValueError("local speculative evaluation currently supports greedy decoding only")

    encoded = verifier.encode_text(prompt_text)
    prefix_ids = encoded["input_ids"][0].tolist()
    eos_token_id = verifier.tokenizer.eos_token_id
    layer_indices = draft.config.layer_indices
    device = next(draft.parameters()).device

    generated: list[int] = []
    accepted_lengths: list[int] = []
    position_attempts = [0 for _ in range(draft_length)]
    position_accepts = [0 for _ in range(draft_length)]

    start = time.perf_counter()
    first_token_elapsed: float | None = None

    while len(generated) < max_new_tokens:
        working_ids = prefix_ids + generated
        proposals: list[int] = []

        for position in range(draft_length):
            teacher_output = verifier.forward_ids(
                working_ids,
                layer_indices=layer_indices,
                logits_mode="none",
            )
            hidden_states = teacher_output.selected_hidden_states
            if hidden_states is None:
                raise RuntimeError("draft proposal requested hidden states but none were returned")

            draft_hidden = hidden_states.unsqueeze(0).to(device)
            draft_mask = teacher_output.attention_mask.to(device)
            proposal = draft.predict_next_token(
                draft_hidden,
                draft_mask,
                temperature=temperature,
            )
            proposals.append(proposal)
            position_attempts[position] += 1
            working_ids = working_ids + [proposal]
            if eos_token_id is not None and proposal == eos_token_id:
                break

        verification = verifier.forward_ids(
            prefix_ids + generated + proposals,
            layer_indices=None,
            logits_mode="full",
        )
        if verification.logits is None:
            raise RuntimeError("verifier did not return logits for speculative verification")
        predicted = verification.logits.argmax(dim=-1)[0]
        base_index = len(prefix_ids) + len(generated) - 1

        accepted = 0
        committed: list[int] = []
        for position, proposal in enumerate(proposals):
            verifier_token = int(predicted[base_index + position].item())
            if verifier_token != proposal:
                break
            accepted += 1
            position_accepts[position] += 1
            committed.append(proposal)
            if eos_token_id is not None and verifier_token == eos_token_id:
                break

        accepted_lengths.append(accepted)
        finished = False
        if committed and eos_token_id is not None and committed[-1] == eos_token_id:
            finished = True
        elif accepted < len(proposals):
            rejection_token = int(predicted[base_index + accepted].item())
            committed.append(rejection_token)
            if eos_token_id is not None and rejection_token == eos_token_id:
                finished = True
        else:
            bonus_index = base_index + len(proposals)
            if bonus_index < predicted.shape[0]:
                bonus_token = int(predicted[bonus_index].item())
                committed.append(bonus_token)
                if eos_token_id is not None and bonus_token == eos_token_id:
                    finished = True

        if not committed:
            break

        remaining = max_new_tokens - len(generated)
        generated.extend(committed[:remaining])
        if first_token_elapsed is None:
            first_token_elapsed = time.perf_counter() - start
        if finished:
            break

    total_time = time.perf_counter() - start
    token_count = len(generated)
    alpha = [
        0.0 if attempts == 0 else accepts / attempts
        for attempts, accepts in zip(position_attempts, position_accepts)
    ]
    return {
        "generated_ids": generated,
        "text": verifier.tokenizer.decode(generated, skip_special_tokens=True),
        "accepted_lengths": accepted_lengths,
        "tau": 0.0 if not accepted_lengths else mean(accepted_lengths),
        "alpha": alpha,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
        "latency_seconds": total_time,
        "ttft_seconds": first_token_elapsed or total_time,
        "itl_seconds": 0.0 if token_count <= 1 else (total_time - (first_token_elapsed or 0.0)) / (token_count - 1),
        "tokens_per_second": 0.0 if total_time == 0 else token_count / total_time,
    }


def evaluate_mt_bench(config: EvaluationConfig) -> dict[str, Any]:
    device = infer_device(config.device)
    verifier = TransformersTeacherRunner(
        config.verifier_model_path,
        quantization=config.verifier_quantization,
        revision=config.verifier_revision,
        device=device,
    )
    draft = _load_draft_model(config.draft_checkpoint, device=device)
    questions = load_mt_bench(config.benchmark_path, max_questions=config.max_questions)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_path = output_path.with_suffix(".jsonl")

    per_turn_records: list[dict[str, Any]] = []
    global_attempts = [0 for _ in range(config.draft_length)]
    global_accepts = [0 for _ in range(config.draft_length)]
    tau_values: list[float] = []
    speedups: list[float] = []
    speculative_tokens_per_second: list[float] = []
    speculative_ttft: list[float] = []
    speculative_itl: list[float] = []

    try:
        for question in questions:
            ar_conversation: list[dict[str, str]] = []
            speculative_conversation: list[dict[str, str]] = []
            for turn_index, user_turn in enumerate(question.turns):
                ar_conversation.append({"role": "user", "content": user_turn})
                speculative_conversation.append({"role": "user", "content": user_turn})
                ar_prompt_text = render_generation_prompt(verifier.tokenizer, ar_conversation)
                speculative_prompt_text = render_generation_prompt(
                    verifier.tokenizer,
                    speculative_conversation,
                )

                ar_result = greedy_autoregressive_generate(
                    verifier,
                    ar_prompt_text,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
                speculative_result = speculative_generate_local(
                    verifier,
                    draft,
                    speculative_prompt_text,
                    draft_length=config.draft_length,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )

                ar_conversation.append({"role": "assistant", "content": ar_result["text"]})
                speculative_conversation.append(
                    {"role": "assistant", "content": speculative_result["text"]}
                )

                tau_values.append(float(speculative_result["tau"]))
                speculative_tokens_per_second.append(float(speculative_result["tokens_per_second"]))
                speculative_ttft.append(float(speculative_result["ttft_seconds"]))
                speculative_itl.append(float(speculative_result["itl_seconds"]))
                speedup = 0.0
                if ar_result["tokens_per_second"] > 0:
                    speedup = float(speculative_result["tokens_per_second"]) / float(
                        ar_result["tokens_per_second"]
                    )
                speedups.append(speedup)
                for index, attempts in enumerate(speculative_result["position_attempts"]):
                    global_attempts[index] += attempts
                for index, accepts in enumerate(speculative_result["position_accepts"]):
                    global_accepts[index] += accepts

                record = {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_index": turn_index,
                    "autoregressive_prompt_text": ar_prompt_text,
                    "speculative_prompt_text": speculative_prompt_text,
                    "speculative": speculative_result,
                    "autoregressive": ar_result,
                    "speedup": speedup,
                }
                per_turn_records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        verifier.close()

    alpha = [
        0.0 if attempts == 0 else accepts / attempts
        for attempts, accepts in zip(global_attempts, global_accepts)
    ]
    summary = {
        "benchmark_path": config.benchmark_path,
        "verifier_model_path": config.verifier_model_path,
        "verifier_quantization": config.verifier_quantization,
        "draft_checkpoint": config.draft_checkpoint,
        "questions": len(questions),
        "turns": len(per_turn_records),
        "tau": 0.0 if not tau_values else mean(tau_values),
        "alpha": alpha,
        "speculative_tokens_per_second": 0.0 if not speculative_tokens_per_second else mean(speculative_tokens_per_second),
        "speculative_ttft_seconds": 0.0 if not speculative_ttft else mean(speculative_ttft),
        "speculative_itl_seconds": 0.0 if not speculative_itl else mean(speculative_itl),
        "speedup_vs_ar": 0.0 if not speedups else mean(speedups),
        "per_turn_records_path": str(records_path),
    }
    save_json(output_path, summary)
    return summary
