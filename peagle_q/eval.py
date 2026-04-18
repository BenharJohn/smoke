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
from peagle_q.teacher import (
    TeacherOutput,
    TransformersTeacherRunner,
    infer_device,
    save_json,
    trim_kv_cache,
)


SPECULATION_MODES = ("sequential", "parallel", "both")


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
    speculation_mode: str = "sequential"

    def __post_init__(self) -> None:
        if self.speculation_mode not in SPECULATION_MODES:
            raise ValueError(
                f"speculation_mode must be one of {SPECULATION_MODES}, "
                f"got {self.speculation_mode!r}"
            )


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    draft_config = DraftHeadConfig(**checkpoint["draft_config"])
    model = Eagle3StyleDraftHead(draft_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)

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
    """Sequential speculative decoding with a persistent KV cache.

    Implementation notes
    --------------------
    The previous version issued one full-sequence verifier forward pass per draft
    position (draft_length + 1 passes of O(L+K) work per speculative step), which
    made evaluation O(K * L) per step and produced meaningless latency numbers.

    This version maintains a single verifier KV cache across the whole generation.
    Each draft position extends the cache by exactly one token (O(1) work after
    the initial prefill). The accept rule remains greedy-equivalent: a proposal
    is accepted iff it matches the verifier's argmax at that position.

    A real parallel-verification speculative decoding pass would require the draft
    head to produce K proposals without intermediate verifier calls. The current
    draft architecture consumes verifier hidden states at each position, so the
    best we can do without redesigning the draft model is sequential proposal
    with a shared cache. This is documented as a limitation in the plan; latency
    columns from this evaluator should not be interpreted as a realistic upper
    bound on speculative-decoding speedup.
    """
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
    verifier_forward_calls = 0

    def verifier_step(
        new_ids: list[int],
        cached_length: int,
        past_kv: Any | None,
    ) -> TeacherOutput:
        nonlocal verifier_forward_calls
        verifier_forward_calls += 1
        total_length = cached_length + len(new_ids)
        attention_mask = torch.ones(1, total_length, dtype=torch.long, device=verifier.device)
        return verifier.forward_ids(
            new_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
            logits_mode="last",
            past_key_values=past_kv,
            use_cache=True,
            hidden_states_on_device=True,
        )

    start = time.perf_counter()
    first_token_elapsed: float | None = None

    prefill = verifier_step(prefix_ids, 0, None)
    if prefill.logits is None or prefill.selected_hidden_states is None:
        raise RuntimeError("verifier prefill did not return logits and hidden states")
    cache = prefill.past_key_values
    cached_length = len(prefix_ids)
    last_hidden = prefill.selected_hidden_states  # [num_taps, seq_len, hidden]
    last_logits = prefill.logits  # [1, 1, V] (logits_mode="last")

    while len(generated) < max_new_tokens:
        proposals: list[int] = []
        num_accepted = 0
        rejection_token: int | None = None
        finished = False

        for position in range(draft_length):
            verifier_prediction = int(last_logits[0, -1].argmax().item())

            draft_hidden = last_hidden[:, -1:, :].unsqueeze(0).to(device)
            draft_mask = torch.ones(1, 1, dtype=torch.long, device=device)
            proposal = draft.predict_next_token(
                draft_hidden,
                draft_mask,
                temperature=temperature,
            )
            proposals.append(proposal)
            position_attempts[position] += 1

            if proposal == verifier_prediction:
                num_accepted += 1
                position_accepts[position] += 1
                committed_token = proposal
            else:
                rejection_token = verifier_prediction
                break

            if eos_token_id is not None and committed_token == eos_token_id:
                finished = True
                break

            step_out = verifier_step([committed_token], cached_length, cache)
            if step_out.logits is None or step_out.selected_hidden_states is None:
                raise RuntimeError("verifier step did not return logits and hidden states")
            cache = step_out.past_key_values
            cached_length += 1
            last_hidden = step_out.selected_hidden_states
            last_logits = step_out.logits

        accepted_lengths.append(num_accepted)
        committed: list[int] = list(proposals[:num_accepted])

        if finished:
            pass
        elif rejection_token is not None:
            committed.append(rejection_token)
            if eos_token_id is not None and rejection_token == eos_token_id:
                finished = True
            else:
                step_out = verifier_step([rejection_token], cached_length, cache)
                if step_out.logits is None or step_out.selected_hidden_states is None:
                    raise RuntimeError("verifier step did not return logits and hidden states")
                cache = step_out.past_key_values
                cached_length += 1
                last_hidden = step_out.selected_hidden_states
                last_logits = step_out.logits
        else:
            bonus = int(last_logits[0, -1].argmax().item())
            committed.append(bonus)
            if eos_token_id is not None and bonus == eos_token_id:
                finished = True
            else:
                step_out = verifier_step([bonus], cached_length, cache)
                if step_out.logits is None or step_out.selected_hidden_states is None:
                    raise RuntimeError("verifier step did not return logits and hidden states")
                cache = step_out.past_key_values
                cached_length += 1
                last_hidden = step_out.selected_hidden_states
                last_logits = step_out.logits

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
    tau = 0.0 if not accepted_lengths else mean(accepted_lengths)
    alpha = [
        0.0 if attempts == 0 else accepts / attempts
        for attempts, accepts in zip(position_attempts, position_accepts)
    ]
    return {
        "generated_ids": generated,
        "text": verifier.tokenizer.decode(generated, skip_special_tokens=True),
        "accepted_lengths": accepted_lengths,
        "tau": tau,
        "theoretical_speedup_ideal": tau + 1,
        "alpha": alpha,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
        "verifier_forward_calls": verifier_forward_calls,
        "latency_seconds": total_time,
        "ttft_seconds": first_token_elapsed or total_time,
        "itl_seconds": 0.0 if token_count <= 1 else (total_time - (first_token_elapsed or 0.0)) / (token_count - 1),
        "tokens_per_second": 0.0 if total_time == 0 else token_count / total_time,
    }


def speculative_generate_parallel(
    verifier: TransformersTeacherRunner,
    draft: Eagle3StyleDraftHead,
    prompt_text: str,
    *,
    draft_length: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Parallel-verification speculative decoding with realistic latency.

    Per speculative step this performs **two** verifier forward passes:

    1. One pass over all ``draft_length`` draft proposals at once (cached,
       so compute is proportional to ``draft_length`` tokens, not the whole
       prefix). This is the real speculative-decoding speedup knob.
    2. One pass on the single committed rejection-or-bonus token to extend
       the KV cache and obtain ``last_hidden``/``last_logits`` for the next
       iteration.

    Compared with the sequential path (``speculative_generate_local``) this
    turns ``K + 1`` verifier forwards per step into ``2`` — genuine
    speedup when acceptance is high. The draft proposes autoregressively
    without verifier help via :meth:`Eagle3StyleDraftHead.propose_sequence`.
    """
    if temperature and temperature > 0:
        raise ValueError("parallel speculative evaluation currently supports greedy decoding only")

    encoded = verifier.encode_text(prompt_text)
    prefix_ids = encoded["input_ids"][0].tolist()
    eos_token_id = verifier.tokenizer.eos_token_id
    layer_indices = draft.config.layer_indices

    generated: list[int] = []
    accepted_lengths: list[int] = []
    position_attempts = [0 for _ in range(draft_length)]
    position_accepts = [0 for _ in range(draft_length)]
    verifier_forward_calls = 0

    def verifier_step(
        new_ids: list[int],
        cached_length: int,
        past_kv: Any | None,
        *,
        logits_mode: str,
    ) -> TeacherOutput:
        nonlocal verifier_forward_calls
        verifier_forward_calls += 1
        total_length = cached_length + len(new_ids)
        attention_mask = torch.ones(1, total_length, dtype=torch.long, device=verifier.device)
        return verifier.forward_ids(
            new_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
            logits_mode=logits_mode,
            past_key_values=past_kv,
            use_cache=True,
            hidden_states_on_device=True,
        )

    start = time.perf_counter()
    first_token_elapsed: float | None = None

    prefill = verifier_step(prefix_ids, 0, None, logits_mode="last")
    if prefill.logits is None or prefill.selected_hidden_states is None:
        raise RuntimeError("verifier prefill did not return logits and hidden states")
    cache = prefill.past_key_values
    cached_length = len(prefix_ids)
    last_hidden = prefill.selected_hidden_states  # [num_taps, L, H]
    last_logits = prefill.logits  # [1, 1, V]

    while len(generated) < max_new_tokens:
        verifier_first_prediction = int(last_logits[0, -1].argmax().item())

        # Draft proposes K tokens autoregressively (no verifier calls).
        initial_hidden = last_hidden[:, -1:, :].unsqueeze(0)  # [1, num_taps, 1, H]
        proposals, _ = draft.propose_sequence(
            initial_hidden,
            draft_length,
            temperature=temperature,
        )
        for position in range(draft_length):
            position_attempts[position] += 1

        # Parallel verify: single verifier pass on all K proposals.
        verify_out = verifier_step(proposals, cached_length, cache, logits_mode="all")
        if verify_out.logits is None or verify_out.selected_hidden_states is None:
            raise RuntimeError("verifier parallel-verify did not return logits and hidden states")
        verify_logits = verify_out.logits  # [1, K, V]

        # Verifier predictions at each proposal position:
        # - position 0: prefill's last logit (predicts token right after prefix)
        # - positions 1..K-1: verify_logits[0, i-1]
        verifier_predictions = [verifier_first_prediction]
        for index in range(draft_length - 1):
            verifier_predictions.append(int(verify_logits[0, index].argmax().item()))
        bonus_token = int(verify_logits[0, -1].argmax().item())

        num_accepted = 0
        for index in range(draft_length):
            if proposals[index] == verifier_predictions[index]:
                num_accepted += 1
                position_accepts[index] += 1
            else:
                break
        accepted_lengths.append(num_accepted)

        committed: list[int] = list(proposals[:num_accepted])
        finished = False
        for token in committed:
            if eos_token_id is not None and token == eos_token_id:
                finished = True
                break

        if not finished:
            if num_accepted < draft_length:
                extension_token = verifier_predictions[num_accepted]
            else:
                extension_token = bonus_token
            committed.append(extension_token)
            if eos_token_id is not None and extension_token == eos_token_id:
                finished = True
        else:
            extension_token = None

        if not committed:
            break

        remaining = max_new_tokens - len(generated)
        generated.extend(committed[:remaining])
        if first_token_elapsed is None:
            first_token_elapsed = time.perf_counter() - start

        if finished:
            break

        # Update the verifier KV cache so the next iteration sees exactly the
        # committed prefix. The parallel-verify pass extended the cache by K
        # positions using possibly-rejected proposals, so we trim it back to
        # the accepted-prefix length and then re-forward the single extension
        # token to advance by one valid position.
        trimmed_cache = trim_kv_cache(verify_out.past_key_values, cached_length + num_accepted)
        cached_length = cached_length + num_accepted
        step_out = verifier_step([extension_token], cached_length, trimmed_cache, logits_mode="last")
        if step_out.logits is None or step_out.selected_hidden_states is None:
            raise RuntimeError("verifier tip-extension did not return logits and hidden states")
        cache = step_out.past_key_values
        cached_length += 1
        last_hidden = step_out.selected_hidden_states
        last_logits = step_out.logits

    total_time = time.perf_counter() - start
    token_count = len(generated)
    tau = 0.0 if not accepted_lengths else mean(accepted_lengths)
    alpha = [
        0.0 if attempts == 0 else accepts / attempts
        for attempts, accepts in zip(position_attempts, position_accepts)
    ]
    return {
        "generated_ids": generated,
        "text": verifier.tokenizer.decode(generated, skip_special_tokens=True),
        "accepted_lengths": accepted_lengths,
        "tau": tau,
        "theoretical_speedup_ideal": tau + 1,
        "alpha": alpha,
        "position_attempts": position_attempts,
        "position_accepts": position_accepts,
        "verifier_forward_calls": verifier_forward_calls,
        "latency_seconds": total_time,
        "ttft_seconds": first_token_elapsed or total_time,
        "itl_seconds": 0.0 if token_count <= 1 else (total_time - (first_token_elapsed or 0.0)) / (token_count - 1),
        "tokens_per_second": 0.0 if total_time == 0 else token_count / total_time,
        "speculation_mode": "parallel",
    }


def _run_speculative(
    mode: str,
    verifier: TransformersTeacherRunner,
    draft: Eagle3StyleDraftHead,
    prompt_text: str,
    *,
    draft_length: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    if mode == "sequential":
        return speculative_generate_local(
            verifier,
            draft,
            prompt_text,
            draft_length=draft_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    if mode == "parallel":
        return speculative_generate_parallel(
            verifier,
            draft,
            prompt_text,
            draft_length=draft_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    raise ValueError(f"unsupported speculation mode: {mode!r}")


@dataclass(slots=True)
class _ModeAggregator:
    mode: str
    draft_length: int
    global_attempts: list[int]
    global_accepts: list[int]
    tau_values: list[float]
    speedups: list[float]
    tokens_per_second: list[float]
    ttft: list[float]
    itl: list[float]
    verifier_forward_calls: list[int]

    @classmethod
    def create(cls, mode: str, draft_length: int) -> "_ModeAggregator":
        return cls(
            mode=mode,
            draft_length=draft_length,
            global_attempts=[0 for _ in range(draft_length)],
            global_accepts=[0 for _ in range(draft_length)],
            tau_values=[],
            speedups=[],
            tokens_per_second=[],
            ttft=[],
            itl=[],
            verifier_forward_calls=[],
        )

    def observe(self, speculative_result: dict[str, Any], ar_tokens_per_second: float) -> float:
        self.tau_values.append(float(speculative_result["tau"]))
        self.tokens_per_second.append(float(speculative_result["tokens_per_second"]))
        self.ttft.append(float(speculative_result["ttft_seconds"]))
        self.itl.append(float(speculative_result["itl_seconds"]))
        self.verifier_forward_calls.append(int(speculative_result["verifier_forward_calls"]))
        speedup = 0.0
        if ar_tokens_per_second > 0:
            speedup = float(speculative_result["tokens_per_second"]) / float(ar_tokens_per_second)
        self.speedups.append(speedup)
        for index, attempts in enumerate(speculative_result["position_attempts"]):
            self.global_attempts[index] += attempts
        for index, accepts in enumerate(speculative_result["position_accepts"]):
            self.global_accepts[index] += accepts
        return speedup

    def summarise(self) -> dict[str, Any]:
        alpha = [
            0.0 if attempts == 0 else accepts / attempts
            for attempts, accepts in zip(self.global_attempts, self.global_accepts)
        ]
        tau = 0.0 if not self.tau_values else mean(self.tau_values)
        return {
            "tau": tau,
            "theoretical_speedup_ideal": tau + 1,
            "alpha": alpha,
            "speculative_tokens_per_second": 0.0 if not self.tokens_per_second else mean(self.tokens_per_second),
            "speculative_ttft_seconds": 0.0 if not self.ttft else mean(self.ttft),
            "speculative_itl_seconds": 0.0 if not self.itl else mean(self.itl),
            "speedup_vs_ar": 0.0 if not self.speedups else mean(self.speedups),
            "mean_verifier_forward_calls_per_turn": (
                0.0
                if not self.verifier_forward_calls
                else mean(self.verifier_forward_calls)
            ),
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

    if config.speculation_mode == "both":
        active_modes: list[str] = ["sequential", "parallel"]
    else:
        active_modes = [config.speculation_mode]
    aggregators: dict[str, _ModeAggregator] = {
        mode: _ModeAggregator.create(mode, config.draft_length) for mode in active_modes
    }

    per_turn_records: list[dict[str, Any]] = []
    # The `speculative_conversation` history is driven by the PRIMARY mode so the
    # assistant turns that feed into the next user turn are deterministic. In
    # `both` mode we use sequential as the conversation driver, and parallel
    # re-runs each prompt for latency measurement only.
    primary_mode = "sequential" if "sequential" in active_modes else active_modes[0]

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

                mode_results: dict[str, dict[str, Any]] = {}
                for mode in active_modes:
                    mode_result = _run_speculative(
                        mode,
                        verifier,
                        draft,
                        speculative_prompt_text,
                        draft_length=config.draft_length,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                    )
                    aggregators[mode].observe(mode_result, ar_result["tokens_per_second"])
                    mode_results[mode] = mode_result

                primary_result = mode_results[primary_mode]
                ar_conversation.append({"role": "assistant", "content": ar_result["text"]})
                speculative_conversation.append(
                    {"role": "assistant", "content": primary_result["text"]}
                )

                record = {
                    "question_id": question.question_id,
                    "category": question.category,
                    "turn_index": turn_index,
                    "autoregressive_prompt_text": ar_prompt_text,
                    "speculative_prompt_text": speculative_prompt_text,
                    "autoregressive": ar_result,
                    "speculation_mode": config.speculation_mode,
                    "primary_mode": primary_mode,
                }
                if len(active_modes) == 1:
                    record["speculative"] = mode_results[active_modes[0]]
                else:
                    record["speculative_by_mode"] = mode_results
                    # Keep `speculative` key populated with the primary mode result so downstream
                    # parsers that predate `speculation_mode=both` still work.
                    record["speculative"] = primary_result
                per_turn_records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        verifier.close()

    summary: dict[str, Any] = {
        "benchmark_path": config.benchmark_path,
        "verifier_model_path": config.verifier_model_path,
        "verifier_quantization": config.verifier_quantization,
        "draft_checkpoint": config.draft_checkpoint,
        "questions": len(questions),
        "turns": len(per_turn_records),
        "speculation_mode": config.speculation_mode,
        "primary_mode": primary_mode,
        "per_turn_records_path": str(records_path),
    }
    if len(active_modes) == 1:
        summary.update(aggregators[active_modes[0]].summarise())
    else:
        summary["by_mode"] = {mode: aggregators[mode].summarise() for mode in active_modes}
        # Flatten primary-mode metrics to the top level for backwards compatibility.
        summary.update(aggregators[primary_mode].summarise())
    save_json(output_path, summary)
    return summary
