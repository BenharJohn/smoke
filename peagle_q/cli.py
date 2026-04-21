from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from peagle_q.layers import parse_layer_indices, resolve_layer_indices


_UNEXPANDED_ENV_PATTERN = re.compile(r"\$(\w+|\{[^}]+\})")


def _expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    return value


def _load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SystemExit(f"could not read config file {config_path}: {exc}") from exc

    if not raw_text.strip():
        raise SystemExit(f"config file is empty: {config_path}")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        preview = raw_text[:120].encode("unicode_escape").decode("ascii")
        raise SystemExit(
            f"invalid JSON in config file {config_path}: {exc.msg} at "
            f"line {exc.lineno} column {exc.colno}. Preview: {preview}"
        ) from exc

    expanded = _expand_config_value(payload)
    unresolved: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, str):
            if _UNEXPANDED_ENV_PATTERN.search(value):
                unresolved.append(f"{prefix}={value}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(item, child_prefix)

    visit(expanded, "")
    if unresolved:
        details = ", ".join(unresolved)
        raise SystemExit(
            "config contains unresolved environment variables; export them before running: "
            f"{details}"
        )

    return expanded


def _merge_config(config: dict[str, Any], namespace: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key, value in vars(namespace).items():
        if key in {"command", "config"}:
            continue
        if value is not None:
            merged[key] = value
    return merged


def _require_fields(payload: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if payload.get(field) in (None, "")]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"missing required arguments: {joined}")


def _add_common_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization-aware EAGLE-3 pilot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract teacher hidden states to disk.")
    _add_common_config_flag(extract)
    extract.add_argument("--dataset", type=str)
    extract.add_argument("--teacher-model-path", type=str)
    extract.add_argument("--teacher-quantization", type=str, choices=["none", "awq", "w8a8"])
    extract.add_argument("--teacher-revision", type=str)
    extract.add_argument("--output-dir", type=str)
    extract.add_argument("--layer-indices", type=str, help="Comma-separated layer indices.")
    extract.add_argument("--resume-from", type=str)
    extract.add_argument("--max-examples", type=int)
    extract.add_argument("--max-length", type=int)
    extract.add_argument("--device", type=str)
    extract.add_argument("--dtype", type=str, choices=["float16", "float32", "bfloat16"])

    train = subparsers.add_parser("train", help="Train a draft head from an extraction manifest.")
    _add_common_config_flag(train)
    train.add_argument("--manifest", type=str)
    train.add_argument("--output-dir", type=str)
    train.add_argument("--projection-model-path", type=str)
    train.add_argument("--projection-quantization", type=str, choices=["none", "awq", "w8a8"])
    train.add_argument("--projection-revision", type=str)
    train.add_argument("--resume-from", type=str)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--grad-accum-steps", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--warmup-ratio", type=float)
    train.add_argument("--gradient-clip", type=float)
    train.add_argument("--max-examples", type=int)
    train.add_argument("--max-length", type=int)
    train.add_argument("--checkpoint-every", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--device", type=str)
    train.add_argument("--dtype", type=str, choices=["float16", "float32", "bfloat16"])
    train.add_argument("--seed", type=int)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a draft head on an MT-Bench style file.")
    _add_common_config_flag(evaluate)
    evaluate.add_argument("--benchmark-path", type=str)
    evaluate.add_argument("--verifier-model-path", type=str)
    evaluate.add_argument("--verifier-quantization", type=str, choices=["none", "awq", "w8a8"])
    evaluate.add_argument("--verifier-revision", type=str)
    evaluate.add_argument("--draft-checkpoint", type=str)
    evaluate.add_argument("--output-path", type=str)
    evaluate.add_argument("--max-questions", type=int)
    evaluate.add_argument("--draft-length", type=int)
    evaluate.add_argument("--max-new-tokens", type=int)
    evaluate.add_argument("--temperature", type=float)
    evaluate.add_argument("--device", type=str)
    evaluate.add_argument(
        "--speculation-mode",
        type=str,
        choices=["sequential", "parallel", "both"],
        help=(
            "sequential = legacy KV-cached eval (one verifier pass per draft position); "
            "parallel = real speculative decoding with K proposals verified in one pass; "
            "both = run each prompt through both paths for matched-control comparison."
        ),
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Compute per-layer drift diagnostics between two extraction manifests.",
    )
    _add_common_config_flag(diagnose)
    diagnose.add_argument("--manifest-a", type=str)
    diagnose.add_argument("--manifest-b", type=str)
    diagnose.add_argument("--output-path", type=str)
    diagnose.add_argument("--max-examples", type=int)
    diagnose.add_argument("--max-length", type=int)
    diagnose.add_argument("--label-a", type=str)
    diagnose.add_argument("--label-b", type=str)

    smoke = subparsers.add_parser("smoke", help="Run a one-prompt model and draft sanity check.")
    _add_common_config_flag(smoke)
    smoke.add_argument("--teacher-model-path", type=str)
    smoke.add_argument("--teacher-quantization", type=str, choices=["none", "awq", "w8a8"])
    smoke.add_argument("--teacher-revision", type=str)
    smoke.add_argument("--prompt", type=str, default="Explain speculative decoding in two sentences.")
    smoke.add_argument("--layer-indices", type=str)
    smoke.add_argument("--draft-checkpoint", type=str)
    smoke.add_argument("--max-new-tokens", type=int, default=32)
    smoke.add_argument("--device", type=str)

    return parser


def run_extract(namespace: argparse.Namespace) -> None:
    from peagle_q.extract import ExtractRunConfig, extract_hidden_states

    payload = _merge_config(_load_config_file(namespace.config), namespace)
    _require_fields(payload, ["dataset", "teacher_model_path", "output_dir"])
    layer_indices = parse_layer_indices(payload.get("layer_indices"))
    config = ExtractRunConfig(
        dataset=payload["dataset"],
        teacher_model_path=payload["teacher_model_path"],
        output_dir=payload["output_dir"],
        teacher_quantization=payload.get("teacher_quantization") or "none",
        teacher_revision=payload.get("teacher_revision"),
        max_examples=payload.get("max_examples"),
        max_length=payload.get("max_length") or 2048,
        layer_indices=layer_indices,
        resume_from=payload.get("resume_from"),
        dtype=payload.get("dtype"),
        device=payload.get("device"),
    )
    summary = extract_hidden_states(config)
    print(json.dumps(summary, indent=2))


def run_train(namespace: argparse.Namespace) -> None:
    from peagle_q.train import TrainRunConfig, train_draft_head

    payload = _merge_config(_load_config_file(namespace.config), namespace)
    _require_fields(payload, ["manifest", "output_dir"])
    config = TrainRunConfig(
        manifest=payload["manifest"],
        output_dir=payload["output_dir"],
        projection_model_path=payload.get("projection_model_path"),
        projection_quantization=payload.get("projection_quantization") or "none",
        projection_revision=payload.get("projection_revision"),
        resume_from=payload.get("resume_from"),
        epochs=payload.get("epochs") or 3,
        batch_size=payload.get("batch_size") or 1,
        grad_accum_steps=payload.get("grad_accum_steps") or 4,
        learning_rate=payload.get("learning_rate") or 1e-4,
        weight_decay=payload.get("weight_decay") or 0.01,
        warmup_ratio=payload.get("warmup_ratio") or 0.015,
        gradient_clip=payload.get("gradient_clip") or 0.5,
        max_examples=payload.get("max_examples"),
        max_length=payload.get("max_length") or 2048,
        checkpoint_every=payload.get("checkpoint_every") or 1000,
        num_workers=payload.get("num_workers") or 0,
        device=payload.get("device"),
        dtype=payload.get("dtype") or "float16",
        seed=payload.get("seed") if payload.get("seed") is not None else 42,
    )
    summary = train_draft_head(config)
    print(json.dumps(summary, indent=2))


def run_evaluate(namespace: argparse.Namespace) -> None:
    from peagle_q.eval import EvaluationConfig, evaluate_mt_bench

    payload = _merge_config(_load_config_file(namespace.config), namespace)
    _require_fields(
        payload,
        ["benchmark_path", "verifier_model_path", "draft_checkpoint", "output_path"],
    )
    config = EvaluationConfig(
        benchmark_path=payload["benchmark_path"],
        verifier_model_path=payload["verifier_model_path"],
        draft_checkpoint=payload["draft_checkpoint"],
        output_path=payload["output_path"],
        verifier_quantization=payload.get("verifier_quantization") or "none",
        verifier_revision=payload.get("verifier_revision"),
        max_questions=payload.get("max_questions"),
        draft_length=payload.get("draft_length") or 5,
        max_new_tokens=payload.get("max_new_tokens") or 128,
        temperature=payload.get("temperature") or 0.0,
        device=payload.get("device"),
        speculation_mode=payload.get("speculation_mode") or "sequential",
    )
    summary = evaluate_mt_bench(config)
    print(json.dumps(summary, indent=2))


def run_diagnose(namespace: argparse.Namespace) -> None:
    from peagle_q.diagnostics import DiagnoseConfig, compute_layer_diagnostics

    payload = _merge_config(_load_config_file(namespace.config), namespace)
    _require_fields(payload, ["manifest_a", "manifest_b", "output_path"])
    config = DiagnoseConfig(
        manifest_a=payload["manifest_a"],
        manifest_b=payload["manifest_b"],
        output_path=payload["output_path"],
        max_examples=payload.get("max_examples"),
        max_length=payload.get("max_length") or 2048,
        label_a=payload.get("label_a") or "teacher_a",
        label_b=payload.get("label_b") or "teacher_b",
    )
    summary = compute_layer_diagnostics(config)
    print(json.dumps(summary, indent=2))


def run_smoke(namespace: argparse.Namespace) -> None:
    from peagle_q.benchmarks import render_generation_prompt
    from peagle_q.eval import (
        _load_draft_model,
        greedy_autoregressive_generate,
        speculative_generate_local,
    )
    from peagle_q.teacher import TransformersTeacherRunner, infer_device

    payload = _merge_config(_load_config_file(namespace.config), namespace)
    _require_fields(payload, ["teacher_model_path"])
    device = infer_device(payload.get("device"))
    teacher = TransformersTeacherRunner(
        payload["teacher_model_path"],
        quantization=payload.get("teacher_quantization") or "none",
        revision=payload.get("teacher_revision"),
        device=device,
    )
    layer_indices = resolve_layer_indices(
        teacher.num_hidden_layers,
        explicit=parse_layer_indices(payload.get("layer_indices")),
    )
    try:
        raw_prompt = payload.get("prompt") or ""
        prompt_text = render_generation_prompt(
            teacher.tokenizer,
            [{"role": "user", "content": raw_prompt}],
        )
        output = teacher.forward_prompt(prompt_text, layer_indices=layer_indices)
        autoregressive = greedy_autoregressive_generate(
            teacher,
            prompt_text,
            max_new_tokens=payload.get("max_new_tokens") or 32,
            temperature=0.0,
        )
        summary: dict[str, Any] = {
            "device": device,
            "teacher_model_path": payload["teacher_model_path"],
            "teacher_quantization": payload.get("teacher_quantization") or "none",
            "raw_prompt": raw_prompt,
            "rendered_prompt_preview": prompt_text[:400],
            "prompt_length": int(output.input_ids.shape[-1]),
            "layer_indices": layer_indices,
            "hidden_state_shape": None
            if output.selected_hidden_states is None
            else list(output.selected_hidden_states.shape),
            "autoregressive": {
                "text": autoregressive["text"],
                "latency_seconds": autoregressive["latency_seconds"],
                "ttft_seconds": autoregressive["ttft_seconds"],
                "itl_seconds": autoregressive["itl_seconds"],
                "tokens_per_second": autoregressive["tokens_per_second"],
            },
        }

        if payload.get("draft_checkpoint"):
            draft_model = _load_draft_model(payload["draft_checkpoint"], device=device)
            speculative = speculative_generate_local(
                teacher,
                draft_model,
                prompt_text,
                draft_length=3,
                max_new_tokens=payload.get("max_new_tokens") or 32,
                temperature=0.0,
            )
            summary["speculative"] = {
                "tau": speculative["tau"],
                "alpha": speculative["alpha"],
                "text": speculative["text"],
            }

        print(json.dumps(summary, indent=2))
    finally:
        teacher.close()


def main() -> None:
    parser = build_parser()
    namespace = parser.parse_args()
    if namespace.command == "extract":
        run_extract(namespace)
        return
    if namespace.command == "train":
        run_train(namespace)
        return
    if namespace.command == "evaluate":
        run_evaluate(namespace)
        return
    if namespace.command == "diagnose":
        run_diagnose(namespace)
        return
    if namespace.command == "smoke":
        run_smoke(namespace)
        return
    raise SystemExit(f"unsupported command: {namespace.command}")


if __name__ == "__main__":
    main()
