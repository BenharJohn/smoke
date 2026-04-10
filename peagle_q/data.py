from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sharegpt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record:
        return [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in record["messages"]
        ]

    if "conversations" in record:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        messages: list[dict[str, str]] = []
        for item in record["conversations"]:
            source_role = str(item["from"]).strip().lower()
            role = role_map.get(source_role, source_role)
            messages.append({"role": role, "content": str(item["value"])})
        return messages

    raise ValueError("record does not contain messages or conversations")


def record_prompt_id(record: dict[str, Any], dataset_index: int) -> str:
    for key in ("prompt_id", "id", "question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"sample-{dataset_index:06d}"


def render_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    return "\n".join(rendered)


def build_training_prompt(record: dict[str, Any], tokenizer: Any) -> str:
    if "prompt" in record:
        return str(record["prompt"])
    if "text" in record:
        return str(record["text"])
    messages = normalize_sharegpt_messages(record)
    return render_messages(messages, tokenizer)
