from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peagle_q.data import iter_jsonl


@dataclass(slots=True)
class BenchmarkQuestion:
    question_id: str
    turns: list[str]
    category: str | None = None


def load_mt_bench(path: str | Path, max_questions: int | None = None) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for record in iter_jsonl(path):
        question = BenchmarkQuestion(
            question_id=str(record.get("question_id") or record.get("id") or f"q-{len(questions):04d}"),
            turns=[str(turn) for turn in record["turns"]],
            category=None if record.get("category") is None else str(record["category"]),
        )
        questions.append(question)
        if max_questions is not None and len(questions) >= max_questions:
            break
    return questions


def render_generation_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)
