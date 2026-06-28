"""Load and format coding instruction datasets for SFT."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


def _format_alpaca(example: dict[str, Any]) -> dict[str, str]:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    if inp:
        user_content = f"{instruction}\n\nInput:\n{inp}"
    else:
        user_content = instruction

    return {
        "text": (
            f"<|im_start|>user\n{user_content}\n"
            f"<|im_start|>assistant\n{output}"
        )
    }


def _format_code_feedback(example: dict[str, Any]) -> dict[str, str]:
    query = (example.get("query") or example.get("instruction") or "").strip()
    answer = (example.get("answer") or example.get("response") or "").strip()
    return {
        "text": (
            f"<|im_start|>user\n{query}\n"
            f"<|im_start|>assistant\n{answer}"
        )
    }


def _format_mbpp(example: dict[str, Any]) -> dict[str, str]:
    prompt = (example.get("prompt") or example.get("text") or "").strip()
    code = (example.get("code") or "").strip()
    return {
        "text": (
            f"<|im_start|>user\n{prompt}\n"
            f"<|im_start|>assistant\n{code}"
        )
    }


def _pick_formatter(dataset_name: str):
    name = dataset_name.lower()
    if "mbpp" in name:
        return _format_mbpp
    if "alpaca" in name or "python_code_instructions" in name:
        return _format_alpaca
    if "code-feedback" in name or "code_feedback" in name:
        return _format_code_feedback
    return _format_alpaca


def _load_raw_dataset(
    dataset_name: str,
    split: str,
    dataset_config: str | None,
):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def load_sft_dataset(
    dataset_name: str,
    split: str = "train",
    dataset_config: str | None = None,
    eval_split: str | None = None,
    max_samples: int | None = None,
    validation_split: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Load a Hugging Face dataset and return train/eval splits with `text` column."""
    formatter = _pick_formatter(dataset_name)

    if eval_split:
        train_raw = _load_raw_dataset(dataset_name, split, dataset_config)
        eval_raw = _load_raw_dataset(dataset_name, eval_split, dataset_config)

        if max_samples is not None and max_samples < len(train_raw):
            train_raw = train_raw.shuffle(seed=seed).select(range(max_samples))

        train_ds = train_raw.map(formatter, remove_columns=train_raw.column_names)
        eval_ds = eval_raw.map(formatter, remove_columns=eval_raw.column_names)
        return train_ds, eval_ds

    raw = _load_raw_dataset(dataset_name, split, dataset_config)
    if max_samples is not None and max_samples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(max_samples))

    formatted = raw.map(formatter, remove_columns=raw.column_names)

    if validation_split <= 0:
        return formatted, None

    split_ds = formatted.train_test_split(test_size=validation_split, seed=seed)
    return split_ds["train"], split_ds["test"]
