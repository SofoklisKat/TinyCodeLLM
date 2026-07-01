"""Load raw text datasets for causal language-model pretraining."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def _load_text_dataset(
    dataset_name: str,
    split: str,
    dataset_config: str | None,
    languages: list[str] | None,
    text_column: str,
    max_samples: int | None,
    seed: int,
) -> Dataset:
    name = dataset_name.lower()

    if "github-code" in name:
        if languages:
            print(
                f"Downloading {dataset_name} ({', '.join(languages)} only). "
                "Full Python subset is ~7.2M files / ~52 GB."
            )
            raw = load_dataset(dataset_name, languages=languages, split=split)
        else:
            print(
                f"Downloading {dataset_name} (all languages). "
                "Full dataset is ~115M files / ~324 GB compressed."
            )
            raw = load_dataset(dataset_name, split=split)
    elif dataset_config:
        raw = load_dataset(dataset_name, dataset_config, split=split)
    else:
        raw = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(raw):,} raw examples from {dataset_name}")

    if text_column not in raw.column_names:
        if "text" in raw.column_names:
            text_column = "text"
        elif "code" in raw.column_names:
            text_column = "code"
        elif "content" in raw.column_names:
            text_column = "content"
        else:
            raise ValueError(
                f"Column {text_column!r} not found in {dataset_name}. "
                f"Available columns: {raw.column_names}"
            )

    raw = raw.shuffle(seed=seed)
    if max_samples is not None and max_samples < len(raw):
        raw = raw.select(range(max_samples))

    return raw.map(
        lambda example: {"text": (example[text_column] or "").strip()},
        remove_columns=raw.column_names,
    ).filter(lambda example: bool(example["text"]))


def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    num_proc: int = 1,
) -> Dataset:
    """Tokenize text and split into fixed-length blocks for next-token prediction."""

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=num_proc if num_proc > 1 else None,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for input_ids in batch["input_ids"]:
            concatenated.extend(input_ids)
            concatenated.append(tokenizer.eos_token_id)

        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}

        chunks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * block_size for _ in chunks],
        }

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"Chunking into blocks of {block_size}",
    )


def load_clm_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str = "train",
    dataset_config: str | None = None,
    languages: list[str] | None = None,
    text_column: str = "text",
    max_samples: int | None = None,
    block_size: int = 512,
    seed: int = 42,
    cache_dir: str | None = None,
    num_proc: int = 1,
) -> Dataset:
    """Return a tokenized dataset ready for causal LM training."""
    from pathlib import Path

    if cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            from datasets import load_from_disk

            print(f"Loading preprocessed dataset from cache: {cache_path}")
            cached = load_from_disk(str(cache_path))
            print(f"Cached training blocks: {len(cached):,}")
            return cached

    text_ds = _load_text_dataset(
        dataset_name=dataset_name,
        split=split,
        dataset_config=dataset_config,
        languages=languages,
        text_column=text_column,
        max_samples=max_samples,
        seed=seed,
    )
    processed = tokenize_and_chunk(text_ds, tokenizer, block_size, num_proc=num_proc)
    print(f"Training blocks after tokenization: {len(processed):,}")

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save_to_disk(str(cache_path))
        print(f"Saved preprocessed dataset cache to {cache_path}")

    return processed
