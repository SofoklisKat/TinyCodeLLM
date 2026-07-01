"""Load raw text datasets for causal language-model pretraining."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


def _load_raw_dataset(
    dataset_name: str,
    split: str,
    dataset_config: str | None,
    languages: list[str] | None,
    trust_remote_code: bool,
) -> Dataset:
    load_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    name = dataset_name.lower()

    if "github-code" in name:
        if languages:
            print(
                f"Downloading {dataset_name} ({', '.join(languages)} only). "
                "Full Python subset is ~7.2M files / ~52 GB."
            )
            raw = load_dataset(
                dataset_name,
                languages=languages,
                split=split,
                **load_kwargs,
            )
        else:
            print(
                f"Downloading {dataset_name} (all languages). "
                "Full dataset is ~115M files / ~324 GB compressed."
            )
            raw = load_dataset(dataset_name, split=split, **load_kwargs)
    elif dataset_config:
        raw = load_dataset(dataset_name, dataset_config, split=split, **load_kwargs)
    else:
        raw = load_dataset(dataset_name, split=split, **load_kwargs)

    return raw


def _load_text_dataset(
    dataset_name: str,
    split: str,
    dataset_config: str | None,
    languages: list[str] | None,
    text_column: str,
    max_samples: int | None,
    seed: int,
    trust_remote_code: bool = False,
) -> Dataset:
    raw = _load_raw_dataset(
        dataset_name=dataset_name,
        split=split,
        dataset_config=dataset_config,
        languages=languages,
        trust_remote_code=trust_remote_code,
    )

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
    trust_remote_code: bool = False,
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
        trust_remote_code=trust_remote_code,
    )
    processed = tokenize_and_chunk(text_ds, tokenizer, block_size, num_proc=num_proc)
    print(f"Training blocks after tokenization: {len(processed):,}")

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        processed.save_to_disk(str(cache_path))
        print(f"Saved preprocessed dataset cache to {cache_path}")

    return processed


class StreamingCLMDataset(IterableDataset):
    """Stream text from Hugging Face and yield fixed-size token blocks.

    Avoids downloading the full dataset and writing a large on-disk cache.
    Recommended for codeparrot/github-code on disks under ~120 GB free.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        *,
        split: str = "train",
        dataset_config: str | None = None,
        languages: list[str] | None = None,
        text_column: str = "text",
        block_size: int = 512,
        max_samples: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.dataset_config = dataset_config
        self.languages = languages
        self.text_column = text_column
        self.block_size = block_size
        self.max_samples = max_samples
        self.trust_remote_code = trust_remote_code

    def _resolve_text_column(self, columns: list[str]) -> str:
        if self.text_column in columns:
            return self.text_column
        if "text" in columns:
            return "text"
        if "code" in columns:
            return "code"
        if "content" in columns:
            return "content"
        raise ValueError(
            f"Column {self.text_column!r} not found in {self.dataset_name}. "
            f"Available columns: {columns}"
        )

    def _open_stream(self):
        load_kwargs: dict[str, Any] = {"streaming": True, "split": self.split}
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        name = self.dataset_name.lower()
        if "github-code" in name:
            lang_note = f" ({', '.join(self.languages)} only)" if self.languages else ""
            print(
                f"Streaming {self.dataset_name}{lang_note}. "
                "No full local download or dataset-cache required."
            )
            if self.languages:
                return load_dataset(self.dataset_name, languages=self.languages, **load_kwargs)
            return load_dataset(self.dataset_name, **load_kwargs)
        if self.dataset_config:
            return load_dataset(self.dataset_name, self.dataset_config, **load_kwargs)
        return load_dataset(self.dataset_name, **load_kwargs)

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        stream = self._open_stream()
        text_column = self._resolve_text_column(stream.column_names)
        token_buffer: list[int] = []
        seen_examples = 0

        for example in stream:
            if self.max_samples is not None and seen_examples >= self.max_samples:
                break

            text = (example.get(text_column) or "").strip()
            if not text:
                continue

            seen_examples += 1
            token_buffer.extend(self.tokenizer.encode(text))
            token_buffer.append(self.tokenizer.eos_token_id)

            while len(token_buffer) >= self.block_size:
                chunk = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * self.block_size,
                }


def build_training_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
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
    trust_remote_code: bool = False,
    streaming: bool = False,
    collate_fn=None,
):
    from torch.utils.data import DataLoader

    if streaming:
        stream_ds = StreamingCLMDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=split,
            dataset_config=dataset_config,
            languages=languages,
            text_column=text_column,
            block_size=block_size,
            max_samples=max_samples,
            trust_remote_code=trust_remote_code,
        )
        return DataLoader(
            stream_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )

    dataset = load_clm_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        dataset_config=dataset_config,
        languages=languages,
        text_column=text_column,
        max_samples=max_samples,
        block_size=block_size,
        seed=seed,
        cache_dir=cache_dir,
        num_proc=num_proc,
        trust_remote_code=trust_remote_code,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training blocks were produced. Increase max_samples or block_size.")

    print(f"Training blocks: {len(dataset):,}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
