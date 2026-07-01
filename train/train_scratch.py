#!/usr/bin/env python3
"""Pretrain the ~30M TinyCode decoder from scratch on Hugging Face text/code data.

Saves a Hugging Face Qwen2 checkpoint that can later be used as the base model
for QLoRA fine-tuning (set model.name to the output directory in train_qlora.yaml).

Example:
    python -m train.train_scratch --config configs/train_scratch_30m.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train.model import count_parameters, export_to_huggingface, tinycode_30m
from train.scratch_dataset import load_clm_dataset


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def collate_batch(features: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
    attention_mask = torch.tensor(
        [feature["attention_mask"] for feature in features],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain TinyCode 30M decoder from scratch")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_scratch_30m.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    output_dir = Path(model_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_name = model_cfg.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {ds_cfg['name']}")
    dataset = load_clm_dataset(
        dataset_name=ds_cfg["name"],
        tokenizer=tokenizer,
        split=ds_cfg.get("split", "train"),
        dataset_config=ds_cfg.get("config"),
        languages=ds_cfg.get("languages"),
        text_column=ds_cfg.get("text_column", "text"),
        max_samples=ds_cfg.get("max_samples"),
        block_size=train_cfg["block_size"],
        seed=train_cfg.get("seed", 42),
        cache_dir=ds_cfg.get("cache_dir"),
        num_proc=int(train_cfg.get("preprocessing_num_proc", 1)),
        trust_remote_code=bool(ds_cfg.get("trust_remote_code", False)),
    )
    if len(dataset) == 0:
        raise RuntimeError("No training blocks were produced. Increase max_samples or block_size.")

    print(f"Training blocks: {len(dataset)}")
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=True,
    )

    model = tinycode_30m()
    model.to(device)
    print(model)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    use_fp16 = bool(train_cfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = int(train_cfg["num_train_epochs"])
    logging_steps = int(train_cfg.get("logging_steps", 50))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)

    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = causal_lm_loss(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            epoch_loss += loss.item() * grad_accum
            epoch_steps += 1

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    elapsed = time.time() - start_time
                    print(
                        f"step={global_step} epoch={epoch + 1}/{num_epochs} "
                        f"loss={avg_loss:.4f} elapsed={elapsed:.1f}s"
                    )
                    running_loss = 0.0

                if global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    export_to_huggingface(model, checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                if max_steps is not None and global_step >= max_steps:
                    break

        if step % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        epoch_avg = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1} complete. avg_loss={epoch_avg:.4f}")

        if max_steps is not None and global_step >= max_steps:
            break

    final_dir = output_dir / "final"
    export_to_huggingface(model, final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Saved model to {final_dir}")
    print(
        "Next step: point QLoRA config model.name to this directory, e.g.\n"
        f"  model:\n    name: {final_dir}"
    )


if __name__ == "__main__":
    main()
