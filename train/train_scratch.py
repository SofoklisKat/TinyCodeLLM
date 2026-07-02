#!/usr/bin/env python3
"""Pretrain the ~30M TinyCode decoder from scratch on Hugging Face text/code data.

Saves a Hugging Face Qwen2 checkpoint that can later be used as the base model
for QLoRA fine-tuning (set model.name to the output directory in train_qlora.yaml).

Example:
    python -m train.train_scratch --config configs/train_scratch_30m.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from train.model import (
    build_tinycode_model,
    count_parameters,
    export_to_huggingface,
    patch_qwen2_config,
)
from train.scratch_dataset import build_training_dataloader


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    output = model(input_ids, attention_mask=attention_mask)
    return output.logits if hasattr(output, "logits") else output


def parse_resume_step(resume_path: Path) -> int:
    name = resume_path.name
    if name.startswith("checkpoint-"):
        return int(name.split("-", maxsplit=1)[1])
    return 0


def load_model_for_training(
    device: torch.device,
    resume_from: Path | None,
    model_size: str = "30m",
):
    if resume_from is None:
        model = build_tinycode_model(model_size)
        model.to(device)
        start_step = 0
        print(f"Model size: {model_size}")
        print(f"Parameters: {count_parameters(model):,}")
        return model, start_step

    if not resume_from.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")

    from transformers import AutoModelForCausalLM

    print(f"Resuming weights from: {resume_from}")
    model = AutoModelForCausalLM.from_pretrained(resume_from)
    patch_qwen2_config(model.config, default_rope_theta=10_000.0)
    model.to(device)
    start_step = parse_resume_step(resume_from)
    print(f"Resuming from step: {start_step}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, start_step


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
    parser = argparse.ArgumentParser(description="Pretrain TinyCode decoder from scratch (10m / 15m / 30m)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_scratch_30m.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint directory to resume from (e.g. outputs/tinycode-30m/checkpoint-185000)",
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

    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    print(f"Free disk space at output_dir: {free_gb:.1f} GB")
    if not ds_cfg.get("streaming", False) and free_gb < 120:
        print(
            "Warning: full download + dataset-cache for Python github-code often needs "
            "~100-120 GB. Set dataset.streaming: true in the config to avoid that."
        )

    tokenizer_name = model_cfg.get("tokenizer", "gpt2")
    resume_from = args.resume
    if resume_from is None and train_cfg.get("resume_from"):
        resume_from = Path(train_cfg["resume_from"])

    if resume_from is not None:
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {ds_cfg['name']}")
    dataloader = build_training_dataloader(
        dataset_name=ds_cfg["name"],
        tokenizer=tokenizer,
        batch_size=train_cfg["per_device_train_batch_size"],
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
        streaming=bool(ds_cfg.get("streaming", False)),
        collate_fn=collate_batch,
    )

    model, start_step = load_model_for_training(
        device,
        resume_from,
        model_size=str(model_cfg.get("size", "30m")),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    use_fp16 = bool(train_cfg.get("fp16", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    num_epochs = int(train_cfg["num_train_epochs"])
    logging_steps = int(train_cfg.get("logging_steps", 50))
    save_steps = int(train_cfg.get("save_steps", 500))
    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)

    global_step = start_step
    running_loss = 0.0
    start_time = time.time()
    model.train()

    if start_step > 0:
        print(
            "Note: streaming restarts from the beginning of the dataset. "
            f"Training continues counting from step {start_step}."
        )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                logits = forward_logits(model, input_ids, attention_mask)
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
