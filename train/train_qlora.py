#!/usr/bin/env python3
"""QLoRA fine-tuning entrypoint for TinyCodeLLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from train.dataset import load_sft_dataset


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_bnb_config(compute_dtype: torch.dtype = torch.float16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune TinyCodeLLM")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_qlora.yaml"),
        help="Path to YAML training config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit matmul uses fp16; training runs in fp32 (no AMP GradScaler).
    compute_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=build_bnb_config(compute_dtype),
        device_map="auto",
        dtype=compute_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    # fp16 AMP GradScaler does not support bf16 grads; keep trainable (LoRA)
    # params in fp32 so unscaling works on Turing-class GPUs (e.g. RTX 2080).
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    print(f"Loading dataset: {ds_cfg['name']}")
    train_ds, eval_ds = load_sft_dataset(
        dataset_name=ds_cfg["name"],
        split=ds_cfg.get("split", "train"),
        dataset_config=ds_cfg.get("config"),
        eval_split=ds_cfg.get("eval_split"),
        max_samples=ds_cfg.get("max_samples"),
        validation_split=ds_cfg.get("validation_split", 0.05),
        seed=train_cfg.get("seed", 42),
    )
    print(f"Train samples: {len(train_ds)}")
    if eval_ds is not None:
        print(f"Eval samples: {len(eval_ds)}")

    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", False))
    print(f"Training precision: fp16={use_fp16}, bf16={use_bf16}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 100) if eval_ds is not None else None,
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        report_to="none",
        seed=train_cfg.get("seed", 42),
        max_length=train_cfg["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
