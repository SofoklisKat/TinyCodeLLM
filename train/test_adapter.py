#!/usr/bin/env python3
"""Load a trained LoRA adapter and run coding prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.model_info import print_model_spec

DEFAULT_PROMPTS = [
    "Write a Python function to reverse a string.",
    "Write a Python function to check if a number is prime.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Fix this Python function so it handles empty lists:\n\ndef average(nums):\n    return sum(nums) / len(nums)",
]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_prompts_file(path: Path) -> list[str]:
    """Load prompts from a text file. Separate prompts with a line containing only ---."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split("\n---\n")]
    return [part for part in parts if part]


def resolve_prompts(prompt_args: list[str] | None, prompts_file: Path | None) -> list[str]:
    if prompts_file is not None:
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        prompts = load_prompts_file(prompts_file)
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_file}")
        return prompts
    if prompt_args:
        return prompt_args
    return DEFAULT_PROMPTS


def build_prompt(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text.strip()}\n<|im_start|>assistant\n"


def extract_assistant_reply(text: str) -> str:
    marker = "<|im_start|>assistant\n"
    if marker in text:
        return text.split(marker, maxsplit=1)[1].strip()
    return text.strip()


def load_model(base_model: str, adapter_path: Path, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if (adapter_path / "tokenizer_config.json").exists() else base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(build_prompt(prompt), return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return extract_assistant_reply(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained LoRA adapter")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_qlora.yaml"),
        help="Training config with base model and output paths",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Adapter directory (default: <output_dir>/adapter from config)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to test; pass multiple times for several prompts",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Text file with prompts separated by a line containing only ---",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 uses greedy decoding",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    output_dir = Path(cfg["training"]["output_dir"])
    adapter_path = args.adapter or (output_dir / "adapter")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    prompts = resolve_prompts(args.prompt, args.prompts_file)

    print_model_spec(cfg, adapter_path, mode="inference")
    model, tokenizer = load_model(
        base_model=model_cfg["name"],
        adapter_path=adapter_path,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    for i, prompt in enumerate(prompts, start=1):
        print("\n" + "=" * 72)
        print(f"Prompt {i}")
        print("-" * 72)
        print(prompt)
        print("-" * 72)
        print("Response")
        print("-" * 72)
        print(generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()
