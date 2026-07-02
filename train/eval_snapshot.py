#!/usr/bin/env python3
"""Evaluate a scratch-pretrain snapshot (latest or a specific step)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def scratch_output_dir(scratch_config: Path) -> Path:
    cfg = load_yaml(scratch_config)
    return Path(cfg["model"]["output_dir"])


def list_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", maxsplit=1)[1]),
    )


def resolve_snapshot(
    output_dir: Path,
    *,
    step: int | None,
    use_final: bool,
) -> Path:
    if use_final:
        final_dir = output_dir / "final"
        if not final_dir.is_dir():
            raise FileNotFoundError(f"Final snapshot not found: {final_dir}")
        return final_dir

    if step is not None:
        snapshot = output_dir / f"checkpoint-{step}"
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")
        return snapshot

    checkpoints = list_checkpoints(output_dir)
    if checkpoints:
        return checkpoints[-1]

    final_dir = output_dir / "final"
    if final_dir.is_dir():
        return final_dir

    raise FileNotFoundError(
        f"No checkpoints found under {output_dir}. "
        "Expected checkpoint-<step>/ or final/."
    )


def write_eval_config(snapshot: Path, config_path: Path) -> None:
    config = {
        "model": {
            "name": str(snapshot),
            "kind": "scratch_pretrain",
            "trust_remote_code": False,
        },
        "dataset": {
            "name": "google-research-datasets/mbpp",
            "config": "full",
            "split": "train",
            "eval_split": "test",
            "validation_split": 0,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "training": {
            "output_dir": "outputs/eval-snapshot-work",
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_generation(snapshot: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = AutoModelForCausalLM.from_pretrained(snapshot).to(device)
    model.eval()

    prompts = [
        "def reverse_string(s):\n",
        "def is_prime(n):\n",
        "import os\n\ndef list_files(path):\n",
    ]

    print("=" * 72)
    print("Quick generation preview (new tokens only)")
    print("Note: scratch snapshots are not instruction-tuned; weak completion is expected.")
    print("=" * 72)
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
            )
        completion = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        print("-" * 72)
        print("Prompt:")
        print(prompt)
        print("Completion:")
        print(completion if completion.strip() else "(empty / whitespace only)")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a scratch-pretrain snapshot on MBPP (and optional code preview).",
    )
    parser.add_argument(
        "step",
        nargs="?",
        type=int,
        default=None,
        help="Checkpoint step number (e.g. 100000). Omit to use the latest snapshot.",
    )
    parser.add_argument(
        "--scratch-config",
        type=Path,
        default=Path("configs/train_scratch_30m.yaml"),
        help="Scratch training config used to locate outputs/<model>/checkpoint-*",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluate outputs/.../final instead of a checkpoint",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="MBPP examples to evaluate (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: outputs/eval/snapshot_<step>_mbpp.jsonl)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip the quick Python generation preview",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available snapshots and exit",
    )
    args = parser.parse_args()

    output_dir = scratch_output_dir(args.scratch_config)

    if args.list:
        checkpoints = list_checkpoints(output_dir)
        if not checkpoints:
            print(f"No checkpoints in {output_dir}")
        else:
            print(f"Snapshots in {output_dir}:")
            for path in checkpoints:
                print(f"  - {path.name}")
        final_dir = output_dir / "final"
        if final_dir.is_dir():
            print(f"  - final")
        return

    snapshot = resolve_snapshot(output_dir, step=args.step, use_final=args.final)
    step_label = snapshot.name.replace("checkpoint-", "") if snapshot.name.startswith("checkpoint-") else snapshot.name
    eval_config = Path("outputs/eval") / f"snapshot_{step_label}_config.yaml"
    result_path = args.output or Path("outputs/eval") / f"snapshot_{step_label}_mbpp{args.limit}.jsonl"

    print(f"Using snapshot: {snapshot}")
    write_eval_config(snapshot, eval_config)
    print(f"Wrote eval config: {eval_config}")

    if not args.no_generate:
        run_generation(snapshot)

    cmd = [
        sys.executable,
        "-m",
        "eval.run_mbpp",
        "--config",
        str(eval_config),
        "--base-only",
        "--limit",
        str(args.limit),
        "--output",
        str(result_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"MBPP results: {result_path}")


if __name__ == "__main__":
    main()
