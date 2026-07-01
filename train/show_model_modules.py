#!/usr/bin/env python3
"""Print the PyTorch nn.Module / transformer layer tree for TinyCodeLLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.model_info import load_config
from train.model_modules import print_model_modules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Qwen2 transformer layers as a PyTorch nn.Module tree",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_qlora_mbpp.yaml"),
        help="Training config YAML",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional saved adapter directory (PeftModel.from_pretrained)",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Ignore saved adapter; build LoRA modules from YAML config instead",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Show base Qwen2ForCausalLM only, without LoRA modules",
    )
    parser.add_argument(
        "--expand-layer",
        type=int,
        default=None,
        metavar="N",
        help="Also print one decoder layer in full detail (0-based index)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_path = args.adapter
    if not args.no_adapter and adapter_path is None and not args.base_only:
        default_adapter = Path(cfg["training"]["output_dir"]) / "adapter"
        if default_adapter.exists():
            adapter_path = default_adapter

    print_model_modules(
        cfg,
        base_only=args.base_only,
        adapter_path=adapter_path,
        expand_layer=args.expand_layer,
    )


if __name__ == "__main__":
    main()
