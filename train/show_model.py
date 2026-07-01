#!/usr/bin/env python3
"""Print the TinyCodeLLM model specification from a training config."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.model_info import load_config, print_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Show TinyCodeLLM model details")
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
        help="Optional adapter directory to include in the report",
    )
    parser.add_argument(
        "--no-arch",
        action="store_true",
        help="Skip loading Qwen architecture from Hugging Face",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_path = args.adapter
    if adapter_path is None:
        default_adapter = Path(cfg["training"]["output_dir"]) / "adapter"
        if default_adapter.exists():
            adapter_path = default_adapter

    print_model_spec(
        cfg,
        adapter_path,
        mode="info",
        include_architecture=not args.no_arch,
    )


if __name__ == "__main__":
    main()
