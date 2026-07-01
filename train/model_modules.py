"""Build and print the PyTorch nn.Module tree for TinyCodeLLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from train.model_info import load_config


def build_base_model(model_name: str, trust_remote_code: bool = False) -> nn.Module:
    """Instantiate the Hugging Face model class from config only (no weight download)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return AutoModelForCausalLM.from_config(config)


def build_lora_model(base_model: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    """Wrap the base model with the same LoRA config used during QLoRA training."""
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, peft_config)


def load_adapter_model(
    base_model: nn.Module,
    adapter_path: Path,
) -> nn.Module:
    """Attach a saved LoRA adapter directory to the base model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, str(adapter_path))


def print_nn_module_tree(
    model: nn.Module,
    *,
    expand_layer: int | None = None,
) -> None:
    """Print the standard PyTorch nn.Module representation."""
    print("=" * 72)
    print("PyTorch nn.Module tree (standard print(model) format)")
    print("=" * 72)
    print(model)
    print("=" * 72)

    if expand_layer is not None:
        layers = _decoder_layers(model)
        if layers is None:
            print(f"Could not find decoder layers to expand layer {expand_layer}.")
            return
        if expand_layer < 0 or expand_layer >= len(layers):
            raise IndexError(
                f"Layer index {expand_layer} out of range for {len(layers)} decoder layers."
            )
        print(f"Expanded decoder layer {expand_layer}:")
        print("-" * 72)
        print(layers[expand_layer])
        print("=" * 72)


def _decoder_layers(model: nn.Module) -> nn.ModuleList | None:
    """Locate the transformer decoder ModuleList across base and PEFT wrappers."""
    candidates = [model]
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    if hasattr(model, "model"):
        candidates.append(model.model)

    for candidate in candidates:
        inner = getattr(candidate, "model", candidate)
        layers = getattr(inner, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    return None


def print_model_modules(
    cfg: dict[str, Any],
    *,
    base_only: bool = False,
    adapter_path: Path | None = None,
    expand_layer: int | None = None,
) -> nn.Module:
    """Build and print the model module tree described by a training config."""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    model_name = model_cfg["name"]
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    print(f"Building model from config: {model_name}")
    print("Note: structure only — weights are not downloaded (from_config).")
    print()

    model = build_base_model(model_name, trust_remote_code=trust_remote_code)

    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        print(f"Attaching saved adapter: {adapter_path}")
        model = load_adapter_model(model, adapter_path)
    elif not base_only:
        print("Wrapping with LoRA modules from training config (get_peft_model).")
        model = build_lora_model(model, lora_cfg)

    print_nn_module_tree(model, expand_layer=expand_layer)
    return model
