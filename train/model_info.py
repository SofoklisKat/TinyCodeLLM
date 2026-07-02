"""Describe the TinyCodeLLM model setup from training config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Qwen/Qwen2.5-Coder-0.5B-Instruct (from Hugging Face config.json)
QWEN_05B_ARCH_FALLBACK: dict[str, Any] = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_hf_model_config(model_name: str, trust_remote_code: bool = False) -> dict[str, Any]:
    """Load the base model config from Hugging Face (config.json only, no weights)."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load model architecture. "
            "Install project dependencies from requirements.txt."
        ) from exc

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    config_dict = hf_config.to_dict()
    if "rope_theta" not in config_dict:
        config_dict["rope_theta"] = float(
            getattr(hf_config, "rope_theta", None) or 10_000.0
        )
    return config_dict


def describe_qwen_architecture(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Extract Qwen2 architecture fields for display."""
    hidden_size = int(hf_config["hidden_size"])
    num_heads = int(hf_config["num_attention_heads"])
    head_dim = int(hf_config.get("head_dim") or hidden_size // num_heads)

    return {
        "model_type": hf_config.get("model_type", "qwen2"),
        "architectures": hf_config.get("architectures", ["Qwen2ForCausalLM"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(hf_config["intermediate_size"]),
        "num_hidden_layers": int(hf_config["num_hidden_layers"]),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(hf_config.get("num_key_value_heads", num_heads)),
        "head_dim": head_dim,
        "hidden_act": hf_config.get("hidden_act", "silu"),
        "vocab_size": int(hf_config["vocab_size"]),
        "max_position_embeddings": int(hf_config["max_position_embeddings"]),
        "rms_norm_eps": float(hf_config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(hf_config.get("rope_theta", 1_000_000.0)),
        "tie_word_embeddings": bool(hf_config.get("tie_word_embeddings", True)),
        "attention_dropout": float(hf_config.get("attention_dropout", 0.0)),
        "use_sliding_window": bool(hf_config.get("use_sliding_window", False)),
        "estimated_total_params": estimate_total_params(hf_config),
    }


def estimate_total_params(hf_config: dict[str, Any]) -> int:
    """Rough parameter count from config (no weight download)."""
    hidden = int(hf_config["hidden_size"])
    layers = int(hf_config["num_hidden_layers"])
    intermediate = int(hf_config["intermediate_size"])
    vocab = int(hf_config["vocab_size"])
    heads = int(hf_config["num_attention_heads"])
    kv_heads = int(hf_config.get("num_key_value_heads", heads))
    head_dim = int(hf_config.get("head_dim") or hidden // heads)
    tie = bool(hf_config.get("tie_word_embeddings", True))

    kv_dim = kv_heads * head_dim
    attn = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    per_layer = attn + mlp + norms

    total = layers * per_layer
    if tie:
        total += vocab * hidden
    else:
        total += 2 * vocab * hidden
    return total


def format_param_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B ({count:,})"
    return f"{count / 1_000_000:.2f}M ({count:,})"


def print_qwen_architecture(arch: dict[str, Any]) -> None:
    """Print Qwen2 transformer architecture details."""
    arch_class = ", ".join(arch["architectures"])
    gqa = arch["num_key_value_heads"] < arch["num_attention_heads"]

    print("-" * 72)
    print("Qwen model architecture (from Hugging Face config)")
    print("-" * 72)
    print(f"HF class:          {arch_class}")
    print(f"Model type:        {arch['model_type']}")
    print(f"Estimated params:  {format_param_count(arch['estimated_total_params'])}")
    print(f"Hidden size:       {arch['hidden_size']}")
    print(f"FFN intermediate:  {arch['intermediate_size']}")
    print(f"Transformer layers:{arch['num_hidden_layers']}")
    print(f"Attention heads:   {arch['num_attention_heads']}")
    print(f"KV heads (GQA):    {arch['num_key_value_heads']}")
    print(f"Head dimension:    {arch['head_dim']}")
    print(f"Activation (FFN):  {arch['hidden_act']} (SwiGLU)")
    print(f"Vocabulary size:   {arch['vocab_size']:,}")
    print(f"Max context:       {arch['max_position_embeddings']:,} tokens")
    print(f"Positional enc.:   RoPE (theta={arch['rope_theta']:,.0f})")
    print(f"Normalization:     RMSNorm (eps={arch['rms_norm_eps']})")
    print(f"Tied embeddings:   {arch['tie_word_embeddings']}")
    print(f"Attention dropout: {arch['attention_dropout']}")
    print(f"Sliding window:    {arch['use_sliding_window']}")
    print("Per-layer stack:")
    print("  Input -> RMSNorm")
    print("       -> Multi-Head Self-Attention (Q/K/V/O projections)")
    print("       -> Residual")
    print("       -> RMSNorm")
    print("       -> SwiGLU MLP (gate_proj, up_proj, down_proj)")
    print("       -> Residual")
    if gqa:
        print(
            f"Note: grouped-query attention — {arch['num_attention_heads']} query heads, "
            f"{arch['num_key_value_heads']} key/value heads."
        )
    print("Output head:       lm_head -> vocab logits (tied to token embeddings)")


def describe_model(
    cfg: dict[str, Any],
    adapter_path: Path | None = None,
    *,
    include_architecture: bool = True,
) -> dict[str, Any]:
    """Return a structured description of the model used in this repo."""
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})
    model_kind = model_cfg.get("kind", "qlora")
    is_scratch = model_kind == "scratch_pretrain" or Path(str(model_cfg["name"])).parts[-1].startswith(
        ("checkpoint-", "final")
    ) or any(tag in str(model_cfg["name"]) for tag in ("tinycode-10m", "tinycode-15m", "tinycode-30m"))

    info: dict[str, Any] = {
        "project": "TinyCodeLLM",
        "base_model": model_cfg["name"],
        "architecture": "Decoder-only causal language model (Qwen2-style)",
        "parameter_size": "unknown",
        "task": "Python code completion" if is_scratch else "Python code generation / instruction following",
        "output_dir": train_cfg.get("output_dir", "n/a"),
    }

    if is_scratch:
        info.update(
            {
                "fine_tuning": "Scratch causal LM pretrain on streamed GitHub Python code",
                "base_weights": "All weights trained from random init",
                "trainable_weights": "Full model",
                "inference_note": (
                    "Scratch snapshots complete raw code; they are not instruction-tuned. "
                    "Expect weak MBPP scores until you run QLoRA fine-tuning on MBPP/Alpaca."
                ),
            }
        )
    else:
        lora_cfg = cfg["lora"]
        info.update(
            {
                "parameter_size": "~0.5B",
                "fine_tuning": "PEFT LoRA on top of a frozen 4-bit quantized base model (QLoRA)",
                "base_weights": "Frozen, loaded in 4-bit NF4 via bitsandbytes",
                "trainable_weights": "LoRA adapter matrices only",
                "lora_rank": lora_cfg["r"],
                "lora_alpha": lora_cfg["lora_alpha"],
                "lora_dropout": lora_cfg["lora_dropout"],
                "lora_target_modules": lora_cfg["target_modules"],
                "adapter_path": str(adapter_path) if adapter_path else None,
                "inference_note": (
                    "At inference, load the same base model and apply the saved LoRA adapter. "
                    "Architecture is unchanged; only small adapter weights differ from the base model."
                ),
            }
        )

    if include_architecture:
        try:
            hf_config = load_hf_model_config(
                model_cfg["name"],
                trust_remote_code=model_cfg.get("trust_remote_code", False),
            )
        except Exception:
            hf_config = QWEN_05B_ARCH_FALLBACK
            info["architecture_source"] = "fallback (offline or transformers unavailable)"
        else:
            info["architecture_source"] = "huggingface config.json"
        arch = describe_qwen_architecture(hf_config)
        info["qwen_architecture"] = arch
        info["parameter_size"] = format_param_count(arch["estimated_total_params"])

    return info


def print_model_spec(
    cfg: dict[str, Any],
    adapter_path: Path | None = None,
    *,
    mode: str = "training",
    include_architecture: bool = True,
) -> None:
    """Print a human-readable model specification."""
    info = describe_model(cfg, adapter_path, include_architecture=include_architecture)

    print("=" * 72)
    print("TinyCodeLLM model specification")
    print("=" * 72)
    print(f"Mode:              {mode}")
    print(f"Project:           {info['project']}")
    print(f"Base model:        {info['base_model']}")
    print(f"Architecture:      {info['architecture']}")
    print(f"Parameter size:    {info['parameter_size']}")
    print(f"Task:              {info['task']}")
    print(f"Fine-tuning:       {info['fine_tuning']}")
    print(f"Base weights:      {info['base_weights']}")
    print(f"Trainable weights: {info['trainable_weights']}")
    if "lora_rank" in info:
        print(f"LoRA rank (r):     {info['lora_rank']}")
        print(f"LoRA alpha:        {info['lora_alpha']}")
        print(f"LoRA dropout:      {info['lora_dropout']}")
        print("LoRA target layers:")
        for module in info["lora_target_modules"]:
            print(f"  - {module}")
    if adapter_path is not None:
        print(f"Adapter path:      {adapter_path}")
    print(f"Output dir:        {info['output_dir']}")

    if include_architecture and "qwen_architecture" in info:
        if info.get("architecture_source"):
            print(f"Arch source:       {info['architecture_source']}")
        print_qwen_architecture(info["qwen_architecture"])

    print("-" * 72)
    print(info["inference_note"])
    print("=" * 72)
