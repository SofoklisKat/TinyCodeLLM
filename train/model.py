"""Reference Qwen2-style decoder-only causal LM for TinyCodeLLM.

This mirrors the architecture of Qwen/Qwen2.5-Coder-0.5B-Instruct using plain
PyTorch nn.Module classes. Training in this repo still uses the Hugging Face
checkpoint; this file is for learning, inspection, and shape checks.

Run:
    python -m train.model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen2Config:
    """Defaults match Qwen/Qwen2.5-Coder-0.5B-Instruct config.json."""

    vocab_size: int = 151_936
    hidden_size: int = 896
    intermediate_size: int = 4_864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 32_768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0


def config_value(config: object, key: str, default: float | int | bool | None = None):
    """Read a config field from a dataclass or Hugging Face PretrainedConfig."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def patch_qwen2_config(config: object, default_rope_theta: float = 10_000.0) -> object:
    """Ensure HF/local Qwen2 configs expose rope_theta for older checkpoints."""
    rope_theta = config_value(config, "rope_theta")
    if rope_theta is None:
        rope_params = config_value(config, "rope_parameters")
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")
    if rope_theta is None:
        rope_theta = default_rope_theta
        if isinstance(config, dict):
            config["rope_theta"] = rope_theta
        else:
            config.rope_theta = rope_theta
    return config


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        config = patch_qwen2_config(config)
        head_dim = int(config_value(config, "head_dim") or config.hidden_size // config.num_attention_heads)
        rope_theta = float(config_value(config, "rope_theta", 10_000.0))
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        max_position_embeddings = int(config_value(config, "max_position_embeddings", 2048))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen2MLP(nn.Module):
    """SwiGLU feed-forward block (gate_proj, up_proj, down_proj)."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        cos, sin = self.rotary_emb(seq_len, hidden_states.device)
        causal_mask = build_causal_mask(batch_size, seq_len, hidden_states.device, hidden_states.dtype)
        if attention_mask is not None:
            causal_mask = causal_mask + build_padding_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, causal_mask)
        return self.norm(hidden_states)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Tiny greedy decoder for smoke tests (not for production eval)."""
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated


def build_causal_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len).to(dtype)


def build_padding_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # attention_mask: (batch, seq) with 1 for real tokens, 0 for padding
    mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return mask


def qwen2_coder_05b(**overrides: int | float | bool) -> Qwen2ForCausalLM:
    """Build a Qwen2.5-Coder-0.5B-shaped model with random weights."""
    config = Qwen2Config(**overrides)
    return Qwen2ForCausalLM(config)


def tinycode_30m_config(**overrides: int | float | bool) -> Qwen2Config:
    """~30M-parameter Qwen2-style decoder tuned for 4-5GB overnight pretraining."""
    defaults: dict[str, int | float | bool] = {
        "vocab_size": 50_257,  # GPT-2 tokenizer size
        "hidden_size": 320,
        "intermediate_size": 1_280,
        "num_hidden_layers": 10,
        "num_attention_heads": 10,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 2_048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
    }
    defaults.update(overrides)
    return Qwen2Config(**defaults)  # type: ignore[arg-type]


def tinycode_30m(**overrides: int | float | bool) -> Qwen2ForCausalLM:
    """Build the ~30M TinyCode decoder with random weights."""
    return Qwen2ForCausalLM(tinycode_30m_config(**overrides))


def export_to_huggingface(model: Qwen2ForCausalLM, output_dir: str | Path) -> None:
    """Save weights in Hugging Face Qwen2 format for later QLoRA fine-tuning."""
    from transformers import Qwen2Config as HFQwen2Config
    from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = model.config
    hf_config = HFQwen2Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        attention_dropout=cfg.attention_dropout,
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
    )
    hf_model = HFQwen2ForCausalLM(hf_config)
    hf_model.load_state_dict(model.state_dict(), strict=True)
    patch_qwen2_config(hf_model.config, default_rope_theta=float(cfg.rope_theta))
    hf_model.save_pretrained(output_path)
    print(f"Saved Hugging Face checkpoint to {output_path}")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    model = tinycode_30m()
    print("TinyCode 30M decoder")
    print(model)
    print()
    print(f"Total parameters:     {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print(f"Input shape:  {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
