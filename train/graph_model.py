#!/usr/bin/env python3
"""Print or save a visual architecture graph for TinyCode models."""

from __future__ import annotations

import argparse
from pathlib import Path

from train.model import count_parameters, tinycode_30m, tinycode_30m_config


def mermaid_tinycode_30m() -> str:
    cfg = tinycode_30m_config()
    params = count_parameters(tinycode_30m())
    return f"""```mermaid
flowchart TB
    subgraph Input
        IDS["input_ids<br/>batch × seq_len"]
    end

    IDS --> EMB["Embedding<br/>{cfg.vocab_size:,} × {cfg.hidden_size}"]
    EMB --> LAYERS

    subgraph LAYERS["{cfg.num_hidden_layers}× Decoder Layer"]
        direction TB
        LN1["RMSNorm"]
        ATTN["Grouped-Query Attention<br/>{cfg.num_attention_heads} Q heads / {cfg.num_key_value_heads} KV heads<br/>RoPE, head_dim={cfg.head_dim}"]
        QKV["q_proj · k_proj · v_proj · o_proj"]
        RES1["+ residual"]
        LN2["RMSNorm"]
        MLP["SwiGLU MLP<br/>gate_proj · up_proj · down_proj<br/>{cfg.hidden_size} → {cfg.intermediate_size} → {cfg.hidden_size}"]
        RES2["+ residual"]
        LN1 --> ATTN --> QKV --> RES1 --> LN2 --> MLP --> RES2
    end

    LAYERS --> NORM["RMSNorm"]
    NORM --> HEAD["lm_head<br/>{cfg.hidden_size} → {cfg.vocab_size:,}"]
    HEAD --> LOGITS["logits<br/>next-token prediction"]
    EMB -. tied weights .-> HEAD

    subgraph Stats["Model stats"]
        P["~{params / 1_000_000:.1f}M parameters"]
    end
```"""


def mermaid_qwen_05b() -> str:
    return """```mermaid
flowchart TB
    subgraph Qwen05B["Qwen2.5-Coder-0.5B-Instruct (HF base)"]
        E1["Embedding 151,936 × 896"]
        L1["24× Decoder Layer"]
        N1["RMSNorm"]
        H1["lm_head → vocab"]
        E1 --> L1 --> N1 --> H1
    end

    subgraph Tiny30M["TinyCode 30M (scratch)"]
        E2["Embedding 50,257 × 320"]
        L2["10× Decoder Layer"]
        N2["RMSNorm"]
        H2["lm_head → vocab"]
        E2 --> L2 --> N2 --> H2
    end

    subgraph QLoRA["QLoRA fine-tune"]
        ADAPTER["LoRA adapters on<br/>q/k/v/o + MLP projections"]
    end

    Qwen05B --> ADAPTER
    Tiny30M --> ADAPTER
    ADAPTER --> OUT["Code generation"]
```"""


def mermaid_one_decoder_layer() -> str:
    cfg = tinycode_30m_config()
    return f"""```mermaid
flowchart TB
    X["hidden states<br/>{cfg.hidden_size}-dim"] --> LN1["input_layernorm"]
    LN1 --> ATTN["self_attn"]
    ATTN --> ADD1["+"]
    X --> ADD1
    ADD1 --> LN2["post_attention_layernorm"]
    LN2 --> MLP["mlp / SwiGLU"]
    MLP --> ADD2["+"]
    ADD1 --> ADD2
    ADD2 --> Y["output<br/>{cfg.hidden_size}-dim"]

    subgraph ATTN_DETAIL["self_attn internals"]
        Q["q_proj"]
        K["k_proj"]
        V["v_proj"]
        O["o_proj"]
        R["RoPE"]
        Q --> R
        K --> R
    end
```"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Show TinyCode model architecture graphs")
    parser.add_argument(
        "--which",
        choices=["30m", "compare", "layer"],
        default="30m",
        help="Which diagram to print",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save mermaid markdown to this file",
    )
    args = parser.parse_args()

    builders = {
        "30m": mermaid_tinycode_30m,
        "compare": mermaid_qwen_05b,
        "layer": mermaid_one_decoder_layer,
    }
    diagram = builders[args.which]()

    if args.save:
        body = diagram.strip()
        if body.startswith("```mermaid"):
            body = body[len("```mermaid") :].strip()
        if body.endswith("```"):
            body = body[:-3].strip()
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(body + "\n", encoding="utf-8")
        print(f"Saved mermaid source to {args.save}")
        print("Open https://mermaid.live and paste the file contents to view as a graph.")

    print(diagram)


if __name__ == "__main__":
    main()
