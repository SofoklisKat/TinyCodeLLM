# PEFT, LoRA, and QLoRA

This project trains a small code model on limited GPU memory. The main idea is to avoid full fine-tuning and update only a small number of extra parameters.

## Full Fine-Tuning

In full fine-tuning, every weight in the base model is updated.

For example, if the model has 500 million parameters, training updates all 500 million parameters. This gives maximum flexibility, but it is expensive:

- high VRAM use
- slow training
- large checkpoints
- higher risk of overfitting on small datasets

For consumer GPUs, full fine-tuning is usually not the best first choice.

## PEFT

PEFT means **Parameter-Efficient Fine-Tuning**.

Instead of updating the whole model, PEFT updates a small set of trainable parameters while keeping the original model mostly frozen.

Practical result:

- the base model stays unchanged
- training uses much less GPU memory
- the output is a small adapter, not a full model copy
- the adapter can be merged into the base model later

PEFT is the general category. LoRA is one PEFT method.

## LoRA

LoRA means **Low-Rank Adaptation**.

LoRA freezes the original model weights and adds small trainable matrices inside selected layers, usually attention and feed-forward layers.

Instead of changing a large weight matrix directly, LoRA learns a small update:

```text
original weight + small learned LoRA update
```

The important LoRA settings are:

| Setting | Meaning |
|---|---|
| `r` | LoRA rank. Higher means more capacity, more memory. |
| `lora_alpha` | Scaling factor for the LoRA update. |
| `lora_dropout` | Dropout applied during LoRA training. |
| `target_modules` | Which model layers receive LoRA adapters. |

In this repo, we target Qwen-style projection layers:

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

This trains adapters in both attention and MLP blocks.

## QLoRA

QLoRA means **Quantized LoRA**.

It combines:

1. a frozen base model loaded in 4-bit precision
2. LoRA adapters trained on top of it

The base model is quantized to save memory, while the LoRA adapter remains trainable.

This is why QLoRA is useful for this project:

- a 0.5B or 1.5B model can fit on a small GPU
- training is much cheaper than full fine-tuning
- checkpoints are small
- quality is usually good enough for instruction tuning

## How Training Works Here

The training flow is:

```text
load base model in 4-bit
freeze base model
add LoRA adapters
train adapters on code dataset
save adapter weights
```

The base model is:

```yaml
model:
  name: Qwen/Qwen2.5-Coder-0.5B-Instruct
```

The output is not a full model. It is a LoRA adapter:

```text
outputs/tinycode-qlora/adapter/
```

Later, this adapter can be:

- loaded together with the base model
- merged into the base model
- exported to formats like GGUF for Ollama or llama.cpp

## Practical Difference

| Method | Updates | Memory Use | Output Size | Best For |
|---|---:|---:|---:|---|
| Full fine-tuning | All weights | Highest | Full model | Large compute, maximum control |
| LoRA | Adapter weights | Medium | Small adapter | Efficient fine-tuning |
| QLoRA | Adapter weights, base in 4-bit | Lowest | Small adapter | Consumer GPUs |

## Why We Use QLoRA

For TinyCodeLLM, QLoRA is the practical default because the project targets small GPUs.

The goal is not to train a model from scratch. The goal is to take an existing small code model and teach it better behavior for our target use case.

For the first run, that means:

- use a small pretrained code model
- train with QLoRA
- save a LoRA adapter
- measure whether the model improves

## Important Limitations

QLoRA does not magically create a strong model from bad data. The dataset matters more than the fine-tuning method.

Also, LoRA adapters are tied to the base model. An adapter trained on `Qwen/Qwen2.5-Coder-0.5B-Instruct` should be used with that same base model.

If we later switch to a 1.5B model, we need to train a new adapter.

## Rule of Thumb

Use:

- **LoRA** when the base model can fit comfortably in memory
- **QLoRA** when memory is tight
- **full fine-tuning** only when there is enough compute and a strong reason

For this repo, start with **QLoRA**.
