# SFTTrainer Guide

This document explains what `SFTTrainer` does in our training pipeline, with concrete examples from `train/train_qlora.py`.

## What is SFTTrainer?

`SFTTrainer` comes from the **TRL** library (Transformers Reinforcement Learning).

```text
SFT = Supervised Fine-Tuning
```

It fine-tunes a language model on labeled examples:

```text
input text  →  expected continuation
```

In our case:

```text
user request  →  Python code answer
```

We use it together with:

- `SFTConfig` — training settings
- `LoraConfig` — LoRA adapter settings
- a Hugging Face `Dataset` with a `text` column

---

## Where it appears in our code

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(...)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
```

`SFTTrainer` is the object that runs the full training loop.

---

## End-to-end example

### 1. Raw dataset row

From `iamtarun/python_code_instructions_18k_alpaca`:

```json
{
  "instruction": "Write a Python function to add two numbers.",
  "input": "",
  "output": "def add(a, b):\n    return a + b"
}
```

### 2. Our formatter converts it to `text`

From `train/dataset.py`:

```text
<|im_start|>user
Write a Python function to add two numbers.
<|im_start|>assistant
def add(a, b):
    return a + b
```

### 3. SFTTrainer tokenizes it

Tokenizer converts text to token IDs:

```text
[151644, 8948, 198, 2610, 525, ...]
```

### 4. SFTTrainer builds tensors

For one training example:

```python
input_ids = [t1, t2, t3, ..., tN]
labels    = [t1, t2, t3, ..., tN]
attention_mask = [1, 1, 1, ..., 1]
```

### 5. Model predicts next token

For causal language modeling, the model learns:

```text
given tokens [t1, t2, t3] predict t4
given tokens [t1, t2, t3, t4] predict t5
...
```

### 6. Loss is computed

If predicted token != target token, loss increases.

### 7. Backprop updates LoRA weights only

Because we pass `peft_config`, only LoRA adapter weights are updated.

---

## What SFTTrainer handles for us

You do **not** need to manually write:

- tokenization loop
- batching
- padding
- forward pass loop
- backward pass loop
- optimizer step
- checkpoint saving
- evaluation loop

`SFTTrainer` does all of that.

---

## Main functionalities

## 1. Read text from dataset

Config:

```yaml
dataset_text_field: "text"
```

Meaning:

> use the `text` column from the dataset as training input.

Example dataset row:

```python
{
  "text": "<|im_start|>user\nWrite a function...\n<|im_start|>assistant\ndef add(...)"
}
```

Without this field, `SFTTrainer` would not know which column to train on.

---

## 2. Tokenization

`SFTTrainer` uses:

```python
processing_class=tokenizer
```

Example:

```python
text = "<|im_start|>user\nWrite a function\n<|im_start|>assistant\ndef add(a,b): return a+b"
tokens = tokenizer(text)
```

Result:

```python
{
  "input_ids": [151644, 8948, 198, ...],
  "attention_mask": [1, 1, 1, ...]
}
```

This happens automatically for every sample.

---

## 3. Truncate long sequences

Config:

```yaml
max_length: 1024
```

If a sample is longer than 1024 tokens, it is cut.

Example:

```text
very long prompt + very long answer = 1800 tokens
```

After truncation:

```text
first 1024 tokens kept
rest discarded
```

This protects VRAM on small GPUs.

---

## 4. Create batches

Config:

```yaml
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
```

Meaning:

> load 1 sample per GPU step.

Example batch with batch size 1:

```python
input_ids = [[151644, 8948, 198, 2610, ...]]
labels    = [[151644, 8948, 198, 2610, ...]]
```

If batch size were 2, two samples would be padded to the same length and stacked.

---

## 5. Padding unequal sequences

If two samples have different lengths, the shorter one is padded.

Example:

```text
sample A: 120 tokens
sample B: 80 tokens
```

After padding:

```text
sample A: 120 tokens
sample B: 80 tokens + 40 pad tokens
```

Pad tokens are ignored by the attention mask.

---

## 6. Causal language modeling loss

By default, `SFTTrainer` trains the model to predict the next token across the full text.

Example:

```text
<|im_start|>user
Write a function
<|im_start|>assistant
def add(a, b):
    return a + b
```

The model is trained to predict every next token in that sequence.

Important:

- this is the default behavior
- we are **not** yet masking prompt tokens
- so loss is applied to both user and assistant text

If we later want answer-only training, we must add label masking.

---

## 7. PEFT / LoRA integration

We pass:

```python
peft_config=peft_config
```

Example LoRA config:

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

What happens:

```text
base model = frozen
LoRA adapters = trainable
```

So when `trainer.train()` runs:

- forward pass uses base model + LoRA
- backward pass updates only LoRA weights

This is how QLoRA training works in practice.

---

## 8. Gradient accumulation

Config:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

Meaning:

> simulate batch size 8 on a GPU that can only hold 1 sample.

Example:

```text
step 1: process sample 1, accumulate gradients
step 2: process sample 2, accumulate gradients
...
step 8: process sample 8, then optimizer update
```

Effective batch size:

```text
1 x 8 = 8
```

This is very useful on 4GB GPUs.

---

## 9. Optimizer and learning rate

Config:

```yaml
learning_rate: 2.0e-4
optim: paged_adamw_8bit
warmup_ratio: 0.03
```

Meaning:

- optimizer updates LoRA weights after accumulated steps
- learning rate controls step size
- warmup slowly increases LR at the start

Example:

```text
step 1-30: LR ramps up
step 31+: LR stays around 2e-4
```

`paged_adamw_8bit` is chosen because it uses less memory.

---

## 10. Mixed precision training

Config:

```yaml
fp16: true
bf16: false
```

Meaning:

> do most math in 16-bit floats to save memory and speed up training.

On many consumer GPUs:

- `fp16` is used
- on newer GPUs, `bf16` may be better

---

## 11. Gradient checkpointing

Config:

```yaml
gradient_checkpointing: true
```

Meaning:

> trade compute for memory.

Instead of storing all activations, the trainer recomputes some intermediate values during backward pass.

Result:

- lower VRAM use
- slightly slower training

This is important for small GPUs.

---

## 12. Training epochs

Config:

```yaml
num_train_epochs: 1
```

If train set has 2000 samples:

```text
epoch 1 = model sees all 2000 samples once
```

If set to 3:

```text
epoch 1, epoch 2, epoch 3
```

---

## 13. Logging

Config:

```yaml
logging_steps: 10
report_to: "none"
```

Every 10 steps, training prints metrics like:

```text
loss = 1.84
learning_rate = 0.0002
step = 100
```

`report_to: "none"` means no Weights & Biases / TensorBoard logging.

---

## 14. Evaluation

Config:

```yaml
eval_strategy: "steps"
eval_steps: 100
```

If `eval_dataset` exists, every 100 training steps:

```text
run model on validation set
compute eval loss
```

Example:

```text
train loss = 1.42
eval loss  = 1.55
```

Lower eval loss usually means better generalization.

In our pipeline, 5% of data is held out in `train/dataset.py`.

---

## 15. Checkpoint saving

Config:

```yaml
save_steps: 200
save_total_limit: 2
output_dir: outputs/tinycode-qlora
```

Meaning:

- every 200 steps, save checkpoint
- keep only the latest 2 checkpoints

Example files:

```text
outputs/tinycode-qlora/checkpoint-200/
outputs/tinycode-qlora/checkpoint-400/
```

At the end, we also save:

```text
outputs/tinycode-qlora/adapter/
```

---

## 16. Reproducibility

Config:

```yaml
seed: 42
```

This helps make training more repeatable:

- dataset shuffle
- weight initialization randomness
- some sampling behavior

Not perfectly deterministic on GPU, but much more stable.

---

## 17. Packing (disabled in our project)

Config:

```yaml
packing: false
```

If `packing: true`, short examples can be concatenated into one long sequence to reduce padding waste.

Example without packing:

```text
sample A: 200 tokens + padding
sample B: 800 tokens + padding
```

Example with packing:

```text
sample A + sample B combined into one 1000-token sequence
```

We disabled it for simplicity in the first run.

---

## Concrete training step example

Assume:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_length: 1024
```

One optimizer update looks like this:

```text
1. load sample
2. tokenize to <= 1024 tokens
3. forward pass through 4-bit base model + LoRA
4. compute loss
5. backward pass
6. accumulate gradients
7. repeat 8 times
8. optimizer step
9. clear gradients
```

---

## What SFTTrainer does NOT do

Important limitations in our current setup:

### 1. It does not create the dataset format

We must build the `text` column ourselves in `train/dataset.py`.

### 2. It does not automatically do answer-only loss

Right now, prompt tokens also contribute to loss.

### 3. It does not export to GGUF

After training, we still need a separate export step for Ollama.

### 4. It does not choose the base model for us

We choose:

```yaml
model:
  name: Qwen/Qwen2.5-Coder-0.5B-Instruct
```

### 5. It does not guarantee good model quality

Dataset quality matters more than trainer choice.

---

## Minimal mental model

Think of `SFTTrainer` as:

```text
dataset text
  → tokenize
  → batch
  → model forward
  → compute loss
  → backward
  → update LoRA
  → log / eval / save
```

It is a training engine specialized for instruction/text fine-tuning.

---

## Mapping config → behavior

| Config field | What it controls |
|---|---|
| `dataset_text_field` | which dataset column to train on |
| `max_length` | max tokens per sample |
| `per_device_train_batch_size` | samples per step |
| `gradient_accumulation_steps` | effective batch size multiplier |
| `learning_rate` | update step size |
| `num_train_epochs` | how many passes over dataset |
| `fp16` | mixed precision |
| `gradient_checkpointing` | memory saving |
| `eval_strategy` | when to validate |
| `save_steps` | when to checkpoint |
| `packing` | combine short sequences |

---

## Summary

`SFTTrainer` is the component that turns our formatted coding dataset into actual weight updates for the LoRA adapter.

In this repo:

1. `train/dataset.py` prepares `text`
2. `train/train_qlora.py` loads model + LoRA + config
3. `SFTTrainer` runs supervised fine-tuning
4. output is a LoRA adapter in `outputs/tinycode-qlora/adapter/`

That adapter is the trained "skill layer" on top of the frozen base model.
