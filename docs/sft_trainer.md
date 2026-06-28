# SFTTrainer Guide (Analytical)

This document explains what `SFTTrainer` does in our training pipeline. Each section has four parts:

- **Mechanism** — what actually happens internally
- **Why it matters** — the reason the knob exists
- **Trade-off** — what you gain and lose
- **In our repo** — the concrete value we use and why

It reflects the current code in `train/train_qlora.py` and `configs/train_qlora.yaml`.

## What is SFTTrainer?

**Mechanism.** `SFTTrainer` (from TRL) is a subclass of the Hugging Face `Trainer`. It wraps the full supervised fine-tuning loop: dataset tokenization, batching, the forward/backward passes, optimizer stepping, evaluation, logging, and checkpointing. It is specialized for the common case "train a causal LM to continue text."

**Why it matters.** Writing a correct training loop by hand is error-prone: gradient accumulation, mixed precision, distributed sync, checkpoint resumption, and label shifting all have subtle bugs. `SFTTrainer` gives a battle-tested loop so we only supply data + config.

**Trade-off.** You trade transparency for reliability. The loop is hidden, so when something breaks (e.g. the bf16/fp16 GradScaler crash we hit), you must understand the internals to debug it. It is not magic — it is a well-tested default.

**In our repo.** We use it with `SFTConfig` (training settings) and a dataset whose rows already contain a `text` field. Note: in the current code we apply LoRA *before* constructing the trainer via `get_peft_model(...)`, so we no longer pass `peft_config` to `SFTTrainer`.

```python
from trl import SFTConfig, SFTTrainer

model = get_peft_model(model, peft_config)   # LoRA applied here
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)
trainer.train()
```

---

## End-to-end data flow

Tracing one example all the way through clarifies what each later section controls.

### 1. Raw dataset row

```json
{
  "instruction": "Write a Python function to add two numbers.",
  "input": "",
  "output": "def add(a, b):\n    return a + b"
}
```

### 2. Our formatter builds `text` (`train/dataset.py`)

```text
<|im_start|>user
Write a Python function to add two numbers.
<|im_start|>assistant
def add(a, b):
    return a + b
```

**Analysis.** The model never sees `instruction`/`input`/`output` as separate fields. We collapse them into one flat string using chat markers. The model learns the *statistical pattern* "after `<|im_start|>assistant`, code that solves the request tends to follow." Everything downstream operates on this string.

### 3. Tokenization → IDs

```text
"def add(a, b):" → [707, 1304, 2386, 11, 293, 1648, ...]
```

### 4. Tensor construction

```python
input_ids      = [t0, t1, t2, ..., tN]
labels         = [t0, t1, t2, ..., tN]   # shifted internally by 1
attention_mask = [1,  1,  1,  ..., 1]
```

### 5. Next-token objective

The model predicts token `t_{i+1}` from tokens `t_0..t_i` for every position simultaneously (causal masking makes this one parallel forward pass, not a Python loop).

### 6. Loss

Cross-entropy between predicted distribution and the true next token, averaged over all non-ignored positions.

### 7. Backward → LoRA update

Gradients flow only into trainable LoRA params (everything else is frozen / 4-bit).

---

## 1. Reading text from the dataset

**Mechanism.** `dataset_text_field="text"` tells the trainer which column holds the training string. It maps a tokenization function over the dataset, producing `input_ids`/`attention_mask`.

**Why it matters.** Datasets often have many columns. This is the explicit contract for "this is the thing to train on." If it points at the wrong column, you silently train on garbage.

**Trade-off.** Using a single pre-formatted `text` field is simple and transparent, but it pushes all prompt-template responsibility onto us (`train/dataset.py`). The alternative — letting TRL apply a chat template from structured messages — is more automatic but hides formatting.

**In our repo.** We pre-format in `dataset.py`, so the field is just `text`. This keeps the template visible and version-controlled.

---

## 2. Tokenization

**Mechanism.** `processing_class=tokenizer` supplies the tokenizer. Each `text` string becomes integer IDs plus an attention mask. The tokenizer also defines special tokens (`<|im_start|>`, EOS, PAD).

**Why it matters.** The model only understands token IDs. Tokenization quality directly affects sequence length (cost) and whether structure like newlines/indentation is preserved — critical for code.

**Trade-off.** A code-aware tokenizer (Qwen Coder's) represents code compactly, but it is fixed to the base model. You cannot swap tokenizers without breaking the embeddings.

**In our repo.** We reuse Qwen2.5-Coder's tokenizer. We also set `pad_token = eos_token` when no pad token exists, so batching/padding has a valid pad id.

---

## 3. Truncation (`max_length`)

**Mechanism.** Sequences longer than `max_length` are cut to that many tokens; the remainder is discarded.

**Why it matters.** Attention cost scales roughly with sequence length (and memory with it), so a hard cap bounds per-step VRAM and time. It also guarantees fixed upper bounds for batching.

**Trade-off.** Too small → long examples get their answers chopped off, so the model learns truncated/incomplete code. Too large → wasted memory and slower steps when most samples are short.

**In our repo.** `max_seq_length: 2048`. The Python instruction dataset is mostly short (well under 2048), so truncation rarely fires, but the headroom covers longer multi-step answers. On a 4GB GPU we would drop this to 1024.

---

## 4. Batching (`per_device_train_batch_size`)

**Mechanism.** N samples are stacked into one tensor of shape `[N, seq_len]` and processed in a single forward/backward pass per device.

**Why it matters.** Larger batches use the GPU's parallelism better (higher utilization, fewer Python/kernel-launch overheads) and give a less noisy gradient estimate.

**Trade-off.** Memory grows roughly linearly with batch size. Very large batches can also *over-smooth* gradients and sometimes need a higher learning rate to converge at the same speed.

**In our repo.** We raised this from 2 → **8** after observing only ~2.4GB VRAM used. At batch 8 we use ~4–5GB and ~95–100% GPU utilization, cutting wall-clock time substantially with no quality loss for this model size.

---

## 5. Padding

**Mechanism.** Within a batch, shorter sequences are padded with the pad token up to the longest sequence in that batch. The attention mask marks pad positions as 0 so they are ignored in attention and loss.

**Why it matters.** Tensors must be rectangular. Padding is what makes variable-length text batchable at all.

**Trade-off.** Padding is wasted compute: if one sample is 1900 tokens and the rest are 200, the whole batch runs at ~1900. Dynamic padding (pad to batch max, not global max) reduces this; packing (Section 17) eliminates most of it.

**In our repo.** We rely on standard dynamic padding and keep `packing: false` for simplicity. Because the dataset is fairly uniform in length, padding waste is modest.

---

## 6. The loss (causal LM, and the masking question)

**Mechanism.** Cross-entropy over next-token predictions. By default every token position contributes to the loss, including the user/prompt tokens.

**Why it matters.** This is the actual learning signal. *What* you compute loss over determines *what behavior* you reinforce.

**Trade-off — full-text vs answer-only:**
- **Full-text (current default):** simpler; the model also learns to model prompts. Wastes capacity on reproducing instructions and can encourage prompt echoing. For a tiny 0.5B model, that wasted capacity is non-trivial.
- **Answer-only (prompt tokens masked to `-100`):** focuses all gradient on the assistant response, usually better instruction-following and less echoing. Costs a bit more code to compute the prompt/response boundary.

**In our repo.** We currently train **full-text** (no masking). This is a known, deliberate first-run simplification. Answer-only masking is the highest-value quality improvement we have queued.

---

## 7. PEFT / LoRA integration

**Mechanism.** `get_peft_model` injects small low-rank matrices (A·B) into the targeted linear layers. The original weights are frozen; only A and B are trainable. The forward pass computes `W·x + (alpha/r)·B(A·x)`.

**Why it matters.** It reduces trainable parameters by ~50–100×, which shrinks optimizer memory and checkpoint size and makes training feasible on small GPUs.

**Trade-off.** LoRA has lower capacity than full fine-tuning. For large behavioral changes it can underfit; `r` and `target_modules` trade capacity against memory/speed. Adapters are also tied to the exact base model.

**In our repo.** `r=16`, `alpha=32`, targeting all attention + MLP projections. The run reports **8.8M trainable / 502M total (1.75%)** — exactly the LoRA-only footprint we want. Important detail: after applying LoRA we cast trainable params to **fp32** so the optimizer/grad path is stable on Turing GPUs.

---

## 8. Gradient accumulation

**Mechanism.** Instead of updating weights every micro-batch, gradients are summed across `K` micro-batches and the optimizer steps once. Effective batch = `per_device_batch × K × num_devices`.

**Why it matters.** It decouples the *statistical* batch size (what the optimizer sees) from the *physical* batch size (what fits in VRAM). This lets a small GPU emulate large-batch training.

**Trade-off.** It does not speed things up — `K` micro-batches still cost `K` forward/backward passes. It only saves memory versus a truly large physical batch. More accumulation also means fewer, larger optimizer steps per epoch.

**In our repo.** Now that batch 8 fits physically, we set `gradient_accumulation_steps: 1` (effective batch 8). Earlier, with batch 2, we used accumulation 4 to reach the same effective 8. Same statistics, different memory/speed profile.

---

## 9. Optimizer, learning rate, warmup

**Mechanism.** `paged_adamw_8bit` is AdamW with 8-bit optimizer states and CUDA "paging" to spill state to CPU under pressure. Warmup linearly ramps LR from 0 to target over `warmup_ratio` of total steps, then (by default) decays.

**Why it matters.** AdamW state (two moments per parameter) is a major memory cost; 8-bit + paging slashes it. Warmup prevents early large, destabilizing updates when Adam's variance estimates are still noisy.

**Trade-off.** 8-bit optimizer states introduce tiny quantization noise (negligible in practice). Too-high LR diverges or spikes `grad_norm`; too-low LR wastes time. Warmup too short → early instability; too long → slow start.

**In our repo.** `lr=2e-4` (standard LoRA range), `optim=paged_adamw_8bit`, `warmup_ratio=0.03`. Because only 1.75% of params train, a relatively high LR like 2e-4 is appropriate and stable — consistent with the healthy `grad_norm ≈ 2.4` we observed.

---

## 10. Mixed precision (and why we turned it off)

**Mechanism.** `fp16`/`bf16` run most matmuls in 16-bit. `fp16` additionally needs a **GradScaler** to prevent gradient underflow; `bf16` has wider exponent range and needs no scaler.

**Why it matters.** 16-bit math is faster and uses less memory than fp32.

**Trade-off.** `fp16` GradScaler only supports fp16/fp32 gradients — **not** bf16. `bf16` AMP requires Ampere+ hardware. On Turing (RTX 2080), bf16 kernels for the scaler path are not implemented, which produced our crash: `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`.

**In our repo.** We set **`fp16: false`, `bf16: false`** and train the LoRA params in **fp32**. For a 0.5B model with a 4-bit frozen base, fp32 LoRA is cheap and removes the entire mixed-precision failure class. The startup log prints `Training precision: fp16=False, bf16=False` to confirm.

---

## 11. Gradient checkpointing

**Mechanism.** Normally all layer activations are stored for the backward pass. Checkpointing discards most activations on the forward pass and *recomputes* them during backward.

**Why it matters.** Activation memory often dominates for long sequences/large batches. Checkpointing can cut it dramatically.

**Trade-off.** It adds ~20–30% compute (extra forward recomputation) for large memory savings. Pure win only when you are memory-bound.

**In our repo.** **Disabled** (`gradient_checkpointing: false`). The 0.5B model leaves plenty of VRAM headroom on 8GB, so we prefer speed. (On a 4GB GPU or a 1.5B model we would re-enable it.)

---

## 12. Epochs

**Mechanism.** `num_train_epochs` is how many full passes over the training set. Total optimizer steps ≈ `(samples / effective_batch) × epochs`.

**Why it matters.** Controls total exposure. Too few → underfitting; too many → overfitting/memorization, especially on small datasets.

**Trade-off.** More epochs cost proportional time and raise overfitting risk; the eval-loss curve is the signal for "enough."

**In our repo.** `num_train_epochs: 1` on ~17,100 training samples. One pass is a sensible first run; we decide on more epochs only after reading the eval curve.

---

## 13. Logging

**Mechanism.** Every `logging_steps` the trainer emits scalars: `loss`, `learning_rate`, `grad_norm`, `mean_token_accuracy`, `epoch`, throughput.

**Why it matters.** This is your real-time health monitor. `loss` should trend down; `grad_norm` should stay bounded; `mean_token_accuracy` should rise.

**Trade-off.** Very frequent logging adds minor overhead and noisy lines; too infrequent hides divergence until late.

**In our repo.** `logging_steps: 10`, `report_to: "none"` (console only, no W&B/TensorBoard). Our first lines showed `loss 1.648`, `grad_norm 2.44`, `mean_token_accuracy 0.69` — a healthy start.

---

## 14. Evaluation

**Mechanism.** When `eval_strategy="steps"`, every `eval_steps` the trainer runs the model over the held-out eval set (no grad) and reports eval loss.

**Why it matters.** Train loss alone can drop while the model overfits. Eval loss on unseen data is the honest generalization signal.

**Trade-off.** Evaluation pauses training and costs time proportional to eval-set size and frequency. Too frequent slows the run; too rare risks missing the overfitting point.

**In our repo.** `eval_steps: 200` against the **900-sample** held-out split (5% carved out in `dataset.py`). Watch for eval loss flattening or rising while train loss keeps falling — that's the cue to stop or regularize.

---

## 15. Checkpointing

**Mechanism.** Every `save_steps` the trainer writes a checkpoint (adapter weights, optimizer state, scheduler, RNG). `save_total_limit` prunes old ones.

**Why it matters.** Crash recovery and the ability to pick an earlier, better-generalizing checkpoint after the fact.

**Trade-off.** Checkpoints cost disk and a brief I/O pause. Keeping many uses more disk; keeping too few risks losing the best one.

**In our repo.** `save_steps: 400`, `save_total_limit: 2`, under `outputs/tinycode-qlora/`. At the end we also export the final adapter to `outputs/tinycode-qlora/adapter/`.

---

## 16. Reproducibility

**Mechanism.** `seed` fixes RNG for data shuffling, LoRA init, and dropout.

**Why it matters.** Makes runs comparable — essential when tuning hyperparameters or writing up results for a paper.

**Trade-off.** GPU kernels are not perfectly deterministic by default; identical seeds get you *close*, not bitwise-identical, unless you also force deterministic algorithms (slower).

**In our repo.** `seed: 42`. Good enough to compare configs meaningfully without paying the determinism speed penalty.

---

## 17. Packing

**Mechanism.** With `packing: true`, multiple short examples are concatenated into one `max_length` sequence, nearly eliminating padding.

**Why it matters.** On datasets of many short samples, packing can raise effective throughput significantly (fewer wasted pad tokens).

**Trade-off.** Naive packing lets one example "attend across" into the next unless boundary handling is correct, which can subtly blur examples. It also complicates per-example loss masking.

**In our repo.** **Disabled** for a clean, easy-to-reason-about first run. A candidate optimization once correctness is locked in.

---

## A full optimizer step, concretely (current config)

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
max_seq_length: 2048
fp16: false
bf16: false
```

```text
1. pull 8 samples from the DataLoader
2. tokenize + dynamic-pad to the batch's longest sequence (≤ 2048)
3. forward: 4-bit frozen base + fp32 LoRA, causal mask, full-text labels
4. cross-entropy loss over all non-pad tokens
5. backward: grads only into fp32 LoRA params
6. (accumulation = 1, so no waiting)
7. paged_adamw_8bit updates LoRA weights
8. scheduler advances LR; grads zeroed
9. every 10 steps log; every 200 eval; every 400 checkpoint
```

---

## What SFTTrainer does NOT do (and our current gaps)

1. **Format the data** — we own the chat template in `dataset.py`.
2. **Answer-only masking** — not enabled; we currently train on prompt tokens too. (Top quality TODO.)
3. **Export to GGUF** — a separate step is needed for Ollama/llama.cpp.
4. **Pick the base model** — that's our config (`Qwen2.5-Coder-0.5B-Instruct`).
5. **Guarantee quality** — data quality and objective design dominate; the trainer just executes faithfully.

---

## Config → behavior cheat sheet

| Config field | Controls | Our value | Why |
|---|---|---|---|
| `dataset_text_field` | training column | `text` | pre-formatted in `dataset.py` |
| `max_length` | tokens/sample cap | `2048` | covers long answers, fits 8GB |
| `per_device_train_batch_size` | samples/step | `8` | uses ~half of 8GB, high util |
| `gradient_accumulation_steps` | effective-batch multiplier | `1` | physical batch already large enough |
| `learning_rate` | step size | `2e-4` | standard LoRA LR |
| `num_train_epochs` | dataset passes | `1` | first-run baseline |
| `fp16` / `bf16` | mixed precision | `false`/`false` | avoid Turing GradScaler crash |
| `gradient_checkpointing` | memory vs compute | `false` | plenty of VRAM, prefer speed |
| `eval_steps` | validation cadence | `200` | watch generalization |
| `save_steps` / `save_total_limit` | checkpoint cadence/retention | `400` / `2` | recovery without disk bloat |
| `packing` | concat short samples | `false` | simplicity first |

---

## Summary

`SFTTrainer` turns our formatted coding dataset into LoRA weight updates. The analytical takeaways:

- **Objective design** (full-text vs answer-only) matters more than most knobs for a tiny model.
- **Precision** is hardware-coupled: Turing forces us off bf16, and fp32 LoRA is the safe, cheap choice here.
- **Batch size and checkpointing** are memory/speed dials; with a 0.5B 4-bit base on 8GB we bias toward speed.
- **Eval loss** is the metric that tells the truth about whether training is helping.

Output: a LoRA adapter in `outputs/tinycode-qlora/adapter/` — the trained "skill layer" over the frozen 4-bit base.
