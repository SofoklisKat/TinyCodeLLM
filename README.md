# TinyCodeLLM

Train a small code LLM that runs efficiently on consumer GPUs (4–8GB VRAM) for coding suggestions.

## Quick start (training)

Install dependencies in your own Python environment (outside this repo):

```bash
pip install -U pip
pip install -r requirements.txt
```

Run QLoRA fine-tuning from the repo root:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m train.train_qlora --config configs/train_qlora_mbpp.yaml
```

MBPP training uses the official **train** split and evaluates on the **test** split (no random holdout).

Or use the previous Alpaca dataset:

```bash
python3 -m train.train_qlora --config configs/train_qlora.yaml
```

Or:

```bash
bash scripts/run_train.sh
```

Output LoRA adapter:

```
outputs/tinycode-qlora/adapter/
```

## Test adapter

Run default coding prompts against the trained adapter:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 -m train.test_adapter --config configs/train_qlora.yaml
```

Or:

```bash
bash scripts/test_adapter.sh
```

Single custom prompt:

```bash
python3 -m train.test_adapter --prompt "Write a Python function to sort a list of integers."
```

Multiple prompts on the command line:

```bash
python3 -m train.test_adapter \
  --prompt "Write a Python function to reverse a string." \
  --prompt "Write a Python function to check if a number is prime."
```

Many prompts from a file (`prompts/test_prompts.txt`):

```text
Write a Python function to reverse a string.
---
Write a Python function to check if a number is prime.
---
Fix this Python function:

def average(nums):
    return sum(nums) / len(nums)
```

Run:

```bash
python3 -m train.test_adapter --prompts-file prompts/test_prompts.txt
```

Requirements:
- trained adapter at `outputs/tinycode-qlora/adapter/`
- same Python environment with `requirements.txt` installed
- GPU optional but recommended (`CUDA_VISIBLE_DEVICES=0`)

## Default setup

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Method | 4-bit QLoRA |
| Dataset | `google-research-datasets/mbpp` (train split, eval on test) |
| VRAM target | 8GB (RTX 2080 class GPUs) |

## Config

Edit `configs/train_qlora.yaml` to change:

- `model.name` — e.g. `Qwen/Qwen2.5-Coder-1.5B-Instruct` if you have 8GB+ VRAM
- `dataset.max_samples` — set to `500`–`2000` for a quick smoke test
- `training.num_train_epochs`, `learning_rate`, `max_seq_length`

## GPU notes

Default config targets **8GB VRAM** (`batch_size=8`, `max_seq_length=2048`).

Use one GPU via `CUDA_VISIBLE_DEVICES`. On 4GB GPUs, reduce `max_seq_length` to `1024`, set `per_device_train_batch_size: 1`, and enable `gradient_checkpointing: true`.

## Project layout

```
configs/train_qlora_mbpp.yaml  # MBPP train/test (recommended)
configs/train_qlora.yaml       # Alpaca instruction dataset
train/dataset.py           # dataset loading + chat formatting
train/train_qlora.py       # QLoRA training entrypoint
train/test_adapter.py      # load adapter and run test prompts
scripts/run_train.sh       # training launcher (no env setup)
scripts/test_adapter.sh    # adapter test launcher
outputs/                   # checkpoints (gitignored)
```
