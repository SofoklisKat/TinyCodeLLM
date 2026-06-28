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

## Default setup

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Method | 4-bit QLoRA |
| Dataset | `iamtarun/python_code_instructions_18k_alpaca` (2k samples for first run) |
| VRAM target | ~4GB (works on GTX 1050 Ti class GPUs) |

## Config

Edit `configs/train_qlora.yaml` to change:

- `model.name` — e.g. `Qwen/Qwen2.5-Coder-1.5B-Instruct` if you have 8GB+ VRAM
- `dataset.max_samples` — set to `18000` for full dataset after smoke test
- `training.num_train_epochs`, `learning_rate`, `max_seq_length`

## GPU notes

Use one GPU via `CUDA_VISIBLE_DEVICES`. For 1.5B models, use at least 8GB VRAM or reduce `max_seq_length` to 512.

## Project layout

```
configs/train_qlora.yaml   # hyperparameters
train/dataset.py           # dataset loading + chat formatting
train/train_qlora.py       # QLoRA training entrypoint
scripts/run_train.sh       # training launcher (no env setup)
outputs/                   # checkpoints (gitignored)
```
