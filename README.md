# TinyCodeLLM

Train a small code LLM that runs efficiently on consumer GPUs (4–8GB VRAM) for coding suggestions.

## Benchmark results

Public scores from the literature (`docs/papers/`). Use these as baselines for TinyCodeLLM.

### Our base model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

Source: [Qwen2.5-Coder Technical Report](docs/papers/qwen2_5_coder_technical_report.pdf) (EvalPlus, instruct models).

| Model | HE | HE+ | MBPP | MBPP+ | BigCodeBench Full | LiveCodeBench |
|-------|---:|----:|-----:|------:|------------------:|--------------:|
| **Qwen2.5-Coder-0.5B-Instruct** (base) | 61.6 | 57.3 | 52.4 | 43.7 | 11.1 | 2.0 |
| Qwen2.5-Coder-1.5B-Instruct | 70.7 | 66.5 | 69.2 | 59.4 | 32.5 | 6.1 |
| Qwen2.5-Coder-3B-Instruct | 84.1 | 80.5 | 73.6 | 62.4 | 35.8 | 10.8 |
| Qwen2.5-Coder-7B-Instruct | 88.4 | 84.1 | 83.5 | 71.7 | 41.0 | 18.2 |
| **TinyCodeLLM** (ours, TBD) | — | — | — | — | — | — |

**HE** = HumanEval pass@1, **HE+** = HumanEval+ pass@1, **MBPP+** = MBPP+ pass@1.

### Local MBPP pass@1

These scores use our local evaluator (`eval/run_mbpp.py`) on the MBPP `test` split with greedy decoding (`temperature=0`). They are directly comparable to each other, but not necessarily identical to the Qwen paper's official evaluation setup.

| Model | Adapter | Passed | Examples | Local MBPP pass@1 |
|-------|---------|-------:|---------:|------------------:|
| Qwen2.5-Coder-0.5B-Instruct | none | 124 | 500 | 0.2480 |
| TinyCodeLLM MBPP QLoRA | `outputs/tinycode-qlora-mbpp/adapter/` | 185 | 500 | 0.3700 |

Local improvement over the base model: **+12.2 absolute points** (`0.3700 - 0.2480`), or about **+49% relative** (`0.3700 / 0.2480 - 1`).

### Our training runs

These are trainer metrics from our local QLoRA runs. They are useful for tracking convergence, but they are **not** pass@1 benchmark scores. Pass@1 requires executing generated code against benchmark tests.

| Run | Dataset | Epochs | Train loss | Eval loss | Eval token accuracy | Runtime | Adapter |
|-----|---------|-------:|-----------:|----------:|--------------------:|--------:|---------|
| MBPP QLoRA | `google-research-datasets/mbpp` (`train` → `test`) | 3 | 0.8173 | 0.9359 | 0.7768 | 207.5s | `outputs/tinycode-qlora-mbpp/adapter/` |

MBPP run details:

| Metric | Value |
|--------|------:|
| Eval runtime | 14.43s |
| Eval samples/sec | 34.65 |
| Eval steps/sec | 8.663 |
| Eval entropy | 0.6815 |
| Eval tokens | 89,670 |
| Train samples/sec | 5.408 |
| Train steps/sec | 0.68 |

### Same-size base (pre-instruct) model

Source: [Qwen2.5-Coder Technical Report](docs/papers/qwen2_5_coder_technical_report.pdf) (base models).

| Model | HE | HE+ | MBPP | MBPP+ |
|-------|---:|----:|-----:|------:|
| Qwen2.5-Coder-0.5B (base) | 28.0 | 23.8 | 52.9 | 47.1 |

### MBPP dataset (training benchmark)

Source: [Program Synthesis with Large Language Models](docs/papers/mbpp_program_synthesis.pdf).

| Split | Problems | Use |
|-------|----------|-----|
| Train | ~374 | fine-tuning (`configs/train_qlora_mbpp.yaml`) |
| Test | ~500 | public benchmark evaluation |
| Total | 974 | Mostly Basic Python Problems |

Do **not** train on the MBPP test split if you want a valid benchmark score.

### Publishability targets (0.5B)

Reasonable first research goal:

| Benchmark | Base (0.5B-Instruct) | Target for TinyCodeLLM |
|-----------|---------------------:|-----------------------:|
| HumanEval+ | 57.3 | **60+** |
| MBPP+ | 43.7 | **48+** |
| MBPP | 52.4 | **58+** |

Stronger claim: beat base by **≥3 points** on at least two benchmarks while staying under **8GB VRAM**.

### Reference papers

Local copies in `docs/papers/`:

| Topic | Paper |
|-------|-------|
| Base model scores | `qwen2_5_coder_technical_report.pdf` |
| HumanEval benchmark | `humaneval_codex.pdf` |
| MBPP benchmark | `mbpp_program_synthesis.pdf` |
| Stricter tests | `evalplus.pdf` |
| Complex code tasks | `bigcodebench.pdf` |
| Newer OOD tasks | `livecodebench.pdf` |
| Small model study | `small_language_models_code_generation_empirical_study.pdf` |

Full list: `docs/papers/README.md`

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
outputs/tinycode-qlora-mbpp/adapter/   # MBPP training
outputs/tinycode-qlora/adapter/        # Alpaca training
```

## Test adapter

Run default coding prompts against the trained adapter:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 -m train.test_adapter --config configs/train_qlora_mbpp.yaml
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

## Evaluate MBPP pass@1

Run a quick smoke test on 10 MBPP test examples:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 -m eval.run_mbpp \
  --config configs/train_qlora_mbpp.yaml \
  --limit 10 \
  --output outputs/eval/mbpp_tinycode_smoke.jsonl
```

Run full MBPP test evaluation:

```bash
python3 -m eval.run_mbpp \
  --config configs/train_qlora_mbpp.yaml \
  --output outputs/eval/mbpp_tinycode.jsonl
```

Compare against the base model without the LoRA adapter:

```bash
python3 -m eval.run_mbpp \
  --config configs/train_qlora_mbpp.yaml \
  --base-only \
  --output outputs/eval/mbpp_base.jsonl
```

Or use the helper:

```bash
bash scripts/run_mbpp_eval.sh --config configs/train_qlora_mbpp.yaml --limit 10
```

The evaluator reports:

- number of examples
- number passed
- `pass@1`
- JSONL records with generated code and errors

Evaluation executes model-generated code with a timeout. Run it only in an environment where executing benchmark code is acceptable.

## Default setup

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Method | 4-bit QLoRA |
| Dataset | `google-research-datasets/mbpp` (train split, eval on test) |
| VRAM target | 8GB (RTX 2080 class GPUs) |

## Config

Edit `configs/train_qlora_mbpp.yaml` or `configs/train_qlora.yaml` to change:

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
scripts/run_mbpp_eval.sh   # MBPP pass@1 evaluator launcher
outputs/                   # checkpoints (gitignored)
```
