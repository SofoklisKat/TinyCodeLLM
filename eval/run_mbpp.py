#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on MBPP pass@1."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    passed: bool
    error: str | None = None


@dataclass
class EvalRecord:
    task_id: int
    prompt: str
    generated: str
    code: str
    passed: bool
    error: str | None


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f)


def build_prompt(problem_prompt: str) -> str:
    return f"<|im_start|>user\n{problem_prompt.strip()}\n<|im_start|>assistant\n"


def extract_code(generated_text: str) -> str:
    """Extract Python code from a model response."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", generated_text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()

    text = generated_text.strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, maxsplit=1)[0].strip()
    return text


def _execute_tests(
    candidate_code: str,
    test_imports: list[str],
    test_list: list[str],
    queue: mp.Queue,
) -> None:
    namespace: dict[str, Any] = {}
    try:
        for import_stmt in test_imports:
            if import_stmt.strip():
                exec(import_stmt, namespace)
        exec(candidate_code, namespace)
        for test in test_list:
            exec(test, namespace)
        queue.put(TestResult(passed=True))
    except BaseException:
        queue.put(TestResult(passed=False, error=traceback.format_exc(limit=2)))


def run_candidate_tests(
    candidate_code: str,
    test_imports: list[str],
    test_list: list[str],
    timeout_seconds: int,
) -> TestResult:
    """Run MBPP assertions in a child process with a timeout."""
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=_execute_tests,
        args=(candidate_code, test_imports, test_list, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return TestResult(passed=False, error=f"Timeout after {timeout_seconds}s")

    if queue.empty():
        return TestResult(passed=False, error="Test process exited without result")
    return queue.get()


def load_model(
    base_model: str,
    adapter_path: Path | None,
    trust_remote_code: bool,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if adapter_path and (adapter_path / "tokenizer_config.json").exists() else base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    import torch

    inputs = tokenizer(build_prompt(prompt), return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return generated, extract_code(generated)


def evaluate_mbpp(
    model,
    tokenizer,
    split: str,
    dataset_config: str,
    limit: int | None,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> tuple[list[EvalRecord], float]:
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", dataset_config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records: list[EvalRecord] = []
    for index, example in enumerate(dataset, start=1):
        generated, code = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=example["text"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = run_candidate_tests(
            candidate_code=code,
            test_imports=[example.get("test_setup_code", "")],
            test_list=example["test_list"],
            timeout_seconds=timeout_seconds,
        )
        record = EvalRecord(
            task_id=int(example["task_id"]),
            prompt=example["text"],
            generated=generated,
            code=code,
            passed=result.passed,
            error=result.error,
        )
        records.append(record)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{index}/{len(dataset)}] task_id={record.task_id} {status}")

    pass_at_1 = sum(record.passed for record in records) / len(records) if records else 0.0
    return records, pass_at_1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MBPP pass@1 evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/train_qlora_mbpp.yaml"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--base-only", action="store_true", help="Evaluate the base model without LoRA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-config", default="full")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for a quick smoke test")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/mbpp_results.jsonl"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    adapter_path = None
    if not args.base_only:
        adapter_path = args.adapter or (Path(cfg["training"]["output_dir"]) / "adapter")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"Base model: {model_cfg['name']}")
    print(f"Adapter: {adapter_path if adapter_path else 'none (base-only)'}")
    print(f"Dataset: google-research-datasets/mbpp ({args.dataset_config}, split={args.split})")

    model, tokenizer = load_model(
        base_model=model_cfg["name"],
        adapter_path=adapter_path,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    records, pass_at_1 = evaluate_mbpp(
        model=model,
        tokenizer=tokenizer,
        split=args.split,
        dataset_config=args.dataset_config,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")

    print("\nMBPP evaluation complete")
    print(f"Examples: {len(records)}")
    print(f"Passed: {sum(record.passed for record in records)}")
    print(f"pass@1: {pass_at_1:.4f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
