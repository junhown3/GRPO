"""Evaluate one or more models on the Countdown dataset and generate reports.

This script can compare zero-shot baselines with GRPO-finetuned checkpoints,
produce a metrics table, plot accuracies, and dump per-example outputs for
manual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from starter import TEMPLATE, _evaluate_equation, _extract_answer, reward_fn


@dataclass
class ModelSpec:
    label: str
    model_path: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0


def parse_model_specs(spec_strs: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw in spec_strs:
        parts = raw.split("::")
        if len(parts) not in (3, 4, 5):
            raise ValueError(
                f"Invalid --model entry '{raw}'. Expected format "
                "'label::model_path::max_tokens[::temperature::top_p]'."
            )
        label, model_path, max_tokens_str, *rest = parts
        try:
            max_tokens = int(max_tokens_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid max_tokens '{max_tokens_str}' for model '{label}'."
            ) from exc
        temperature = float(rest[0]) if rest else 1.0
        top_p = float(rest[1]) if len(rest) > 1 else 1.0
        specs.append(ModelSpec(label=label, model_path=model_path, max_tokens=max_tokens, temperature=temperature, top_p=top_p))
    return specs


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def ensure_tokenizer_ready(tokenizer: AutoTokenizer) -> AutoTokenizer:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_prompts(tokenizer: AutoTokenizer, dataset_split: Sequence[dict], max_tokens: int) -> List[str]:
    prompts: List[str] = []
    for ex in dataset_split:
        prompt_text = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=max_tokens)
        chat_formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(chat_formatted)
    return prompts


def evaluate_model(spec: ModelSpec, dataset_split: Sequence[dict], output_dir: Path, max_model_len: int, gpu_mem_util: float) -> dict:
    print(f"Evaluating {spec.label} (model_path={spec.model_path}, max_tokens={spec.max_tokens})")
    tokenizer = ensure_tokenizer_ready(AutoTokenizer.from_pretrained(spec.model_path))
    prompts = build_prompts(tokenizer, dataset_split, spec.max_tokens)

    sampling_params = SamplingParams(
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        min_tokens=0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    torch.cuda.empty_cache()
    llm = LLM(
        model=spec.model_path,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )
    try:
        responses = llm.generate(prompts, sampling_params, use_tqdm=True)
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass
        torch.cuda.empty_cache()

    records = []
    count_correct = count_partial = count_failed = 0
    for ex, rollout in zip(dataset_split, responses):
        response_text = rollout.outputs[0].text if rollout.outputs else ""
        reward_value = reward_fn(response_text, {"target": ex["target"], "numbers": ex["nums"]})
        equation = _extract_answer(response_text)
        result = _evaluate_equation(equation) if equation is not None else None
        record = {
            "label": spec.label,
            "model_path": spec.model_path,
            "max_tokens": spec.max_tokens,
            "prompt": rollout.prompt,
            "response": response_text,
            "numbers": ex["nums"],
            "target": float(ex["target"]),
            "reward": reward_value,
            "equation": equation,
            "result": result,
        }
        records.append(record)
        if math.isclose(reward_value, 1.0, abs_tol=1e-6):
            count_correct += 1
        elif math.isclose(reward_value, 0.1, abs_tol=1e-6):
            count_partial += 1
        else:
            count_failed += 1

    total = len(records)
    accuracy = (count_correct / total) * 100 if total else 0.0
    mean_reward = sum(r["reward"] for r in records) / total if total else 0.0
    summary = {
        "label": spec.label,
        "model_path": spec.model_path,
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "num_examples": total,
        "count_correct": count_correct,
        "count_partial": count_partial,
        "count_failed": count_failed,
        "accuracy": accuracy,
        "mean_reward": mean_reward,
    }

    slug = slugify(spec.label)
    jsonl_path = output_dir / f"{slug}_responses.jsonl"
    success_path = output_dir / f"{slug}_successes.jsonl"
    failure_path = output_dir / f"{slug}_failures.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f_all, \
            success_path.open("w", encoding="utf-8") as f_success, \
            failure_path.open("w", encoding="utf-8") as f_fail:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            f_all.write(line + "\n")
            if math.isclose(rec["reward"], 1.0, abs_tol=1e-6):
                f_success.write(line + "\n")
            elif math.isclose(rec["reward"], 0.0, abs_tol=1e-6):
                f_fail.write(line + "\n")

    return summary


def load_default_specs(saved_checkpoint_dir: str | None) -> List[ModelSpec]:
    specs = [
        ModelSpec(label="baseline_zero_shot_L256", model_path="Qwen/Qwen3-1.7B", max_tokens=256),
    ]
    if saved_checkpoint_dir and Path(saved_checkpoint_dir).exists():
        specs.append(ModelSpec(label="grpo_finetuned_L256", model_path=saved_checkpoint_dir, max_tokens=256))
    return specs


def save_summary(results: Sequence[dict], output_dir: Path) -> None:
    fieldnames = [
        "label",
        "model_path",
        "max_tokens",
        "temperature",
        "top_p",
        "num_examples",
        "count_correct",
        "count_partial",
        "count_failed",
        "accuracy",
        "mean_reward",
    ]
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(list(results), f_json, indent=2)


def plot_accuracy(results: Sequence[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping accuracy plot.")
        return

    labels = [r["label"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    plt.figure(figsize=(max(6, len(labels) * 1.6), 4))
    bars = plt.bar(labels, accuracies, color="#4C72B0")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Countdown Task Accuracy")
    plt.xticks(rotation=20, ha="right")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plot_path = output_dir / "accuracy_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate models and generate performance reports.")
    parser.add_argument(
        "--model",
        action="append",
        metavar="LABEL::MODEL_PATH::MAX_TOKENS[::TEMPERATURE::TOP_P]",
        help="Specify a model to evaluate. Repeat for multiple models.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./output/hw_a2_solution_1761352930",
        help="Default path to the GRPO checkpoint (used if --model not provided).",
    )
    parser.add_argument(
        "--output-root",
        default="./output/eval_reports",
        help="Directory to store evaluation artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for vLLM generation.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length for vLLM engines.",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.6,
        help="vLLM gpu_memory_utilization value to use for evaluation.",
    )
    args = parser.parse_args(argv)

    if args.model:
        specs = parse_model_specs(args.model)
    else:
        specs = load_default_specs(args.checkpoint_dir)

    if not specs:
        print("No models to evaluate. Provide --model entries or ensure the checkpoint directory exists.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to {output_dir}")

    eval_dataset = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    eval_items = list(eval_dataset)

    vllm_set_random_seed(args.seed)

    summaries = []
    for spec in specs:
        if spec.model_path.startswith("./") or spec.model_path.startswith("../") or spec.model_path.startswith("/"):
            if not Path(spec.model_path).exists():
                print(f"Skipping {spec.label}: model path '{spec.model_path}' not found.")
                continue
        summary = evaluate_model(
            spec=spec,
            dataset_split=eval_items,
            output_dir=output_dir,
            max_model_len=args.max_model_len,
            gpu_mem_util=args.gpu_mem_util,
        )
        summaries.append(summary)

    if not summaries:
        print("No successful evaluations completed.")
        return

    save_summary(summaries, output_dir)
    plot_accuracy(summaries, output_dir)

    print("\nSummary:")
    for row in summaries:
        print(
            f"- {row['label']} | max_tokens={row['max_tokens']} | accuracy={row['accuracy']:.2f}% "
            f"({row['count_correct']}/{row['num_examples']})"
        )
    print(f"\nArtifacts written to: {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
