# In this assignment, you will implement the core components of the
# Group-Reward Policy Optimization (GRPO) algorithm. You'll be working with a
# math-solving task called "Countdown," where the model has to generate an
# equation to reach a target number using a given set of numbers.
#
# You will need to implement six key functions:
# 1. `_extract_answer`: To parse the model's response.
# 2. `_validate_numbers`: To ensure the generated equation uses the correct numbers.
# 3. `_evaluate_equation`: To safely calculate the result of the generated equation.
# 4. `reward_fn`: To score the model's generated equations using the helpers.
# 5. `compute_group_normalized_advantages`: To calculate advantages.
# 6. `compute_loss`: To compute per-token loss.
# 7. `masked_mean`: To compute the mean of masked tensor, used in GRPO
# 8. `masked_mean_drgrpo`: To compute the mean of masked tensor, used in DR-GRPO
# ==============================================================================

import os
import datetime
import random
from typing import Callable, Dict, List, Tuple, Any
import logging
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import re
import ast
from dotenv import load_dotenv
load_dotenv()
load_dotenv()

logging.getLogger("vllm.engine.scheduler").setLevel(logging.ERROR)
os.environ["VLLM_USE_V1"] = "1"

from collections import Counter

from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -------------------------
# Prompting helpers (Countdown-style)
# -------------------------
TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""


# vLLM utilities
def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int | None = None,
    max_model_len: int = 2048,
) -> LLM:
    vllm_set_random_seed(seed)
    # Use only public, stable args; avoid internal patches.
    return LLM(
        model=model_id,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    seed: int,
    gpu_memory_utilization: float = 0.4,
    max_num_seqs: int | None = None,
    max_model_len: int = 2048,
) -> LLM:
    """
    Sync the current HF policy to vLLM using a public API workflow:
    - Save the HF model and tokenizer to a temporary directory
    - Initialize a fresh vLLM LLM instance pointing at that directory
    This avoids relying on internal attributes and works across vLLM versions.
    """
    tmp_dir = os.path.join("./output", "tmp_vllm_sync")
    os.makedirs(tmp_dir, exist_ok=True)
    # Save weights and tokenizer for vLLM to load
    policy.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    # Initialize a new LLM from the saved directory
    vllm_set_random_seed(seed)
    synced_llm = LLM(
        model=tmp_dir,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    return synced_llm


def init_sampling_params(temperature: float, min_tokens: int, max_tokens: int) -> SamplingParams:
    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        logprobs=0,
    )
    sp.stop = ["</answer>"]
    sp.include_stop_str_in_output = True
    return sp


# Tokenization utilities
def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    batch_data = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt)["input_ids"]
        output_tokens = tokenizer(output)["input_ids"]
        combined_tokens = prompt_tokens + output_tokens
        max_len = max(max_len, len(combined_tokens))
        batch_data.append({
            "tokens": combined_tokens, "prompt_len": len(prompt_tokens), "total_len": len(combined_tokens)
        })
    batch_size = len(batch_data)
    input_ids = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), tokenizer.eos_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)
    for i, data in enumerate(batch_data):
        tokens, seq_len = torch.tensor(data["tokens"]), len(data["tokens"])
        input_ids[i, :seq_len-1], labels[i, :seq_len-1] = tokens[:-1], tokens[1:]
        response_start, response_end = data["prompt_len"] - 1, seq_len - 1
        if response_end > response_start:
            response_mask[i, response_start:response_end] = True
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}



# Reward function and evaluation (Countdown solution)

# ==============================================================================
# TASK 1: Implement Reward Helper Functions
# Before you can write the reward function, you need to create three helper 
# functions to parse and evaluate the model's generated text.
# ==============================================================================

def _extract_answer(solution_str: str) -> str | None:
    """
    Extract the content from the last <answer>...</answer> tag in the given string.

    Hint: Use the `re` module. `re.finditer` can find all occurrences of a pattern.
    Args:
        solution_str: The full text generated by the model.

    Returns:
        The stripped string content of the last answer tag, or None if no tag is found.
    """
    ### YOUR CODE HERE ###
    try:
        matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            return None
        last = matches[-1].group(1)
        return last.strip()
    except Exception:
        return None
    ### END YOUR CODE ###


def _validate_numbers(equation_str: str, available_numbers: List[int]) -> bool:
    """
    Check if the numbers used in the equation are exactly the ones available.

    Hint:
    1. Use `re.findall(r"\d+", equation_str)` to get all number strings from the equation.
    2. Remember to handle potential errors and return False.

    Args:
        equation_str: The equation string to validate.
        available_numbers: A list of integers that are allowed to be used.

    Returns:
        True if the equation uses the correct numbers, False otherwise.
    """
    ### YOUR CODE HERE ###
    try:
        found = re.findall(r"\d+", equation_str)
        if not found:
            return True  # no numbers used -> valid as a (empty) subset
        found_ints = [int(x) for x in found]
        found_cnt = Counter(found_ints)
        avail_cnt = Counter(available_numbers)
        # each used number must exist in avail and not exceed counts
        for num, cnt in found_cnt.items():
            if avail_cnt.get(num, 0) < cnt:
                return False
        return True
    except Exception:
        return False
    ### END YOUR CODE ###


def _evaluate_equation(equation_str: str) -> float | None:
    """
    Safely evaluate a mathematical equation string.
    Args:
        equation_str: The equation string to evaluate.
    Returns:
        The result of the equation as a float, or None if it's invalid or unsafe.
    """
    ### YOUR CODE HERE ###
    if equation_str is None:
        return None
    equation_str = equation_str.strip()
    if equation_str == "":
        return None
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Expr, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Mod, ast.Pow
    )
    try:
        node = ast.parse(equation_str, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, allowed_nodes):
                return None
        # Evaluate using a safe eval of compiled AST
        compiled = compile(node, "<ast>", "eval")
        result = eval(compiled, {"__builtins__": {}})
        return float(result)
    except Exception:
        return None
    ### END YOUR CODE ###



# ==============================================================================
# TASK 2: Implement the Reward Function
# ==============================================================================
def reward_fn(generated_text: str, ground_truth: Dict) -> float:
    """
    Reward function for countdown problems.
    - 1.0 (Perfect): The equation is valid, uses the correct numbers, and evaluates to the target.
    - 0.1 (Partial): The text contains an <answer> tag, but the equation is incorrect for any reason.
    - 0.0 (Failed): The `generated_text` does not contain an <answer> tag.

    Args:
        generated_text: The full text output from the language model.
        ground_truth: A dictionary containing `target` and `numbers`.

    Returns:
        A float value representing the reward, such as 1.0, 0.1, or 0.0
    """
    ### YOUR CODE HERE ###
    
    ans = _extract_answer(generated_text)
    if ans is None:
        return 0.0
    # Check numbers first
    numbers = ground_truth.get("numbers", [])
    target = ground_truth.get("target")
    if not _validate_numbers(ans, numbers):
        return 0.1
    result = _evaluate_equation(ans)
    if result is None:
        return 0.1
    try:
        target_f = float(target)
    except Exception:
        return 0.1
    if abs(result - target_f) <= 1e-3:
        return 1.0
    return 0.1
    ### END YOUR CODE ###


def evaluate_model(llm: LLM, sampling_params: SamplingParams, eval_prompts: List[str], eval_answers: List[Dict]) -> Dict[str, Any]:
    rollouts = llm.generate(eval_prompts, sampling_params)
    examples, rewards, output_token_lengths = [], [], []
    for rollout, gt in zip(rollouts, eval_answers):
        response_text = rollout.outputs[0].text
        reward_value = reward_fn(response_text, gt)
        equation = _extract_answer(response_text)
        result = _evaluate_equation(equation) if equation is not None else None
        output_tokens = len(llm.get_tokenizer().encode(response_text))
        output_token_lengths.append(output_tokens)
        examples.append({
            "prompt": rollout.prompt, "response": response_text, "answer": gt, "equation": equation,
            "result": result, "reward": reward_value, "output_tokens": output_tokens,
        })
        rewards.append(reward_value)
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_partial = sum(1 for r in rewards if abs(r - 0.1) < tol)
    count_failed = sum(1 for r in rewards if abs(r - 0.0) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards), "examples": examples, "count_correct": count_correct,
        "count_partial": count_partial, "count_failed": count_failed, "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def evaluate_model_hf(policy: PreTrainedModel, tokenizer: AutoTokenizer, sampling_params: SamplingParams, eval_prompts: List[str], eval_answers: List[Dict], batch_size: int = 16) -> Dict[str, Any]:
    """Evaluate using HF generate() to avoid spinning up a vLLM engine on the same GPU as training.
    Generates one sample per prompt with the same sampling configuration.
    """
    device = next(policy.parameters()).device
    examples, rewards, output_token_lengths = [], [], []
    stop_str = "</answer>"
    gen_kwargs = dict(
        do_sample=True,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        min_new_tokens=getattr(sampling_params, "min_tokens", 0) or 0,
        max_new_tokens=sampling_params.max_tokens,
        use_cache=True,
    )
    policy.eval()
    with torch.no_grad():
        for i in range(0, len(eval_prompts), batch_size):
            batch_prompts = eval_prompts[i:i+batch_size]
            batch_answers = eval_answers[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = policy.generate(**inputs, **gen_kwargs)
            for j, prompt in enumerate(batch_prompts):
                prompt_len = inputs["input_ids"][j].shape[-1]
                full_ids = output_ids[j]
                gen_ids = full_ids[prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if stop_str in text:
                    idx = text.find(stop_str) + (len(stop_str) if getattr(sampling_params, "include_stop_str_in_output", False) else 0)
                    text = text[:idx]
                gt = batch_answers[j]
                reward_value = reward_fn(text, gt)
                equation = _extract_answer(text)
                result = _evaluate_equation(equation) if equation is not None else None
                output_tokens = len(tokenizer.encode(text))
                output_token_lengths.append(output_tokens)
                examples.append({
                    "prompt": prompt, "response": text, "answer": gt, "equation": equation,
                    "result": result, "reward": reward_value, "output_tokens": output_tokens,
                })
                rewards.append(reward_value)
    policy.train()
    rewards_tensor = torch.tensor(rewards) if rewards else torch.tensor([0.0])
    tol = 1e-8
    count_correct = sum(1 for r in rewards if abs(r - 1.0) < tol)
    count_partial = sum(1 for r in rewards if abs(r - 0.1) < tol)
    count_failed = sum(1 for r in rewards if abs(r - 0.0) < tol)
    accuracy = (count_correct / len(rewards)) * 100 if rewards else 0.0
    avg_output_tokens = sum(output_token_lengths) / len(output_token_lengths) if output_token_lengths else 0.0
    return {
        "mean_reward": float(rewards_tensor.mean().item()),
        "std_reward": float(rewards_tensor.std().item()) if rewards_tensor.numel() > 1 else 0.0,
        "num_examples": len(rewards), "examples": examples, "count_correct": count_correct,
        "count_partial": count_partial, "count_failed": count_failed, "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
    }


def _format_eval_example(example: Dict[str, Any]) -> str:
    target = example["answer"]["target"] if isinstance(example.get("answer"), dict) and "target" in example["answer"] else "?"
    numbers = example["answer"].get("numbers") if isinstance(example.get("answer"), dict) else None
    return (
        f"Prompt: {example.get('prompt', '')}\n"
        f"Response: {example.get('response', '')}\n"
        f"Equation: {example.get('equation', None)} | Result: {example.get('result', None)} | Target: {target} | Numbers: {numbers}\n"
        f"Reward: {example.get('reward', 0.0):.3f}\n"
    )


def log_train(rollout_batch_loss: float, grad_norm: float, reward_metadata: Dict[str, Any], avg_output_tokens: float, writer: SummaryWriter | None, step: int) -> None:
    writer.add_scalar("train/loss", float(rollout_batch_loss), global_step=step)
    writer.add_scalar("train/grad_norm", float(grad_norm), global_step=step)
    writer.add_scalar("train/reward_mean", float(reward_metadata["mean"]), global_step=step)
    writer.add_scalar("train/reward_std", float(reward_metadata["std"]), global_step=step)
    writer.add_scalar("train/avg_output_tokens", float(avg_output_tokens), global_step=step)
    print(f"Step {step} | Loss: {rollout_batch_loss:.4f} | Grad norm: {grad_norm:.4f} | Reward mean: {float(reward_metadata['mean']):.4f} | Reward std: {float(reward_metadata['std']):.4f} | Avg output tokens: {avg_output_tokens:.1f}")


def log_eval(metrics: Dict[str, Any], writer: SummaryWriter | None, step: int) -> None:
    examples = metrics.get("examples", []) or []
    if not examples: return
    tol = 1e-8
    correct_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 1.0) < tol][:10]
    partial_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.1) < tol][:10]
    failed_examples = [ex for ex in examples if abs(float(ex.get("reward", 0.0)) - 0.0) < tol][:10]
    if correct_examples:
        print(f"\n=== Eval examples (CORRECT, reward=1.0) @ step {step} ===")
        for idx, ex in enumerate(correct_examples[:2], 1): print(f"[CORRECT #{idx}]\n" + _format_eval_example(ex))
    if partial_examples:
        print(f"\n=== Eval examples (PARTIAL, reward=0.1) @ step {step} ===")
        for idx, ex in enumerate(partial_examples[:2], 1): print(f"[PARTIAL #{idx}]\n" + _format_eval_example(ex))
    if failed_examples:
        print(f"\n=== Eval examples (FAILED, reward=0.0) @ step {step} ===")
        for idx, ex in enumerate(failed_examples[:2], 1): print(f"[FAILED #{idx}]\n" + _format_eval_example(ex))
    if writer:
        correct_text = "\n\n".join([_format_eval_example(ex) for ex in correct_examples]) or ""
        partial_text = "\n\n".join([_format_eval_example(ex) for ex in partial_examples]) or ""
        failed_text = "\n\n".join([_format_eval_example(ex) for ex in failed_examples]) or ""
        if correct_text: writer.add_text("eval/examples_correct", correct_text, global_step=step)
        if partial_text: writer.add_text("eval/examples_partial", partial_text, global_step=step)
        if failed_text: writer.add_text("eval/examples_failed", failed_text, global_step=step)
    print(f"Eval @ step {step}: accuracy={metrics['accuracy']:.1f}% mean_reward={metrics['mean_reward']:.4f} "
          f"avg_tokens={metrics['avg_output_tokens']:.1f} | correct:{metrics['count_correct']} "
          f"partial:{metrics['count_partial']} failed:{metrics['count_failed']}")


# -------------------------
# Advantages and GRPO-clip loss
# -------------------------
# ==============================================================================
# TASK 3: Implement Group Normalized Advantages
# ==============================================================================
def compute_group_normalized_advantages(
    rollout_responses: List[str],
    repeated_ground_truths: List[Dict],
    reward_fn: Callable[[str, Dict], float],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """ Computes advantages by normalizing rewards within groups.
    
    Args:
        rollout_responses: List of generated responses (length = batch_size * group_size)
        repeated_ground_truths: Ground truth repeated for each response
        reward_fn: Function to compute rewards
        group_size: Number of responses per question (G)
        advantage_eps: Small constant for numerical stability (epsilon)
        normalize_by_std: If True, normalize by std (GRPO); if False, don't (DR-GRPO)
    
    Returns:
        advantages: Flattened tensor of advantages (shape: [batch_size * group_size])
        raw_rewards: Original rewards before normalization
        metadata: Dictionary with 'mean', 'std', 'max', 'min' of raw rewards
            (e.g. metadata = {
                "mean": torch.mean(raw_rewards),
                "std": torch.std(raw_rewards),
                "max": torch.max(raw_rewards),
                "min": torch.min(raw_rewards),
            })
    """

    # Steps:
    # 1. Calculate the raw reward for each response using the provided `reward_fn`.
    # 2. Reshape the 1D tensor of raw rewards into a 2D tensor of shape (-1, `group_size`).
    # 3. Calculate the mean reward for each group.
    # 4. Compute the advantage by subtracting the group's mean reward.
    # 5. If `normalize_by_std` is True, normalize advantages by `(group_std + advantage_eps)`.
    # 6. Flatten the advantages tensor back into a 1D tensor.
    # 7. Create a `metadata` dictionary with overall statistics of the raw rewards.
    advantages, raw_rewards, metadata = None, None, {}
    ### YOUR CODE HERE ###
    # 1. raw rewards
    raw = [float(reward_fn(resp, gt)) for resp, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(raw, dtype=torch.float32)
    if raw_rewards.numel() == 0:
        advantages = torch.tensor([], dtype=torch.float32)
        metadata = {"mean": torch.tensor(0.0), "std": torch.tensor(0.0), "max": torch.tensor(0.0), "min": torch.tensor(0.0)}
        return advantages, raw_rewards, metadata
    # reshape into (-1, 'groupsize')
    assert raw_rewards.numel() % group_size == 0, "raw_rewards length must be divisible by group_size"
    rewards_2d = raw_rewards.view(-1, group_size)
    # 3. group means
    group_means = rewards_2d.mean(dim=1, keepdim=True)
    # 4. advantages = reward - group_mean
    adv = rewards_2d - group_means
    # 5. normalize by std if needed
    if normalize_by_std:
        group_std = rewards_2d.std(dim=1, unbiased=False, keepdim=True)
        adv = adv / (group_std + advantage_eps)
    # 6. flatten
    advantages = adv.view(-1)
    # 7. metadata
    metadata = {
        "mean": torch.mean(raw_rewards),
        "std": torch.std(raw_rewards) if raw_rewards.numel() > 1 else torch.tensor(0.0),
        "max": torch.max(raw_rewards),
        "min": torch.min(raw_rewards),
    }
    ### END YOUR CODE ###
    return advantages, raw_rewards, metadata


# ==============================================================================
# TASK 4: Implement per-token Loss
# ==============================================================================
def compute_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the per-token PPO clipped surrogate loss.

    Why omit the KL divergence term?
    For simplicity and following trends in recent RLVR (e.g Dr.GRPO),
    we omit the KL penalty term often found in PPO. This simplifies the implementation
    and has been shown to work well in practice.

    Steps:
    1. Calculate the probability ratio `pi_ratio = exp(policy_log_probs - old_log_probs)`.
    2. Calculate the unclipped term: `advantages * pi_ratio`.
    3. Calculate the clipped term by clipping `pi_ratio` to `[1-clip_range, 1+clip_range]`
       and then multiplying by `advantages`.
    4. The final loss is `-torch.minimum(unclipped_term, clipped_term)`.
    """
    loss = 0.0
    ### YOUR CODE HERE ###
    # pi_ratio = exp(policy_log_probs - old_log_probs)
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_term = advantages * pi_ratio
    clipped_ratio = torch.clamp(pi_ratio, 1.0 - clip_range, 1.0 + clip_range)
    clipped_term = advantages * clipped_ratio
    # loss per token
    loss_per_token = -torch.min(unclipped_term, clipped_term)
    metadata = {
        "mean_unclipped": torch.mean(unclipped_term).detach(),
        "mean_clipped": torch.mean(clipped_term).detach(),
        "mean_ratio": torch.mean(pi_ratio).detach(),
    }
    return loss_per_token, metadata
    ### END YOUR CODE ###

# ==============================================================================
# TASK 5: masked mean for grpo and dr-grpo
# ==============================================================================

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of tensor values where mask=True for each row, then average across the batch.
    """
    ### YOUR CODE HERE ###
    tensor = tensor.to(torch.float32)
    mask = mask.to(torch.bool)
    # sum per row where mask is True
    masked_sum = (tensor * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    row_means = masked_sum / counts
    return row_means.mean()
    ### END YOUR CODE ###

def masked_mean_drgrpo(tensor: torch.Tensor, mask: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Compute the sum of tensor values where mask=True, divided by num_tokens, then average across the batch.
    This is used for the DR-GRPO loss
    """
    ### YOUR CODE HERE ###
    tensor = tensor.to(torch.float32)
    mask = mask.to(torch.bool)
    masked_sum = (tensor * mask).sum(dim=1)
    per_row = masked_sum / float(num_tokens)
    return per_row.mean()
    ### END YOUR CODE ###

def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs


# Training helpers and loop
def duplicate_data(arr: List, group_size: int) -> List:
    return [x for x in arr for _ in range(group_size)]


def rollout_with_vllm(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    prompts_batch: List[str],
    group_size: int,
    *,
    seed: int,
    gpu_memory_utilization: float = 0.4,
    max_num_seqs: int | None = None,
    max_model_len: int = 2048,
) -> Tuple[List[str], List[str], List[int]]:
    # Sync current policy into a fresh vLLM instance
    llm = load_policy_into_vllm_instance(
        policy, tokenizer, seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    prompts_dup = duplicate_data(prompts_batch, group_size)
    vllm_rollouts = llm.generate(prompts_dup, sampling_params, use_tqdm=False)
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for rollout in vllm_rollouts:
        for r in rollout.outputs:
            rollout_input_text.append(rollout.prompt)
            rollout_response_text.append(r.text)
            rollout_output_tokens.append(len(llm.get_tokenizer().encode(r.text)))
    # Explicitly release the vLLM engine to free GPU memory for HF policy and next steps
    try:
        llm.shutdown()
    except Exception:
        pass
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def tokenize_rollouts(rollout_input_text: List[str], rollout_response_text: List[str], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(rollout_input_text, rollout_response_text, tokenizer)


def rollout_with_hf(
    policy: PreTrainedModel,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    prompts_batch: List[str],
    group_size: int,
) -> Tuple[List[str], List[str], List[int]]:
    """Generate rollouts using HF generate() to avoid extra vLLM engine memory during training.
    Mirrors sampling settings: temperature, top_p, min_tokens, max_tokens, and applies stop string "</answer>".
    """
    device = next(policy.parameters()).device
    prompts_dup = duplicate_data(prompts_batch, group_size)
    inputs = tokenizer(prompts_dup, return_tensors="pt", padding=True, truncation=True).to(device)
    gen_kwargs = dict(
        do_sample=True,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        min_new_tokens=getattr(sampling_params, "min_tokens", 0) or 0,
        max_new_tokens=sampling_params.max_tokens,
        use_cache=True,
    )
    with torch.no_grad():
        output_ids = policy.generate(**inputs, **gen_kwargs)
    # Extract only the new tokens beyond the prompt and apply stop trimming
    rollout_input_text, rollout_response_text, rollout_output_tokens = [], [], []
    for i, prompt in enumerate(prompts_dup):
        prompt_len = inputs["input_ids"][i].shape[-1]
        full_ids = output_ids[i]
        gen_ids = full_ids[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Apply stop string trimming similar to vLLM behavior
        stop_str = "</answer>"
        if stop_str in text:
            idx = text.find(stop_str) + (len(stop_str) if getattr(sampling_params, "include_stop_str_in_output", False) else 0)
            text = text[:idx]
        rollout_input_text.append(prompt)
        rollout_response_text.append(text)
        rollout_output_tokens.append(len(tokenizer.encode(text)))
    return rollout_input_text, rollout_response_text, rollout_output_tokens


def grpo_microbatch_step(
    policy: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor,
    advantages_per_seq: torch.Tensor, gradient_accumulation_steps: int, clip_range: float,
    loss_type: str = "grpo", max_completion_length: int = 256,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    policy_log_probs = get_response_log_probs(policy, input_ids, labels)
    old_log_probs = policy_log_probs.detach()
    advantages = advantages_per_seq.unsqueeze(-1)
    loss_per_token, metadata = compute_loss(advantages, policy_log_probs, old_log_probs, clip_range)
    if loss_type == "grpo":
        loss = masked_mean(loss_per_token, response_mask)
    elif loss_type == "dr_grpo":
        loss = masked_mean_drgrpo(loss_per_token, response_mask, max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def train(
    policy: PreTrainedModel, tokenizer: AutoTokenizer, sampling_params: SamplingParams, *,
    train_prompts: List[str], train_answers: List[Dict], eval_prompts: List[str], eval_answers: List[Dict],
    optimizer: torch.optim.Optimizer, scheduler, n_grpo_steps: int, rollout_batch_size: int,
    group_size: int, gradient_accumulation_steps: int, clip_range: float, use_std_normalization: bool,
    advantage_eps: float, device: str, eval_every: int = 5, writer: SummaryWriter = None, seed: int,
    loss_type: str = "grpo", max_completion_length: int = 256,
    vllm_gpu_memory_utilization: float = 0.4, vllm_max_num_seqs: int | None = None,
    vllm_max_model_len: int = 2048,
) -> None:
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = rollout_batch_size // gradient_accumulation_steps
    random.seed(seed)
    train_step = 0

    # Initial eval: use HF generate to avoid duplicating weights on GPU
    metrics = evaluate_model_hf(policy, tokenizer, sampling_params, eval_prompts, eval_answers)
    if writer:
        for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct", "count_partial", "count_failed"]:
            writer.add_scalar(f"eval/{k}", metrics[k], global_step=train_step)
        log_eval(metrics, writer, train_step)

    for _ in range(n_grpo_steps):
        sampled = random.sample(list(zip(train_prompts, train_answers)), n_prompts_per_rollout_batch)
        prompts_batch, answers_batch = [p for p, _ in sampled], [a for _, a in sampled]
        rollout_input, rollout_response, rollout_tokens = rollout_with_hf(
            policy, tokenizer, sampling_params, prompts_batch, group_size,
        )
        answers_dup = duplicate_data(answers_batch, group_size)
        avg_output_tokens = sum(rollout_tokens) / len(rollout_tokens) if rollout_tokens else 0.0
        advantages, _, reward_meta = compute_group_normalized_advantages(
            rollout_response, answers_dup, reward_fn, group_size, advantage_eps, use_std_normalization
        )
        tokenized = tokenize_rollouts(rollout_input, rollout_response, tokenizer)
        optimizer.zero_grad()
        rollout_loss = 0.0
        for micro_idx in range(0, rollout_batch_size, micro_train_batch_size):
            s = slice(micro_idx, micro_idx + micro_train_batch_size)
            loss, _ = grpo_microbatch_step(
                policy, tokenized["input_ids"][s].to(device), tokenized["labels"][s].to(device),
                tokenized["response_mask"][s].to(device), advantages[s].to(device),
                gradient_accumulation_steps, clip_range, loss_type=loss_type, max_completion_length=max_completion_length
            )
            rollout_loss += float(loss.item())
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.grad is not None], 1.0)
        optimizer.step()
        scheduler.step()
        rollout_loss /= (rollout_batch_size / micro_train_batch_size)
        train_step += 1
        print(f"Step {train_step} | Loss: {rollout_loss:.4f} | Grad: {grad_norm:.4f} | "
              f"Reward mean: {float(reward_meta['mean']):.4f} | Reward std: {float(reward_meta['std']):.4f}")
        log_train(rollout_loss, grad_norm, reward_meta, avg_output_tokens, writer, train_step)
        if train_step % eval_every == 0:
            metrics = evaluate_model_hf(policy, tokenizer, sampling_params, eval_prompts, eval_answers)
            log_eval(metrics, writer, train_step)


def init_policy(model_id: str, device: str) -> Tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device).train()
    return model, tokenizer


def main() -> None:
    # Hyperparameters
    model_id = "Qwen/Qwen3-1.7B"
    device = "cuda"
    seed = 42
    # Allow env overrides for stability on single GPU
    # Separate GPU memory utilization for eval-time LLM vs rollout-time LLM to avoid contention
    gpu_mem_util = float(os.getenv("VLLM_MEM_UTIL_EVAL", os.getenv("VLLM_MEM_UTIL", "0.25")))
    gpu_mem_util_rollout = float(os.getenv("VLLM_MEM_UTIL_ROLLOUT", os.getenv("VLLM_MEM_UTIL", "0.10")))
    vllm_max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "48"))
    vllm_max_model_len_eval = int(os.getenv("VLLM_MAX_MODEL_LEN_EVAL", "2048"))
    vllm_max_model_len_rollout = int(os.getenv("VLLM_MAX_MODEL_LEN_ROLLOUT", "1024"))
    n_grpo_steps = 80
    rollout_batch_size = int(os.getenv("ROLLOUT_BATCH_SIZE", "128"))
    group_size, grad_acc_steps = 8, 32
    lr, clip_range, adv_eps = 7e-6, 0.2, 1e-6
    temperature, min_tokens = 1.0, 4
    eval_every = 10

    # CHANGING HYPERPARAMETERS for main assignment
    loss_type = "grpo" # or "dr_grpo"
    max_tokens = int(os.getenv("MAX_TOKENS", "256")) # or 512, 1024
    
    # Initialization
    use_std_norm = loss_type == "grpo"
    policy, tokenizer = init_policy(model_id=model_id, device=device)
    sampling_params = init_sampling_params(temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens)
    
    # Dataset
    def build_dataset(split):
        data = []
        for ex in split:
            prompt = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=max_tokens)
            prompt = tokenizer.apply_chat_template(
                [dict(role="system", content="You are a helpful assistant."),
                dict(role="user", content=prompt)],
                add_generation_prompt=True, tokenize=False)
            data.append({"prompt": prompt,"answer": {"target": ex["target"], "numbers": ex["nums"]},})
        return data

    # Load properly split dataset
    train_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="train")
    eval_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    
    train_examples = build_dataset(train_data)
    eval_examples = build_dataset(eval_data)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Logging
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join("./output", "tb", f"hw_a2_{loss_type}", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training
    train(
        policy=policy, tokenizer=tokenizer, sampling_params=sampling_params,
        train_prompts=[ex["prompt"] for ex in train_examples], train_answers=[ex["answer"] for ex in train_examples],
        eval_prompts=[ex["prompt"] for ex in eval_examples], eval_answers=[ex["answer"] for ex in eval_examples],
        optimizer=optimizer, scheduler=scheduler, n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size, group_size=group_size,
        gradient_accumulation_steps=grad_acc_steps, clip_range=clip_range,
        use_std_normalization=use_std_norm, advantage_eps=adv_eps, device=device,
        eval_every=eval_every, writer=writer, seed=seed, loss_type=loss_type,
        max_completion_length=max_tokens,
        vllm_gpu_memory_utilization=gpu_mem_util_rollout, vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_max_model_len=vllm_max_model_len_rollout,
    )
    
    # Save model
    out_dir = os.path.join("./output", f"hw_a2_solution_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model and tokenizer to {out_dir}")
    writer.close()

if __name__ == "__main__":
    main()
