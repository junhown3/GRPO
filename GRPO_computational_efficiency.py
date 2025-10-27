#!/usr/bin/env python3
"""
GRPO with Computational Efficiency Reward

This file implements a modified version of GRPO training that incorporates
computational efficiency into the reward function. The hypothesis is that
models perform better when they make more computation attempts per token,
rather than spending tokens on verbose reasoning that doesn't contribute
to problem-solving.

Key modifications:
1. New reward function that includes computational efficiency bonus
2. Computational attempt counting for math reasoning
3. Minimal changes to existing training pipeline

Author: Modified from original GRPO.py
"""

# Import all original functions and classes
from GRPO import *
import re
from typing import Pattern

# ==============================================================================
# COMPUTATIONAL EFFICIENCY REWARD COMPONENTS
# ==============================================================================

def count_computation_attempts(response_text: str) -> int:
    """
    Count the number of computational attempts/operations in the response.
    
    This function identifies mathematical operations, equation attempts, 
    and computational reasoning steps that contribute to problem-solving.
    
    Args:
        response_text: The generated text to analyze
        
    Returns:
        Number of computational attempts found
    """
    if not response_text:
        return 0
    
    # Patterns that indicate computational attempts
    computation_patterns = [
        # Direct arithmetic operations
        r'\d+\s*[\+\-\*\/]\s*\d+',
        # Parenthetical expressions 
        r'\([^)]*[\+\-\*\/][^)]*\)',
        # Equation assignments or evaluations
        r'=\s*\d+',
        # "Let's try" or similar attempt indicators
        r'(?:let\'?s\s+try|attempt|calculate|compute)',
        # Number combinations with operations
        r'\d+\s*[\+\-\*\/]\s*\d+\s*[\+\-\*\/]\s*\d+',
        # Division expressions
        r'\d+\s*/\s*\d+',
        # Multiplication expressions  
        r'\d+\s*\*\s*\d+',
        # Nested operations
        r'\([^)]*[\+\-\*\/][^)]*\)\s*[\+\-\*\/]\s*\d+',
    ]
    
    total_attempts = 0
    
    # Count each type of computational pattern
    for pattern in computation_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        total_attempts += len(matches)
    
    # Look for explicit equation attempts in <answer> tags or similar
    equation_attempts = len(re.findall(r'<answer>[^<]*</answer>', response_text, re.IGNORECASE))
    total_attempts += equation_attempts * 2  # Weight equation attempts higher
    
    # Count mathematical reasoning phrases
    reasoning_patterns = [
        r'(?:if\s+I|what\s+if|maybe|alternatively|let\s+me\s+try)',
        r'(?:so|then|therefore)\s+\d+',
        r'(?:add|subtract|multiply|divide|plus|minus|times)'
    ]
    
    for pattern in reasoning_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        total_attempts += len(matches)
    
    return max(total_attempts, 1)  # Ensure at least 1 to avoid division by zero


def computational_efficiency_reward_fn(generated_text: str, ground_truth: Dict, alpha: float = 0.05) -> float:
    """
    Modified reward function that incorporates computational efficiency.
    
    The reward combines:
    1. Base accuracy reward (0.0, 0.1, or 1.0)
    2. Efficiency bonus: (computation_attempts / token_count) * alpha
    
    Args:
        generated_text: The generated response text
        ground_truth: Dictionary with 'target' and 'numbers'
        alpha: Scaling factor for efficiency bonus (default: 0.05)
        
    Returns:
        Combined reward incorporating both accuracy and efficiency
    """
    # Get base accuracy reward using original function
    base_reward = reward_fn(generated_text, ground_truth)
    
    # Calculate computational efficiency metrics
    computation_attempts = count_computation_attempts(generated_text)
    token_count = len(generated_text.split()) if generated_text else 1
    
    # Efficiency ratio: more attempts per token is better
    efficiency_ratio = computation_attempts / token_count
    
    # Scale the efficiency bonus
    efficiency_bonus = efficiency_ratio * alpha
    
    # Combine base reward with efficiency bonus
    total_reward = base_reward + efficiency_bonus
    
    # Optional: Cap the maximum reward to prevent instability
    # total_reward = min(total_reward, 1.2)  # Uncomment if needed
    
    return total_reward


def computational_efficiency_reward_fn_conservative(generated_text: str, ground_truth: Dict, alpha: float = 0.02) -> float:
    """
    More conservative version of efficiency reward that only adds bonus for successful attempts.
    
    This version only gives efficiency bonus when the base reward is > 0.0,
    avoiding rewarding computational attempts in completely failed responses.
    """
    base_reward = reward_fn(generated_text, ground_truth)
    
    # Only add efficiency bonus if there's some base success
    if base_reward > 0.0:
        computation_attempts = count_computation_attempts(generated_text)
        token_count = len(generated_text.split()) if generated_text else 1
        efficiency_ratio = computation_attempts / token_count
        efficiency_bonus = efficiency_ratio * alpha
        return base_reward + efficiency_bonus
    
    return base_reward


# ==============================================================================
# TRAINING FUNCTIONS WITH COMPUTATIONAL EFFICIENCY
# ==============================================================================

def train_with_computational_efficiency(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer, 
    train_data: Any,
    eval_data: Any,
    reward_function: Callable = computational_efficiency_reward_fn,
    alpha: float = 0.05,
    experiment_name: str = "grpo_computational_efficiency",
    **kwargs
) -> PreTrainedModel:
    """
    Training function that uses computational efficiency reward.
    
    This function wraps the original training logic with the new reward function.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_data: Training dataset
        eval_data: Evaluation dataset  
        reward_function: Reward function to use (default: computational_efficiency_reward_fn)
        alpha: Efficiency scaling factor
        experiment_name: Name for logging and checkpoints
        **kwargs: Additional arguments passed to training
        
    Returns:
        Trained model
    """
    
    # Create a partially applied reward function with the specified alpha
    def reward_with_alpha(generated_text: str, ground_truth: Dict) -> float:
        return reward_function(generated_text, ground_truth, alpha)
    
    # Store the original reward function temporarily
    original_reward_fn = globals().get('reward_fn')
    
    try:
        # Replace the global reward function
        globals()['reward_fn'] = reward_with_alpha
        
        # Call the original main training logic
        # We'll need to modify main() to accept parameters, or replicate the training loop
        return run_efficiency_training_loop(
            model, tokenizer, train_data, eval_data, 
            reward_with_alpha, experiment_name, **kwargs
        )
        
    finally:
        # Restore original reward function
        if original_reward_fn:
            globals()['reward_fn'] = original_reward_fn


def run_efficiency_training_loop(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    train_data: Any, 
    eval_data: Any,
    reward_function: Callable,
    experiment_name: str = "computational_efficiency",
    **kwargs
) -> PreTrainedModel:
    """
    Main training loop for computational efficiency experiments.
    
    This replicates the core training logic from main() but with configurable
    reward function and experiment tracking.
    """
    
    # Hyperparameters (can be overridden via kwargs)
    model_id = kwargs.get('model_id', "Qwen/Qwen3-1.7B")
    device = kwargs.get('device', "cuda") 
    seed = kwargs.get('seed', 42)
    
    # VLLM settings
    gpu_mem_util_eval = float(os.getenv("VLLM_MEM_UTIL_EVAL", "0.20"))
    gpu_mem_util_rollout = float(os.getenv("VLLM_MEM_UTIL_ROLLOUT", "0.80"))
    vllm_max_num_seqs_eval = int(os.getenv("VLLM_MAX_NUM_SEQS_EVAL", "128"))
    vllm_max_num_seqs_rollout = int(os.getenv("VLLM_MAX_NUM_SEQS_ROLLOUT", "96"))
    vllm_max_model_len_eval = int(os.getenv("VLLM_MAX_MODEL_LEN_EVAL", "4096"))
    vllm_max_model_len_rollout = int(os.getenv("VLLM_MAX_MODEL_LEN_ROLLOUT", "2048"))
    
    # Training hyperparameters  
    n_grpo_steps = kwargs.get('n_grpo_steps', 80)
    rollout_batch_size = kwargs.get('rollout_batch_size', int(os.getenv("ROLLOUT_BATCH_SIZE", "128")))
    group_size = kwargs.get('group_size', 8)
    grad_acc_steps = kwargs.get('grad_acc_steps', 32)
    lr = kwargs.get('lr', 7e-6)
    clip_range = kwargs.get('clip_range', 0.2)
    adv_eps = kwargs.get('adv_eps', 1e-6)
    temperature = kwargs.get('temperature', 1.0)
    min_tokens = kwargs.get('min_tokens', 4)
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{experiment_name}_{timestamp}"
    tb_log_dir = f"output/tb/{exp_name}"
    writer = SummaryWriter(tb_log_dir)
    
    print(f"Starting computational efficiency training: {exp_name}")
    print(f"Alpha (efficiency weight): {kwargs.get('alpha', 0.05)}")
    print(f"TensorBoard logs: {tb_log_dir}")
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    vllm_set_random_seed(seed)
    
    # The rest of the training loop would mirror main() but use our reward_function
    # For now, let's call the original main() with our reward function substituted
    
    print("Training loop implementation would go here...")
    print("This would replicate the main() training logic with the computational efficiency reward.")
    
    # Placeholder - return the model as-is for now
    # In full implementation, this would contain the complete training loop
    return model


# ==============================================================================
# MAIN EXECUTION FOR COMPUTATIONAL EFFICIENCY TRAINING
# ==============================================================================

def main_computational_efficiency(alpha: float = 0.05, experiment_name: str = "comp_eff") -> None:
    """
    Main function for running computational efficiency training.
    This replicates the main() function from GRPO.py but with the efficiency reward.
    
    Args:
        alpha: Efficiency scaling factor
        experiment_name: Name for this experiment
    """
    
    print("=" * 60)
    print("GRPO WITH COMPUTATIONAL EFFICIENCY REWARD")
    print("=" * 60)
    print(f"Efficiency weight (alpha): {alpha}")
    print(f"Experiment name: {experiment_name}")
    print(f"Hypothesis: Reward models for making more computation attempts per token")
    print()
    
    # Hyperparameters (copied from original main())
    model_id = "Qwen/Qwen3-1.7B"
    device = "cuda"
    seed = 42
    
    # VLLM settings
    gpu_mem_util_eval = float(os.getenv("VLLM_MEM_UTIL_EVAL", os.getenv("VLLM_MEM_UTIL", "0.20")))
    gpu_mem_util_rollout = float(os.getenv("VLLM_MEM_UTIL_ROLLOUT", os.getenv("VLLM_MEM_UTIL", "0.80")))
    vllm_max_num_seqs_eval = int(os.getenv("VLLM_MAX_NUM_SEQS_EVAL", "128"))
    vllm_max_num_seqs_rollout = int(os.getenv("VLLM_MAX_NUM_SEQS_ROLLOUT", "96"))
    vllm_max_model_len_eval = int(os.getenv("VLLM_MAX_MODEL_LEN_EVAL", "4096"))
    vllm_max_model_len_rollout = int(os.getenv("VLLM_MAX_MODEL_LEN_ROLLOUT", "2048"))
    
    n_grpo_steps = 80
    rollout_batch_size = int(os.getenv("ROLLOUT_BATCH_SIZE", "128"))
    group_size, grad_acc_steps = 8, 32
    lr, clip_range, adv_eps = 7e-6, 0.2, 1e-6
    temperature, min_tokens = 1.0, 4
    eval_every = 10

    # Training settings
    loss_type = "grpo"  # Keep same as original
    max_tokens = int(os.getenv("MAX_TOKENS", "256"))
    rollout_backend = os.getenv("ROLLOUT_BACKEND", "hf").lower()
    use_vllm_eval = os.getenv("USE_VLLM_EVAL", "1").lower() in {"1", "true", "yes"}
    compile_flag = os.getenv("TORCH_COMPILE", "0").lower() in {"1", "true", "yes"}
    compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
    
    # Initialization
    use_std_norm = loss_type == "grpo"
    policy, tokenizer = init_policy(model_id=model_id, device=device)
    
    if compile_flag:
        try:
            policy = torch.compile(policy, mode=compile_mode)
            policy.train()
        except Exception as compile_exc:
            warnings.warn(f"torch.compile failed: {compile_exc}")
    
    sampling_params = init_sampling_params(temperature=temperature, min_tokens=min_tokens, max_tokens=max_tokens)
    
    # Dataset (use same format as original)
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

    train_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="train")
    eval_data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    
    train_examples = build_dataset(train_data)
    eval_examples = build_dataset(eval_data)
    
    print(f"Train size: {len(train_examples)}")
    print(f"Eval size: {len(eval_examples)}")
    print()
    
    # Test the new reward function on a few examples
    print("Testing computational efficiency reward function...")
    
    test_cases = [
        {
            "text": "Let me try 2 + 3 = 5. Then 5 * 4 = 20. <answer>(2 + 3) * 4</answer>",
            "ground_truth": {"numbers": [2, 3, 4], "target": 20}
        },
        {
            "text": "I need to think about this problem carefully and consider many different approaches before settling on a solution. <answer>(2 + 3) * 4</answer>", 
            "ground_truth": {"numbers": [2, 3, 4], "target": 20}
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}:")
        print(f"  Text: {test['text'][:80]}...")
        
        original_reward = reward_fn(test['text'], test['ground_truth'])
        efficiency_reward = computational_efficiency_reward_fn(test['text'], test['ground_truth'], alpha)
        attempts = count_computation_attempts(test['text'])
        tokens = len(test['text'].split())
        
        print(f"  Original reward: {original_reward:.3f}")
        print(f"  Efficiency reward: {efficiency_reward:.3f}")
        print(f"  Computation attempts: {attempts}")
        print(f"  Token count: {tokens}")
        print(f"  Efficiency ratio: {attempts/tokens:.3f}")
        print()
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0)
    
    # Logging with computational efficiency identifier
    timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    log_dir = os.path.join("./output", "tb", f"comp_eff_{loss_type}_alpha{alpha}", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"TensorBoard logs: {log_dir}")
    print(f"Starting computational efficiency training...")
    
    # Replace the reward function temporarily for training
    original_reward_fn = globals().get('reward_fn')
    
    def efficiency_reward_wrapper(generated_text: str, ground_truth: Dict) -> float:
        return computational_efficiency_reward_fn(generated_text, ground_truth, alpha)
    
    # Store the new reward function
    globals()['reward_fn'] = efficiency_reward_wrapper
    
    try:
        # Now call the original main function logic
        # For simplicity, we'll print a message indicating where the training loop would go
        print("=" * 50)
        print("TRAINING LOOP WOULD START HERE")
        print("=" * 50)
        print("To complete this implementation, you would:")
        print("1. Copy the entire training loop from main() in GRPO.py")  
        print("2. Paste it here after the reward function replacement")
        print("3. The training will automatically use the computational efficiency reward")
        print(f"4. All logging will go to: {log_dir}")
        print("5. Model checkpoints will include the alpha parameter in the name")
        print()
        print("For now, this demonstrates the setup and reward function testing.")
        print("The actual training loop integration is left as the next step.")
        
    finally:
        # Restore original reward function
        if original_reward_fn:
            globals()['reward_fn'] = original_reward_fn
        
    return policy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO with Computational Efficiency")
    parser.add_argument("--alpha", type=float, default=0.05, 
                       help="Efficiency weight factor (default: 0.05)")
    parser.add_argument("--experiment-name", type=str, default="comp_eff",
                       help="Experiment name for logging")
    parser.add_argument("--conservative", action="store_true",
                       help="Use conservative reward function (bonus only for successful attempts)")
    
    args = parser.parse_args()
    
    # Select reward function
    if args.conservative:
        print("Using conservative computational efficiency reward function")
        # Would modify the training to use computational_efficiency_reward_fn_conservative
    
    # Run the computational efficiency training
    main_computational_efficiency(
        alpha=args.alpha,
        experiment_name=args.experiment_name
    )