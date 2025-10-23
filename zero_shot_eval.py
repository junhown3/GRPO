
import os, re, argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from dotenv import load_dotenv
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from starter import reward_fn

SEED = 42
vllm_set_random_seed(SEED)

os.environ["VLLM_USE_V1"] = "1"

TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your reasoning in <think> </think> tags. And return the final equation in <answer> </answer> tags. Keep your reasoning under {max_tokens} tokens.
For example, numbers = [1, 2, 3, 4] and target = 5, the answer is <answer>(1 + 2) * 3 - 4</answer>."""

def is_correct(response: str, target: float, numbers: list) -> bool:
    """Check if response contains correct equation."""
    return reward_fn(response, {"target": target, "numbers": numbers}) == 1.

def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of math reasoning")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-1.7B", help="Model ID to evaluate")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens for generation (default: 256)")
    args = parser.parse_args()
    
    MODEL_ID = args.model
    MAX_TOKENS = args.max_tokens
    
    print(f"Evaluating {MODEL_ID} with max_tokens={MAX_TOKENS}...")
    
    # Load model
    llm = LLM(model=MODEL_ID, dtype=torch.bfloat16, max_model_len=2048, enable_prefix_caching=True, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, logprobs=0, max_tokens=MAX_TOKENS, stop=["</answer>"], include_stop_str_in_output=True)
    
    # Load test split from properly split dataset
    eval_raw = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    
    # Prepare prompts
    prompts = []
    targets = []
    numbers_list = []
    
    for ex in eval_raw:
        prompt_content = TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=MAX_TOKENS)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt_content}],
            add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)
        targets.append(float(ex["target"]))
        numbers_list.append(ex["nums"])
    
    # Generate and evaluate
    print(f"Generating {len(prompts)} responses...")
    responses = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    correct = 0
    for i, response in enumerate(responses):
        if is_correct(response.outputs[0].text, targets[i], numbers_list[i]):
            correct += 1
    
    accuracy = (correct / len(responses)) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{len(responses)})")

if __name__ == "__main__":
    main()
