#!/usr/bin/env python3
"""
Test script for computational efficiency reward function.
"""

from GRPO_computational_efficiency import (
    computational_efficiency_reward_fn, 
    count_computation_attempts,
    computational_efficiency_reward_fn_conservative
)
from GRPO import reward_fn

def test_computational_efficiency():
    """Test the computational efficiency reward function with example cases."""
    
    print("Testing Computational Efficiency Reward Function")
    print("=" * 50)
    
    # Test cases with different levels of computational efficiency
    test_cases = [
        {
            "name": "High Efficiency (Multiple Operations)",
            "text": """<think>
Let me try: 10 + 5 = 15. Then 15 * 2 = 30. So 10 + 5 * 2 = 20. 
Wait, that's not right due to order of operations. Let me try (10 + 5) * 2 = 30.
Actually, let me calculate: 25 - 7 = 18. That's closer to 21.
Maybe 25 - 4 = 21. Yes!
</think>
<answer>25 - 4</answer>""",
            "ground_truth": {"numbers": [25, 4], "target": 21}
        },
        {
            "name": "Medium Efficiency (Some Operations)", 
            "text": """<think>
I need to find a way to get 21 from these numbers. Let me think...
Maybe I can subtract 4 from 25? 25 - 4 = 21. That works!
</think>
<answer>25 - 4</answer>""",
            "ground_truth": {"numbers": [25, 4], "target": 21}
        },
        {
            "name": "Low Efficiency (Verbose Reasoning)",
            "text": """<think>
This is a challenging problem that requires careful consideration of all possible approaches. 
I need to think systematically about how to combine these numbers. Let me consider various 
strategies and methodologies that might be applicable to this type of mathematical reasoning task.
After much deliberation, I believe the answer is straightforward.
</think>
<answer>25 - 4</answer>""",
            "ground_truth": {"numbers": [25, 4], "target": 21}
        },
        {
            "name": "Failed Attempt (No Answer)",
            "text": """<think>
Let me try various combinations: 25 + 4 = 29, that's too big.
25 / 4 = 6.25, that's too small. 25 * 4 = 100, way too big.
Hmm, this is tricky...
</think>
I'm not sure how to solve this.""",
            "ground_truth": {"numbers": [25, 4], "target": 21}
        },
        {
            "name": "Wrong Answer with High Efficiency",
            "text": """<think>
Let me calculate: 25 + 4 = 29. Maybe 25 / 4 = 6.25? No.
What about 25 * 4 = 100? Too big. 
Let me try 4 * 25 = 100 again. Still too big.
</think>
<answer>25 + 4</answer>""",
            "ground_truth": {"numbers": [25, 4], "target": 21}
        }
    ]
    
    # Test with different alpha values
    alphas = [0.02, 0.05, 0.1]
    
    for alpha in alphas:
        print(f"\nTesting with alpha = {alpha}")
        print("-" * 30)
        
        for test_case in test_cases:
            print(f"\n{test_case['name']}:")
            
            text = test_case['text']
            ground_truth = test_case['ground_truth']
            
            # Calculate metrics
            original_reward = reward_fn(text, ground_truth)
            efficiency_reward = computational_efficiency_reward_fn(text, ground_truth, alpha)
            conservative_reward = computational_efficiency_reward_fn_conservative(text, ground_truth, alpha)
            
            attempts = count_computation_attempts(text)
            tokens = len(text.split())
            efficiency_ratio = attempts / tokens if tokens > 0 else 0
            
            print(f"  Original reward: {original_reward:.3f}")
            print(f"  Efficiency reward: {efficiency_reward:.3f}")
            print(f"  Conservative reward: {conservative_reward:.3f}")
            print(f"  Computation attempts: {attempts}")
            print(f"  Token count: {tokens}")
            print(f"  Efficiency ratio: {efficiency_ratio:.4f}")
            print(f"  Efficiency bonus: {efficiency_ratio * alpha:.4f}")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Key observations:")
    print("1. High efficiency cases should get bonus rewards")
    print("2. Verbose reasoning with low computation density gets smaller bonuses")
    print("3. Failed attempts (reward=0.0) should not get efficiency bonuses in conservative mode")
    print("4. Wrong answers (reward=0.1) can still get efficiency bonuses to encourage computation")
    print("5. Higher alpha values give larger efficiency bonuses")

if __name__ == "__main__":
    test_computational_efficiency()