#!/usr/bin/env python3
"""
Simple runner script for GRPO with computational efficiency reward.

This script temporarily replaces the reward function in GRPO.py and then
calls the original main() function, ensuring minimal code duplication.
"""

import os
import sys
from GRPO_computational_efficiency import computational_efficiency_reward_fn
from GRPO import main, reward_fn as original_reward_fn

def run_computational_efficiency_training(alpha: float = 0.05):
    """
    Run GRPO training with computational efficiency reward by patching the original main().
    
    Args:
        alpha: Efficiency scaling factor
    """
    
    print("=" * 60)
    print("GRPO WITH COMPUTATIONAL EFFICIENCY REWARD")
    print("=" * 60)
    print(f"Efficiency weight (alpha): {alpha}")
    print(f"Patching reward function in original GRPO training...")
    print()
    
    # Import the module to patch
    import GRPO
    
    # Create wrapper with alpha
    def efficiency_reward_with_alpha(generated_text: str, ground_truth: dict) -> float:
        return computational_efficiency_reward_fn(generated_text, ground_truth, alpha)
    
    # Store original reward function  
    original_fn = GRPO.reward_fn
    
    # Set environment variable to identify this run
    os.environ["EXPERIMENT_TYPE"] = f"computational_efficiency_alpha_{alpha}"
    
    try:
        # Patch the reward function
        GRPO.reward_fn = efficiency_reward_with_alpha
        
        print("Reward function patched successfully!")
        print("Starting original GRPO main() with computational efficiency reward...")
        print()
        
        # Call original main function
        main()
        
    finally:
        # Restore original reward function
        GRPO.reward_fn = original_fn
        print("Original reward function restored.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO with Computational Efficiency Reward")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Efficiency weight factor (default: 0.05)")
    
    args = parser.parse_args()
    
    run_computational_efficiency_training(alpha=args.alpha)