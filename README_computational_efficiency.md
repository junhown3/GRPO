# GRPO with Computational Efficiency Reward

This directory contains an experimental extension to GRPO that incorporates computational efficiency into the reward function.

## Hypothesis

Based on analysis of response length distributions, we hypothesize that one of the main reasons models fail at math reasoning tasks is that they spend too many tokens on verbose reasoning traces instead of attempting computational steps. 

**Key Insight**: Models that make more computation attempts per token tend to perform better.

## New Files

- `GRPO_computational_efficiency.py` - Main implementation with computational efficiency reward functions
- `run_computational_efficiency.py` - Simple runner that patches the original GRPO training
- `test_computational_efficiency.py` - Test script to validate the reward function behavior

## Computational Efficiency Reward Function

The new reward function combines:
1. **Base accuracy reward** (0.0, 0.1, or 1.0 from original `reward_fn`)
2. **Efficiency bonus**: `(computation_attempts / token_count) * alpha`

Where:
- `computation_attempts` counts mathematical operations, equation attempts, and reasoning steps
- `token_count` is the number of tokens in the response
- `alpha` is a small scaling factor (default: 0.05)

## Usage

### Option 1: Using the Runner Script (Recommended)
```bash
# Run with default alpha=0.05
python run_computational_efficiency.py

# Run with custom alpha
python run_computational_efficiency.py --alpha 0.1
```

### Option 2: Direct Testing
```bash
# Test the reward function
python test_computational_efficiency.py

# Test the full implementation
python GRPO_computational_efficiency.py --alpha 0.05
```

## Configuration

Key parameters:
- `alpha`: Efficiency scaling factor (default: 0.05)
  - Lower values (0.01-0.03): More conservative, smaller bonuses
  - Higher values (0.1-0.2): More aggressive, larger bonuses
- `conservative`: Use conservative mode that only gives bonuses for successful attempts

## Expected Results

The computational efficiency reward should:
1. **Encourage computational attempts**: Reward responses with more math operations per token
2. **Discourage verbosity**: Penalize long reasoning with few actual computations
3. **Maintain accuracy**: Preserve the base accuracy reward structure
4. **Improve success rate**: Lead to higher success rates by encouraging focused computation

## Environment Variables

Same as original GRPO, plus:
- `EXPERIMENT_TYPE`: Automatically set to identify computational efficiency runs

## Logging

TensorBoard logs are saved to:
```
./output/tb/comp_eff_grpo_alpha{alpha}/{timestamp}/
```

## Integration with Original Code

This implementation:
- ✅ Keeps `GRPO.py` completely intact
- ✅ Imports and reuses all original functions
- ✅ Provides minimal, clean modifications
- ✅ Allows easy comparison between original and modified training
- ✅ Maintains all original hyperparameters and settings

## Next Steps

1. **Push to GitHub**: Commit all new files
2. **GPU Training**: Use RunPod or similar GPU instance for training
3. **Comparison**: Run both original and computational efficiency versions
4. **Analysis**: Compare success rates and response patterns
5. **Hyperparameter Tuning**: Experiment with different alpha values

## Example Output

```
Testing with alpha = 0.05
------------------------------

High Efficiency (Multiple Operations):
  Original reward: 1.000
  Efficiency reward: 1.089
  Computation attempts: 8
  Token count: 45
  Efficiency ratio: 0.1778
  Efficiency bonus: 0.0889

Low Efficiency (Verbose Reasoning):
  Original reward: 1.000  
  Efficiency reward: 1.022
  Computation attempts: 2
  Token count: 47
  Efficiency ratio: 0.0426
  Efficiency bonus: 0.0213
```

This shows the efficiency reward properly incentivizes computational density while maintaining the base accuracy reward.