#!/usr/bin/env python3
"""
Script to generate histograms for response length distributions across different conditions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_jsonl(filepath):
    """Load JSONL file and return list of dictionaries"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {filepath}: {e}")
        return []
    return data

def get_response_length(response):
    """Calculate the length of a response string"""
    if response is None:
        return 0
    return len(str(response))

def main():
    # Define the base directory
    base_dir = "/Users/junhowon/Documents/GitHub/GRPO/output/eval_reports/20251025_032944"
    
    # Define the files to analyze (L512 only)
    files_to_analyze = {
        "Baseline L512 Failure": "baseline-zero-shot-l512_failures.jsonl",
        "Baseline L512 Success": "baseline-zero-shot-l512_successes.jsonl",
        "GRPO L512 Failure": "grpo-finetuned-l512_failures.jsonl", 
        "GRPO L512 Success": "grpo-finetuned-l512_successes.jsonl"
    }
    
    # Load data for each condition
    all_data = {}
    
    for condition, filename in files_to_analyze.items():
        filepath = os.path.join(base_dir, filename)
        print(f"Loading {condition} from {filename}...")
        
        data = load_jsonl(filepath)
        if data:
            # Calculate response lengths
            response_lengths = [get_response_length(item.get('response', '')) for item in data]
            all_data[condition] = response_lengths
            print(f"  Loaded {len(response_lengths)} samples")
            print(f"  Mean length: {np.mean(response_lengths):.1f} characters")
            print(f"  Min: {min(response_lengths)}, Max: {max(response_lengths)}")
        else:
            print(f"  No data loaded for {condition}")
        print()
    
    if not all_data:
        print("No data loaded. Exiting.")
        return
    
        # Create a single plot with all four histograms using density normalization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Colors for each condition
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    # Plot all conditions on the same plot with density normalization
    for i, (condition, lengths) in enumerate(all_data.items()):
        ax.hist(lengths, bins=30, alpha=0.7, label=f'{condition} (n={len(lengths)})', 
               color=colors[i], edgecolor='black', linewidth=0.5, density=True)
    
    ax.set_title('Response Length Distributions - All Conditions (L512)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Response Length (characters)')
    ax.set_ylabel('Density (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_lines = []
    for condition, lengths in all_data.items():
        stats_lines.append(f"{condition}: Mean={np.mean(lengths):.0f}, Median={np.median(lengths):.0f}")
    
    stats_text = '\n'.join(stats_lines)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(base_dir, 'response_length_combined_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined histogram saved to: {output_path}")
    
    plt.show()
    
    # Create an additional subplot version for better comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Response Length Distributions - Individual Plots (L512)', fontsize=16, fontweight='bold')
    
    axes2_flat = axes2.flatten()
    
    for i, (condition, lengths) in enumerate(all_data.items()):
        if i >= len(axes2_flat):
            break
            
        ax = axes2_flat[i]
        
        # Create histogram with proper scaling
        n, bins, patches = ax.hist(lengths, bins=30, alpha=0.8, color=colors[i], 
                                 edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_title(f'{condition}\n(n={len(lengths)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Response Length (characters)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (f'Mean: {np.mean(lengths):.0f}\n'
                     f'Median: {np.median(lengths):.0f}\n'
                     f'Std: {np.std(lengths):.0f}')
        ax.text(0.75, 0.75, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        # Set consistent y-axis scale if needed
        ax.set_ylim(0, max(n) * 1.1)
    
    # Remove any unused subplots
    for j in range(len(all_data), len(axes2_flat)):
        fig2.delaxes(axes2_flat[j])
    
    plt.tight_layout()
    
    # Save the subplot version
    subplot_path = os.path.join(base_dir, 'response_length_subplots.png')
    plt.savefig(subplot_path, dpi=300, bbox_inches='tight')
    print(f"Individual subplot histograms saved to: {subplot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for condition, lengths in all_data.items():
        print(f"\n{condition}:")
        print(f"  Count: {len(lengths)}")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std Dev: {np.std(lengths):.1f}")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  25th percentile: {np.percentile(lengths, 25):.1f}")
        print(f"  75th percentile: {np.percentile(lengths, 75):.1f}")

if __name__ == "__main__":
    main()