#!/usr/bin/env python3
"""
Compare response length distributions between L256 and L512 models.
Generate normalized histogram and calculate statistics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse

def load_responses(jsonl_file):
    """Load responses from JSONL file and extract response lengths."""
    responses = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            response_text = data.get('response', '')
            # Count tokens/words as a proxy for length
            response_length = len(response_text.split())
            responses.append(response_length)
    return responses

def calculate_statistics(lengths):
    """Calculate mean, median, and other statistics."""
    lengths_array = np.array(lengths)
    return {
        'mean': np.mean(lengths_array),
        'median': np.median(lengths_array),
        'std': np.std(lengths_array),
        'min': np.min(lengths_array),
        'max': np.max(lengths_array),
        'count': len(lengths_array)
    }

def main():
    # File paths
    l256_file = "output/eval_reports/20251027_061403/comp-eff-256_responses.jsonl"
    l512_file = "output/eval_reports/20251027_061403/comp-eff-512_responses.jsonl"
    
    # Load response lengths
    print("Loading response data...")
    l256_lengths = load_responses(l256_file)
    l512_lengths = load_responses(l512_file)
    
    # Calculate statistics
    l256_stats = calculate_statistics(l256_lengths)
    l512_stats = calculate_statistics(l512_lengths)
    
    print("\n" + "="*60)
    print("RESPONSE LENGTH STATISTICS")
    print("="*60)
    
    print(f"\nL256 Model Statistics:")
    print(f"  Count: {l256_stats['count']:,}")
    print(f"  Mean:  {l256_stats['mean']:.2f} words")
    print(f"  Median: {l256_stats['median']:.2f} words")
    print(f"  Std:   {l256_stats['std']:.2f} words")
    print(f"  Range: {l256_stats['min']} - {l256_stats['max']} words")
    
    print(f"\nL512 Model Statistics:")
    print(f"  Count: {l512_stats['count']:,}")
    print(f"  Mean:  {l512_stats['mean']:.2f} words")
    print(f"  Median: {l512_stats['median']:.2f} words")
    print(f"  Std:   {l512_stats['std']:.2f} words")
    print(f"  Range: {l512_stats['min']} - {l512_stats['max']} words")
    
    print(f"\nComparison:")
    print(f"  Mean difference: {l512_stats['mean'] - l256_stats['mean']:.2f} words")
    print(f"  Median difference: {l512_stats['median'] - l256_stats['median']:.2f} words")
    
    # Create normalized histogram
    plt.figure(figsize=(12, 8))
    
    # Define bins for histogram
    max_length = max(max(l256_lengths), max(l512_lengths))
    bins = np.linspace(0, max_length, 50)
    
    # Create normalized histograms
    plt.hist(l256_lengths, bins=bins, alpha=0.7, label='L256', 
             density=True, color='skyblue', edgecolor='black', linewidth=0.5)
    plt.hist(l512_lengths, bins=bins, alpha=0.7, label='L512', 
             density=True, color='lightcoral', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    plt.axvline(l256_stats['mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'L256 Mean: {l256_stats["mean"]:.1f}')
    plt.axvline(l512_stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f'L512 Mean: {l512_stats["mean"]:.1f}')
    
    # Add vertical lines for medians
    plt.axvline(l256_stats['median'], color='blue', linestyle=':', linewidth=2, 
                label=f'L256 Median: {l256_stats["median"]:.1f}')
    plt.axvline(l512_stats['median'], color='red', linestyle=':', linewidth=2, 
                label=f'L512 Median: {l512_stats["median"]:.1f}')
    
    plt.xlabel('Response Length (words)', fontsize=12)
    plt.ylabel('Normalized Frequency (Density)', fontsize=12)
    plt.title('Response Length Distribution: L256 vs L512\n(Computational Efficiency GRPO Model)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with key statistics
    stats_text = f"""
    L256: n={l256_stats['count']:,}, μ={l256_stats['mean']:.1f}, M={l256_stats['median']:.1f}
    L512: n={l512_stats['count']:,}, μ={l512_stats['mean']:.1f}, M={l512_stats['median']:.1f}
    """
    plt.text(0.02, 0.98, stats_text.strip(), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "output/response_length_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Generate detailed bins analysis
    print("\n" + "="*60)
    print("DETAILED DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create bins for analysis
    bin_edges = [0, 20, 40, 60, 80, 100, 150, 200, max_length]
    bin_labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-150', '151-200', '200+']
    
    def count_in_bins(lengths, bin_edges):
        counts = []
        for i in range(len(bin_edges)-1):
            count = sum(1 for length in lengths if bin_edges[i] <= length < bin_edges[i+1])
            counts.append(count)
        return counts
    
    l256_bin_counts = count_in_bins(l256_lengths, bin_edges)
    l512_bin_counts = count_in_bins(l512_lengths, bin_edges)
    
    print(f"\n{'Length Range':<12} {'L256 Count':<12} {'L256 %':<10} {'L512 Count':<12} {'L512 %':<10}")
    print("-" * 60)
    
    for i, label in enumerate(bin_labels):
        l256_pct = (l256_bin_counts[i] / l256_stats['count']) * 100
        l512_pct = (l512_bin_counts[i] / l512_stats['count']) * 100
        print(f"{label:<12} {l256_bin_counts[i]:<12} {l256_pct:<9.1f}% {l512_bin_counts[i]:<12} {l512_pct:<9.1f}%")

if __name__ == "__main__":
    main()