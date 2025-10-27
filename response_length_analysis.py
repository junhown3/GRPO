#!/usr/bin/env python3
"""
Response Length Distribution Analysis
Analyzes response lengths across baseline and GRPO methods for L512 conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def get_response_length(response):
    """Calculate response length"""
    if pd.isna(response) or response is None:
        return 0
    return len(str(response))

def main():
    # File paths
    base_path = "output/eval_reports/20251025_032944"
    files = {
        "baseline_l512_failure": f"{base_path}/baseline-zero-shot-l512_failures.jsonl",
        "baseline_l512_success": f"{base_path}/baseline-zero-shot-l512_successes.jsonl",
        "grpo_l512_failure": f"{base_path}/grpo-finetuned-l512_failures.jsonl",
        "grpo_l512_success": f"{base_path}/grpo-finetuned-l512_successes.jsonl"
    }
    
    # Load data
    all_data = []
    
    for condition, file_path in files.items():
        data = load_jsonl(file_path)
        for item in data:
            method = "Baseline" if "baseline" in condition else "GRPO"
            success = "Success" if "success" in condition else "Failure"
            all_data.append({
                'method': method,
                'success': success,
                'condition': f"{method} L512 {success}",
                'response': item.get('response', ''),
                'response_length': get_response_length(item.get('response', '')),
                'reward': item.get('reward', 0),
                'target': item.get('target', 0),
                'numbers': item.get('numbers', [])
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print("=== RESPONSE LENGTH STATISTICS BY CONDITION ===\n")
    
    # Calculate statistics for each condition
    conditions = ["Baseline L512 Failure", "Baseline L512 Success", 
                 "GRPO L512 Failure", "GRPO L512 Success"]
    
    stats_summary = []
    
    for condition in conditions:
        condition_data = df[df['condition'] == condition]['response_length']
        
        if len(condition_data) > 0:
            stats_dict = {
                'Condition': condition,
                'Count': len(condition_data),
                'Mean': condition_data.mean(),
                'Median': condition_data.median(),
                'Std': condition_data.std(),
                'Min': condition_data.min(),
                'Max': condition_data.max(),
                'Q25': condition_data.quantile(0.25),
                'Q75': condition_data.quantile(0.75),
                'IQR': condition_data.quantile(0.75) - condition_data.quantile(0.25)
            }
            stats_summary.append(stats_dict)
            
            print(f"{condition}:")
            print(f"  Count: {stats_dict['Count']}")
            print(f"  Mean: {stats_dict['Mean']:.2f}")
            print(f"  Median: {stats_dict['Median']:.2f}")
            print(f"  Std Dev: {stats_dict['Std']:.2f}")
            print(f"  Min: {stats_dict['Min']}")
            print(f"  Max: {stats_dict['Max']}")
            print(f"  Q25: {stats_dict['Q25']:.2f}")
            print(f"  Q75: {stats_dict['Q75']:.2f}")
            print(f"  IQR: {stats_dict['IQR']:.2f}")
            print()
    
    # Create summary table
    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        print("Summary Statistics Table:")
        print(stats_df.round(2).to_string(index=False))
        print("\n")
    
    # Statistical tests
    print("=== STATISTICAL TESTS ===\n")
    
    # Group data for comparisons
    baseline_failure = df[df['condition'] == 'Baseline L512 Failure']['response_length']
    baseline_success = df[df['condition'] == 'Baseline L512 Success']['response_length']
    grpo_failure = df[df['condition'] == 'GRPO L512 Failure']['response_length']
    grpo_success = df[df['condition'] == 'GRPO L512 Success']['response_length']
    
    # Perform statistical tests
    comparisons = [
        ("Baseline Success vs Failure", baseline_success, baseline_failure),
        ("GRPO Success vs Failure", grpo_success, grpo_failure),
        ("Success: Baseline vs GRPO", baseline_success, grpo_success),
        ("Failure: Baseline vs GRPO", baseline_failure, grpo_failure)
    ]
    
    for name, group1, group2 in comparisons:
        if len(group1) > 0 and len(group2) > 0:
            # Mann-Whitney U test (non-parametric)
            try:
                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                print(f"{name}:")
                print(f"  Mann-Whitney U statistic: {statistic:.4f}")
                print(f"  p-value: {p_value:.6f}")
                print(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Group 1 mean: {group1.mean():.2f}")
                print(f"  Group 2 mean: {group2.mean():.2f}")
                print()
            except Exception as e:
                print(f"{name}: Error in statistical test - {e}")
                print()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Response Length Distribution Analysis (L512 Conditions)', fontsize=16, fontweight='bold')
    
    # 1. Box plot
    ax1 = axes[0, 0]
    box_data = []
    box_labels = []
    
    for condition in conditions:
        data = df[df['condition'] == condition]['response_length']
        if len(data) > 0:
            box_data.append(data.values)
            box_labels.append(condition)
    
    if box_data:
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax1.set_title('Box Plot of Response Lengths')
        ax1.set_ylabel('Response Length (characters)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # 2. Histogram with overlaid distributions
    ax2 = axes[0, 1]
    colors = ['blue', 'green', 'red', 'orange']
    alpha = 0.6
    
    for i, condition in enumerate(conditions):
        data = df[df['condition'] == condition]['response_length']
        if len(data) > 0:
            ax2.hist(data, bins=20, alpha=alpha, label=condition, color=colors[i], density=True)
    
    ax2.set_title('Histogram of Response Lengths')
    ax2.set_xlabel('Response Length (characters)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Violin plot
    ax3 = axes[1, 0]
    violin_data = []
    violin_labels = []
    for condition in conditions:
        data = df[df['condition'] == condition]['response_length']
        if len(data) > 0:
            violin_data.append(data.values)
            violin_labels.append(condition)
    
    if violin_data:
        parts = ax3.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
        ax3.set_xticks(range(len(violin_labels)))
        ax3.set_xticklabels(violin_labels, rotation=45, ha='right')
        ax3.set_title('Violin Plot of Response Lengths')
        ax3.set_ylabel('Response Length (characters)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Mean comparison with error bars
    ax4 = axes[1, 1]
    means = []
    stds = []
    labels = []
    
    for condition in conditions:
        data = df[df['condition'] == condition]['response_length']
        if len(data) > 0:
            means.append(data.mean())
            stds.append(data.std())
            labels.append(condition)
    
    if means:
        x_pos = np.arange(len(labels))
        bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, 
                       color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(means)])
        ax4.set_title('Mean Response Length by Condition')
        ax4.set_ylabel('Mean Response Length (characters)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax4.text(i, mean + std + max(means) * 0.01, f'{mean:.0f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('response_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Success rate impact on length
    print("=== SUCCESS RATE IMPACT ON RESPONSE LENGTH ===\n")
    
    method_success_analysis = []
    for method in ['Baseline', 'GRPO']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            success_data = method_data[method_data['success'] == 'Success']['response_length']
            failure_data = method_data[method_data['success'] == 'Failure']['response_length']
            
            analysis = {
                'Method': method,
                'Success_Mean': success_data.mean() if len(success_data) > 0 else None,
                'Failure_Mean': failure_data.mean() if len(failure_data) > 0 else None,
                'Success_Count': len(success_data),
                'Failure_Count': len(failure_data),
            }
            
            if analysis['Success_Mean'] is not None and analysis['Failure_Mean'] is not None:
                analysis['Mean_Difference'] = analysis['Success_Mean'] - analysis['Failure_Mean']
            else:
                analysis['Mean_Difference'] = None
                
            method_success_analysis.append(analysis)
            
            print(f"{method}:")
            print(f"  Success cases - Mean length: {analysis['Success_Mean']:.2f if analysis['Success_Mean'] else 'N/A'} (n={analysis['Success_Count']})")
            print(f"  Failure cases - Mean length: {analysis['Failure_Mean']:.2f if analysis['Failure_Mean'] else 'N/A'} (n={analysis['Failure_Count']})")
            print(f"  Difference (Success - Failure): {analysis['Mean_Difference']:.2f if analysis['Mean_Difference'] else 'N/A'}")
            print()
    
    # Create a summary comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    methods = []
    success_means = []
    failure_means = []
    
    for analysis in method_success_analysis:
        if analysis['Success_Mean'] is not None and analysis['Failure_Mean'] is not None:
            methods.append(analysis['Method'])
            success_means.append(analysis['Success_Mean'])
            failure_means.append(analysis['Failure_Mean'])
    
    if methods:
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, success_means, width, label='Success', color='lightgreen')
        bars2 = ax.bar(x + width/2, failure_means, width, label='Failure', color='lightcoral')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Mean Response Length (characters)')
        ax.set_title('Mean Response Length: Success vs Failure by Method')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('success_failure_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed statistics to CSV
    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv('response_length_statistics.csv', index=False)
        print(f"Detailed statistics saved to 'response_length_statistics.csv'")
    
    print("Analysis complete! Plots saved as 'response_length_analysis.png' and 'success_failure_comparison.png'")

if __name__ == "__main__":
    main()