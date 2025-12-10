import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict

# Import your algorithms and loader
from src.hf_femnist_loader import HFfemnistLoader, HFClient
from src.algorithms import RandomSelection, GreedySelection, DynamicProgramming
from src.simulator import FederatedLearningSimulator
from src.metrics import MetricsCalculator


def calculate_quality_metrics(qualities: List[float]) -> Dict[str, float]:
    """Calculate advanced statistics for quality values"""
    if not qualities:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'variance': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': float(np.mean(qualities)),
        'median': float(np.median(qualities)),
        'std': float(np.std(qualities)),
        'variance': float(np.var(qualities)),
        'min': float(np.min(qualities)),
        'max': float(np.max(qualities))
    }


def print_quality_metrics(algorithm_name: str, metrics: Dict[str, float]):
    """Print quality metrics in a formatted way"""
    print(f"\n   Quality Statistics for {algorithm_name}:")
    print(f"      Mean:     {metrics['mean']:.6f}")
    print(f"      Median:   {metrics['median']:.6f}")
    print(f"      Std Dev:  {metrics['std']:.6f}")
    print(f"      Variance: {metrics['variance']:.6f}")
    print(f"      Min:      {metrics['min']:.6f}")
    print(f"      Max:      {metrics['max']:.6f}")


def generate_visualizations(results: Dict, output_dir: str):
    """Generate comprehensive visualizations for the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    algorithms = list(results.keys())
    
    # Extract data for plotting
    num_selected = [results[algo]['num_selected'] for algo in algorithms]
    mean_qualities = [results[algo]['quality_metrics']['mean'] for algo in algorithms]
    total_costs = [results[algo]['total_cost'] for algo in algorithms]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Bar chart: Number of selected clients
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(algorithms, num_selected, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_ylabel('Number of Selected Clients', fontsize=11, fontweight='bold')
    ax1.set_title('Clients Selected by Algorithm', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, num_selected)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Bar chart: Mean Quality
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(algorithms, mean_qualities, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('Mean Quality', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Quality Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, mean_qualities)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Bar chart: Total Cost
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(algorithms, total_costs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax3.set_ylabel('Total Communication Cost', fontsize=11, fontweight='bold')
    ax3.set_title('Total Cost Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, total_costs)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Bar chart: Quality Standard Deviation
    quality_stds = [results[algo]['quality_metrics']['std'] for algo in algorithms]
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(algorithms, quality_stds, color=['#d62728', '#9467bd', '#8c564b'], alpha=0.8)
    ax4.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax4.set_title('Quality Standard Deviation', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars4, quality_stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Bar chart: Quality Variance
    quality_vars = [results[algo]['quality_metrics']['variance'] for algo in algorithms]
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.bar(algorithms, quality_vars, color=['#d62728', '#9467bd', '#8c564b'], alpha=0.8)
    ax5.set_ylabel('Variance', fontsize=11, fontweight='bold')
    ax5.set_title('Quality Variance', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars5, quality_vars)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Box plot: Quality Distribution
    ax6 = plt.subplot(2, 3, 6)
    quality_data = [results[algo]['quality_values'] for algo in algorithms]
    bp = ax6.boxplot(quality_data, labels=algorithms, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_ylabel('Quality Values', fontsize=11, fontweight='bold')
    ax6.set_title('Quality Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hf_femnist_comprehensive_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n Saved comprehensive visualization: {output_path}")
    
    # Create a separate summary comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Summary: Mean Quality vs Cost Efficiency
    x_pos = np.arange(len(algorithms))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos - width/2, mean_qualities, width, label='Mean Quality', 
                    color='#2ca02c', alpha=0.8)
    ax1.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Quality', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Quality by Algorithm', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms)
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, mean_qualities)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos - width/2, num_selected, width, label='Clients Selected',
                    color='#1f77b4', alpha=0.8)
    ax2_twin = ax2.twinx()
    bars3 = ax2_twin.bar(x_pos + width/2, total_costs, width, label='Total Cost',
                        color='#ff7f0e', alpha=0.8)
    ax2.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Clients', fontsize=11, fontweight='bold', color='#1f77b4')
    ax2_twin.set_ylabel('Total Cost', fontsize=11, fontweight='bold', color='#ff7f0e')
    ax2.set_title('Selection vs Cost Efficiency', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, num_selected)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f77b4')
    for i, (bar, val) in enumerate(zip(bars3, total_costs)):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#ff7f0e')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hf_femnist_summary_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved summary comparison: {output_path}")


def generate_dataset_visualizations(dataset_stats: Dict, output_dir: str):
    """Generate visualizations for the FEMNIST dataset statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Samples per client distribution (histogram)
    ax1 = plt.subplot(2, 3, 1)
    samples_data = dataset_stats['samples_per_client']
    samples_list = samples_data.get('data', [])
    if samples_list:
        bins = np.linspace(samples_data['min'], samples_data['max'], 20)
        ax1.hist(samples_list, bins=bins, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax1.axvline(samples_data['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {samples_data['mean']:.1f}")
        ax1.axvline(samples_data['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {samples_data['median']:.1f}")
    else:
        # Fallback if no data
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Samples per Client Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Samples per client box plot
    ax2 = plt.subplot(2, 3, 2)
    samples_list = samples_data.get('data', [])
    if samples_list:
        bp = ax2.boxplot([samples_list], labels=['FEMNIST'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#ff7f0e')
        bp['boxes'][0].set_alpha(0.7)
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax2.set_title('Samples per Client (Box Plot)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Class distribution (bar chart)
    ax3 = plt.subplot(2, 3, 3)
    class_dist = dataset_stats['class_distribution']
    classes = sorted(class_dist.keys())
    counts = [class_dist[c] for c in classes]
    ax3.bar(classes[:20], counts[:20], color='#2ca02c', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Class Label', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax3.set_title(f'Class Distribution (Top 20 of {len(classes)} classes)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Dataset summary statistics (text)
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    stats_text = f"""
FEMNIST Dataset Statistics

Total Clients Available: {dataset_stats['total_clients_available']:,}
Clients Analyzed: {dataset_stats['clients_analyzed']}
Total Samples: {dataset_stats['total_samples']:,}

Samples per Client:
  Mean: {samples_data['mean']:.1f}
  Median: {samples_data['median']:.1f}
  Std Dev: {samples_data['std']:.1f}
  Min: {samples_data['min']}
  Max: {samples_data['max']}
  Q25: {samples_data['q25']:.1f}
  Q75: {samples_data['q75']:.1f}

Number of Classes: {dataset_stats['num_classes']}

Labels per Client:
  Mean: {dataset_stats['labels_per_client']['mean']:.1f}
  Std Dev: {dataset_stats['labels_per_client']['std']:.1f}
  Min: {dataset_stats['labels_per_client']['min']}
  Max: {dataset_stats['labels_per_client']['max']}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Samples statistics bar chart
    ax5 = plt.subplot(2, 3, 5)
    stat_names = ['Mean', 'Median', 'Min', 'Max']
    stat_values = [
        samples_data['mean'],
        samples_data['median'],
        samples_data['min'],
        samples_data['max']
    ]
    bars = ax5.bar(stat_names, stat_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax5.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax5.set_title('Samples per Client Statistics', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, stat_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Labels per client statistics
    ax6 = plt.subplot(2, 3, 6)
    label_stats = dataset_stats['labels_per_client']
    label_stat_names = ['Mean', 'Min', 'Max']
    label_stat_values = [
        label_stats['mean'],
        label_stats['min'],
        label_stats['max']
    ]
    bars = ax6.bar(label_stat_names, label_stat_values, color=['#9467bd', '#8c564b', '#e377c2'], alpha=0.8)
    ax6.set_ylabel('Number of Unique Labels', fontsize=11, fontweight='bold')
    ax6.set_title('Unique Labels per Client', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, label_stat_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'femnist_dataset_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved dataset statistics visualization: {output_path}")


def generate_accuracy_visualizations(results: Dict, output_dir: str):
    """Generate accuracy vs rounds visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    algorithms = list(results.keys())
    
    # Create accuracy vs rounds plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy over rounds
    ax1 = axes[0]
    for algo_name in algorithms:
        if 'accuracy_history' in results[algo_name]:
            accuracy_history = results[algo_name]['accuracy_history']
            rounds = range(1, len(accuracy_history) + 1)
            ax1.plot(rounds, accuracy_history, marker='o', label=algo_name, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Training Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy vs Training Rounds', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Final accuracy comparison
    ax2 = axes[1]
    final_accuracies = []
    for algo_name in algorithms:
        if 'accuracy_history' in results[algo_name] and len(results[algo_name]['accuracy_history']) > 0:
            final_acc = results[algo_name]['accuracy_history'][-1]
            final_accuracies.append(final_acc)
        else:
            final_accuracies.append(0.0)
    
    bars = ax2.bar(algorithms, final_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_ylabel('Final Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, final_accuracies)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_vs_rounds.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved accuracy visualization: {output_path}")


print("\n" + "="*80)
print("REAL FEMNIST - CLIENT SELECTION")
print("="*80 + "\n")

# Load real data
clients = HFfemnistLoader.load_femnist(num_clients=100)

if not clients:
    print(" Failed to load")
    exit(1)

HFfemnistLoader.print_statistics(clients)

# Get and print dataset statistics
print("\n" + "="*80)
print("FEMNIST DATASET STATISTICS")
print("="*80)
dataset_stats = HFfemnistLoader.get_dataset_statistics(num_clients=100)
if dataset_stats:
    print(f"\nTotal Clients Available: {dataset_stats['total_clients_available']:,}")
    print(f"Clients Analyzed: {dataset_stats['clients_analyzed']}")
    print(f"Total Samples: {dataset_stats['total_samples']:,}")
    print(f"Number of Classes: {dataset_stats['num_classes']}")
    print(f"\nSamples per Client:")
    samples_stats = dataset_stats['samples_per_client']
    print(f"  Mean: {samples_stats['mean']:.1f}, Median: {samples_stats['median']:.1f}")
    print(f"  Std Dev: {samples_stats['std']:.1f}")
    print(f"  Range: [{samples_stats['min']}, {samples_stats['max']}]")
    print(f"  Q25: {samples_stats['q25']:.1f}, Q75: {samples_stats['q75']:.1f}")
    print(f"\nLabels per Client:")
    label_stats = dataset_stats['labels_per_client']
    print(f"  Mean: {label_stats['mean']:.1f} ± {label_stats['std']:.1f}")
    print(f"  Range: [{label_stats['min']}, {label_stats['max']}]")
print("="*80 + "\n")

# Calculate budget and fair target_k (same for all algorithms)
alpha, beta = MetricsCalculator.ALPHA, MetricsCalculator.BETA
all_costs = [alpha * c.latency + beta / c.bandwidth for c in clients]
budget = sum(all_costs) / 2.0  # Use half of total cost as budget

mean_cost = np.mean(all_costs)
target_k = max(1, min(len(clients), int(budget / mean_cost)))

print(f"Communication Budget: {budget:.4f}")
print(f"Mean cost per client: {mean_cost:.4f}")
print(f"Target clients per round (fair for all algos): {target_k}\n")

# Initialize algorithms
algorithms = {
    'Random': RandomSelection(seed=42),
    'Greedy': GreedySelection(),
    'DynamicProgramming': DynamicProgramming()
}

# Run algorithms with simulator to get accuracy over rounds
print("Running Client Selection Algorithms with Training Simulation:\n")
num_rounds = 20
results = {}

# Run each algorithm with simulator
for algo_name, algorithm in algorithms.items():
    print(f"{algo_name} Selection:")
    
    # Create simulator
    simulator = FederatedLearningSimulator(
        clients=clients,
        selection_algorithm=algorithm,
        budget=budget,
        seed=42,
        target_k=target_k
    )
    
    # Run training
    simulator.run_training(num_rounds=num_rounds)
    
    # Get selected clients from last round (for quality metrics)
    selected_clients = simulator.selected_clients_history[-1] if simulator.selected_clients_history else []
    quality_values = [c.quality for c in selected_clients]
    quality_metrics = calculate_quality_metrics(quality_values)
    
    print(f"   Selected (target/avg per round): {target_k} / {np.mean([len(s) for s in simulator.selected_clients_history]):.1f} clients")
    print(f"   Final Accuracy: {simulator.accuracy_history[-1]:.4f}")
    print(f"   Total Cost (avg per round): {np.mean(simulator.communication_costs):.4f}")
    print_quality_metrics(algo_name, quality_metrics)
    
    results[algo_name] = {
        'num_selected': target_k,
        'mean_quality': float(quality_metrics['mean']),
        'total_cost': float(np.mean(simulator.communication_costs)),
        'quality_values': quality_values,
        'quality_metrics': quality_metrics,
        'accuracy_history': simulator.accuracy_history,
        'communication_costs': simulator.communication_costs,
        'final_accuracy': float(simulator.accuracy_history[-1])
    }
    print()


# Save results
output_dir = '../results/hf_femnist'
os.makedirs(output_dir, exist_ok=True)

# Convert numpy arrays to lists for JSON serialization
results_json = {}
for algo_name, algo_results in results.items():
    results_json[algo_name] = {
        'num_selected': algo_results['num_selected'],
        'mean_quality': algo_results['mean_quality'],
        'total_cost': algo_results['total_cost'],
        'quality_metrics': algo_results['quality_metrics'],
        'quality_values': [float(q) for q in algo_results['quality_values']],
        'accuracy_history': [float(a) for a in algo_results['accuracy_history']],
        'communication_costs': [float(c) for c in algo_results['communication_costs']],
        'final_accuracy': algo_results['final_accuracy']
    }

with open(os.path.join(output_dir, 'hf_femnist_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# Generate visualizations
print("\n" + "="*80)
print("Generating Visualizations")
print("="*80)

# Generate dataset statistics visualizations
if dataset_stats:
    generate_dataset_visualizations(dataset_stats, output_dir)

# Generate accuracy visualizations
generate_accuracy_visualizations(results, output_dir)

# Generate quality metrics visualizations
generate_visualizations(results, output_dir)

print("\n" + "="*80)
print(" Visualizations generated!")
print("="*80)

# Print comprehensive summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
for algo_name, algo_results in results.items():
    print(f"\n{algo_name}:")
    print(f"  Clients Selected (avg): {algo_results['num_selected']}")
    print(f"  Total Cost (avg per round): {algo_results['total_cost']:.4f}")
    print(f"  Final Accuracy: {algo_results['final_accuracy']:.4f}")
    metrics = algo_results['quality_metrics']
    print(f"  Quality - Mean: {metrics['mean']:.6f}, Median: {metrics['median']:.6f}")
    print(f"  Quality - Std: {metrics['std']:.6f}, Variance: {metrics['variance']:.6f}")
    print(f"  Quality - Range: [{metrics['min']:.6f}, {metrics['max']:.6f}]")
print("="*80 + "\n")