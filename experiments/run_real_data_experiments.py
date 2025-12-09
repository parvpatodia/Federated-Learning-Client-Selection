"""
Real Data Experiments
Tests algorithms on actual FEMNIST federated learning data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.femnist_loader import FemnistDataLoader
from src.algorithms import RandomSelection, GreedySelection, DynamicProgramming
from src.simulator import FederatedLearningSimulator
from src.metrics import MetricsCalculator
from src.visualizations import ResultsVisualizer
import numpy as np


def main():
    print("="*70)
    print("FEDERATED LEARNING WITH REAL DATA (FEMNIST)")
    print("="*70)
    
    # Load real data
    loader = FemnistDataLoader()
    real_clients = loader.load_clients(num_clients=100)
    
    # Print statistics
    stats = loader.get_statistics(real_clients)
    print(f"\nDataset Statistics:")
    print(f"  Clients: {stats['num_clients']}")
    print(f"  Avg data per client: {stats['data_size_mean']:.0f} samples")
    print(f"  Quality (accuracy) mean: {stats['quality_mean']:.3f}")
    print(f"  Quality (accuracy) std: {stats['quality_std']:.3f}")
    print(f"  Avg bandwidth: {stats['bandwidth_mean']:.2f} Mbps")
    print(f"  Avg latency: {stats['latency_mean']:.2f}s")
    print(f"  Heterogeneity: {stats['heterogeneity']:.3f}")
    
    # Calculate budget
    all_costs = [MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth 
                 for c in real_clients]
    budget = sum(all_costs) / 2.0
    
    print(f"\nCommunication Budget: {budget:.4f}")
    
    # Run algorithms
    algorithms = {
        'Random': RandomSelection(seed=42),
        'Greedy': GreedySelection(),
        'DynamicProgramming': DynamicProgramming()
    }
    
    results = {}
    
    for algo_name, algorithm in algorithms.items():
        print(f"\n{'='*70}")
        print(f"Algorithm: {algo_name}")
        print(f"{'='*70}")
        
        run_accuracies = []
        run_costs = []
        
        for run_num in range(10):
            simulator = FederatedLearningSimulator(
                clients=real_clients,
                selection_algorithm=algorithm,
                budget=budget,
                seed=42 + run_num
            )
            
            simulator.run_training(num_rounds=20)
            run_accuracies.append(simulator.accuracy_history)
            run_costs.append(simulator.communication_costs)
            
            print(f"  Run {run_num+1}: Final Accuracy = {simulator.accuracy_history[-1]:.4f}")
        
        mean_accuracy = np.mean(run_accuracies, axis=0)
        std_accuracy = np.std(run_accuracies, axis=0)
        mean_cost = np.mean(run_costs, axis=0)
        
        results[algo_name] = {
            'accuracy_history': mean_accuracy.tolist(),
            'accuracy_std': std_accuracy.tolist(),
            'communication_costs': mean_cost.tolist(),
            'final_accuracy': float(mean_accuracy[-1]),
        }
        
        print(f"\n  Final Accuracy: {mean_accuracy[-1]:.4f} ± {std_accuracy[-1]:.4f}")
        print(f"  Avg Communication Cost: {np.mean(mean_cost):.4f}")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}")
    
    os.makedirs('results/real_data', exist_ok=True)
    
    # Accuracy plot
    ResultsVisualizer.plot_accuracy_comparison(
        results,
        output_path='results/real_data/accuracy_real_data.png'
    )
    
    # Communication plot
    ResultsVisualizer.plot_communication_cost_comparison(
        results,
        output_path='results/real_data/communication_real_data.png'
    )
    
    # Summary plot
    ResultsVisualizer.plot_summary_comparison(
        results,
        scenario='Real FEMNIST Data',
        output_path='results/real_data/summary_real_data.png'
    )
    
    print("\n All visualizations generated")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}\n")
    
    for algo_name, algo_results in results.items():
        print(f"{algo_name}:")
        print(f"  Final Accuracy: {algo_results['final_accuracy']:.4f}")
        print(f"  Total Communication: {sum(algo_results['communication_costs']):.4f}\n")
    
    # Calculate improvements
    random_acc = results['Random']['final_accuracy']
    greedy_acc = results['Greedy']['final_accuracy']
    dp_acc = results['DynamicProgramming']['final_accuracy']
    
    greedy_improvement = ((greedy_acc - random_acc) / random_acc) * 100
    dp_improvement = ((dp_acc - random_acc) / random_acc) * 100
    
    print(f"{'='*70}")
    print("INSIGHTS")
    print(f"{'='*70}")
    print(f"Greedy vs Random: {greedy_improvement:+.2f}% accuracy improvement")
    print(f"DP vs Random:     {dp_improvement:+.2f}% accuracy improvement")
    print(f"Greedy vs DP:     {((greedy_acc - dp_acc) / dp_acc) * 100:+.2f}% accuracy difference")


if __name__ == "__main__":
    main()
