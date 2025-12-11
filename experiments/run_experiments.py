"""
Main Experiment Runner
Executes federated learning simulations across all three scenarios and algorithms
Generates results, statistics, and visualizations
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import ClientSimulator, Client
from src.algorithms import RandomSelection, GreedySelection, DynamicProgramming
from src.simulator import FederatedLearningSimulator
from src.metrics import MetricsCalculator
from src.visualizations import ResultsVisualizer
from src.femnist_loader import FemnistDataLoader


def calculate_quality_metrics(qualities: List[float]) -> Dict[str, float]:
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


class ExperimentRunner:
    
    def __init__(self, num_runs: int = 10, num_rounds: int = 20, num_clients: int = 50):
        self.num_runs = num_runs
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        
        self.results = {}
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def calculate_budget(self, clients: List[Client], divisor: float = 2.0) -> float:
        all_costs = [
            MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth
            for c in clients
        ]
        budget = sum(all_costs) / divisor
        return budget
    
    def run_scenario(self, scenario_name: str, clients_generator_func) -> Dict:
        print(f"\n{'='*70}")
        print(f"Running Scenario: {scenario_name}")
        print(f"{'='*70}")
        
        # Generate clients for this scenario
        clients = clients_generator_func()
        budget = self.calculate_budget(clients, divisor=2.0)
        
        # Calculate target_k for fair comparison (same number of clients selected)
        all_costs = [MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth for c in clients]
        mean_cost = np.mean(all_costs)
        target_k = max(1, int(budget / mean_cost))  # Ensure at least 1 client
        
        print(f"Clients: {len(clients)}, Budget: {budget:.4f}, Target k: {target_k}")
        
        # Initialize algorithms
        algorithms = {
            'Random': RandomSelection(seed=42),
            'Greedy': GreedySelection(),
            'DynamicProgramming': DynamicProgramming()
        }
        
        scenario_results = {}
        
        # Run each algorithm
        for algo_name, algorithm in algorithms.items():
            print(f"\n  Running {algo_name}")
            
            # Collect results from multiple runs
            run_accuracies = []
            run_costs = []
            run_convergence = []
            run_final_accuracies = []
            run_quality_values = []  # Track quality of selected clients
            
            for run_num in range(self.num_runs):
                # Create simulator with target_k for fair comparison
                simulator = FederatedLearningSimulator(
                    clients=clients,
                    selection_algorithm=algorithm,
                    budget=budget,
                    seed=42 + run_num,
                    target_k=target_k
                )
                
                # Run training
                simulator.run_training(num_rounds=self.num_rounds)
                
                # Collect metrics
                run_accuracies.append(simulator.accuracy_history)
                run_costs.append(simulator.communication_costs)
                run_final_accuracies.append(simulator.accuracy_history[-1])
                convergence = simulator.get_convergence_speed(target_accuracy=0.9)
                run_convergence.append(convergence if convergence > 0 else self.num_rounds)
                
                # Collect quality values from selected clients (average across all rounds)
                all_selected_qualities = []
                for selected_clients in simulator.selected_clients_history:
                    all_selected_qualities.extend([c.quality for c in selected_clients])
                run_quality_values.append(all_selected_qualities)
            
            # Aggregate results across runs
            mean_accuracy = np.mean(run_accuracies, axis=0)
            std_accuracy = np.std(run_accuracies, axis=0)
            mean_cost = np.mean(run_costs, axis=0)
            std_cost = np.std(run_costs, axis=0)
            
            # Calculate quality metrics from all selected clients across all runs
            all_quality_values = [q for qualities in run_quality_values for q in qualities]
            quality_metrics = calculate_quality_metrics(all_quality_values)
            
            scenario_results[algo_name] = {
                'accuracy_history': mean_accuracy.tolist(),
                'accuracy_std': std_accuracy.tolist(),
                'communication_costs': mean_cost.tolist(),
                'communication_std': std_cost.tolist(),
                'final_accuracy': np.mean(run_final_accuracies),
                'final_accuracy_std': np.std(run_final_accuracies),
                'avg_convergence_speed': np.mean(run_convergence),
                'convergence_std': np.std(run_convergence),
                'total_communication': np.sum(mean_cost),
                'avg_clients_selected': 0,  # Will calculate below
                'quality_metrics': quality_metrics,
                'quality_values': all_quality_values[:1000] if len(all_quality_values) > 1000 else all_quality_values  # Sample for storage
            }
            
            print(f" Final Accuracy: {scenario_results[algo_name]['final_accuracy']:.4f} ± {scenario_results[algo_name]['final_accuracy_std']:.4f}")
            print(f" Quality - Mean: {quality_metrics['mean']:.4f}, Std: {quality_metrics['std']:.4f}")
            print(f" Avg Communication Cost: {np.mean(mean_cost):.4f}")
            print(f" Convergence Speed (90%): {scenario_results[algo_name]['avg_convergence_speed']:.1f} rounds")
        
        return scenario_results
    
    def run_all_scenarios(self):
        simulator_gen = ClientSimulator(num_clients=self.num_clients)
        
        scenarios = {
            'Homogeneous': simulator_gen.get_homogeneous_clients,
            'ModeratelyHeterogeneous': simulator_gen.get_moderately_heterogeneous_clients,
            'HighlyHeterogeneous': simulator_gen.get_highly_heterogeneous_clients,
        }
        
        for scenario_name, generator_func in scenarios.items():
            self.results[scenario_name] = self.run_scenario(scenario_name, generator_func)
    
    def generate_visualizations(self):
        print(f"\n{'='*70}")
        print("Generating Visualizations")
        print(f"{'='*70}")
        
        for scenario_name, algo_results in self.results.items():
            # Prepare data for plotting
            results_for_plot = {}
            for algo_name, results in algo_results.items():
                results_for_plot[algo_name] = {
                    'accuracy_history': results['accuracy_history'],
                    'communication_costs': results['communication_costs']
                }
            
            # Plot accuracy comparison
            output_path = f"results/accuracy_{scenario_name}.png"
            ResultsVisualizer.plot_accuracy_comparison(
                results_for_plot,
                output_path=output_path
            )
            
            # Plot communication cost comparison
            output_path = f"results/communication_{scenario_name}.png"
            ResultsVisualizer.plot_communication_cost_comparison(
                results_for_plot,
                output_path=output_path
            )
            
            # Plot summary comparison
            output_path = f"results/summary_{scenario_name}.png"
            ResultsVisualizer.plot_summary_comparison(
                algo_results,
                scenario=scenario_name,
                output_path=output_path
            )
    
    def generate_report(self):
        report_path = "results/experiment_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEDERATED LEARNING CLIENT SELECTION ALGORITHMS - EXPERIMENT REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Experiment Configuration:\n")
            f.write(f"  Number of clients: {self.num_clients}\n")
            f.write(f"  Number of runs per scenario: {self.num_runs}\n")
            f.write(f"  Training rounds per run: {self.num_rounds}\n")
            f.write(f"  Metrics: Alpha (latency weight) = {MetricsCalculator.ALPHA}\n")
            f.write(f"  Metrics: Beta (bandwidth weight) = {MetricsCalculator.BETA}\n\n")
            
            for scenario_name, algo_results in self.results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"SCENARIO: {scenario_name}\n")
                f.write(f"{'='*80}\n\n")
                
                for algo_name, results in algo_results.items():
                    f.write(f"\n{algo_name}:\n")
                    f.write(f"  Final Accuracy:        {results['final_accuracy']:.4f} ± {results['final_accuracy_std']:.4f}\n")
                    f.write(f"  Convergence Speed:     {results['avg_convergence_speed']:.1f} ± {results['convergence_std']:.1f} rounds\n")
                    f.write(f"  Total Communication:   {results['total_communication']:.4f}\n")
                    if 'quality_metrics' in results:
                        qm = results['quality_metrics']
                        f.write(f"  Quality Metrics:\n")
                        f.write(f"    Mean:     {qm['mean']:.6f}\n")
                        f.write(f"    Median:   {qm['median']:.6f}\n")
                        f.write(f"    Std Dev:  {qm['std']:.6f}\n")
                        f.write(f"    Variance: {qm['variance']:.6f}\n")
                        f.write(f"    Range:    [{qm['min']:.6f}, {qm['max']:.6f}]\n")
                    f.write(f"  Accuracy History:      {[f'{a:.4f}' for a in results['accuracy_history'][:5]]} ... (first 5 rounds)\n\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("SUMMARY AND INSIGHTS\n")
            f.write(f"{'='*80}\n\n")
            
            for scenario_name, algo_results in self.results.items():
                f.write(f"\n{scenario_name}:\n")
                
                # Find best algorithm for this scenario
                best_accuracy_algo = max(algo_results.items(), 
                                        key=lambda x: x[1]['final_accuracy'])
                best_efficiency_algo = min(algo_results.items(),
                                          key=lambda x: x[1]['total_communication'])
                
                f.write(f"  Best Accuracy:  {best_accuracy_algo[0]} ({best_accuracy_algo[1]['final_accuracy']:.4f})\n")
                f.write(f"  Best Efficiency:  {best_efficiency_algo[0]} ({best_efficiency_algo[1]['total_communication']:.4f} total cost)\n")
                
                # Algorithm comparisons
                random_acc = algo_results.get('Random', {}).get('final_accuracy', 0)
                greedy_acc = algo_results.get('Greedy', {}).get('final_accuracy', 0)
                dp_acc = algo_results.get('DynamicProgramming', {}).get('final_accuracy', 0)
                
                if random_acc > 0 and greedy_acc > 0:
                    greedy_improvement = ((greedy_acc - random_acc) / random_acc) * 100
                    f.write(f"  Greedy vs Random:      {greedy_improvement:+.2f}% accuracy improvement\n")
                
                if greedy_acc > 0 and dp_acc > 0:
                    dp_improvement = ((dp_acc - greedy_acc) / greedy_acc) * 100
                    approximation_ratio = greedy_acc / dp_acc if dp_acc > 0 else 0
                    f.write(f"  DP vs Greedy:          {dp_improvement:+.2f}% accuracy improvement\n")
                    f.write(f"  Greedy/DP Ratio:       {approximation_ratio:.4f}\n")
                
                # Quality metrics comparison
                if 'quality_metrics' in algo_results.get('Random', {}):
                    random_qm = algo_results['Random']['quality_metrics']
                    greedy_qm = algo_results.get('Greedy', {}).get('quality_metrics', {})
                    dp_qm = algo_results.get('DynamicProgramming', {}).get('quality_metrics', {})
                    
                    f.write(f"\n  Quality Metrics Comparison:\n")
                    f.write(f"    Random Mean Quality:      {random_qm.get('mean', 0):.6f}\n")
                    if greedy_qm:
                        f.write(f"    Greedy Mean Quality:      {greedy_qm.get('mean', 0):.6f}\n")
                        quality_improvement = ((greedy_qm.get('mean', 0) - random_qm.get('mean', 0)) / random_qm.get('mean', 1)) * 100
                        f.write(f"    Quality Improvement:      {quality_improvement:+.2f}%\n")
                    if dp_qm:
                        f.write(f"    DP Mean Quality:           {dp_qm.get('mean', 0):.6f}\n")
        
        print(f"Report saved to: {report_path}")
    
    def save_results_json(self):
        json_path = "results/results.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {json_path}")

    def run_with_real_data(self):
        print("\n" + "="*70)
        print("EXPERIMENTS WITH REAL FEDERATED LEARNING DATA (FEMNIST)")
        print("="*70)
        
        loader = FemnistDataLoader()
        real_clients = loader.load_clients(num_clients=100)
        
        # Print statistics
        stats = loader.get_statistics(real_clients)
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Run experiments with real data
        budget = self.calculate_budget(real_clients, divisor=2.0)
        
        # Run each algorithm
        algorithms = {
            'Random': RandomSelection(seed=42),
            'Greedy': GreedySelection(),
            'DynamicProgramming': DynamicProgramming()
        }
        
        real_data_results = {}
        
        for algo_name, algorithm in algorithms.items():
            
            run_accuracies = []
            run_costs = []
            
            for run_num in range(self.num_runs):
                simulator = FederatedLearningSimulator(
                    clients=real_clients,
                    selection_algorithm=algorithm,
                    budget=budget,
                    seed=42 + run_num
                )
                
                simulator.run_training(num_rounds=self.num_rounds)
                run_accuracies.append(simulator.accuracy_history)
                run_costs.append(simulator.communication_costs)
            
            mean_accuracy = np.mean(run_accuracies, axis=0)
            mean_cost = np.mean(run_costs, axis=0)
            
            real_data_results[algo_name] = {
                'accuracy_history': mean_accuracy.tolist(),
                'communication_costs': mean_cost.tolist(),
                'final_accuracy': mean_accuracy[-1],
            }
            
            print(f"  Final Accuracy: {mean_accuracy[-1]:.4f}")
            print(f"  Avg Cost per round: {np.mean(mean_cost):.4f}")
        
        return real_data_results



def main():
    print("Federated Learning Client Selection - Experiment Runner")
    print("="*70)
    
    # Create and run experiment
    runner = ExperimentRunner(num_runs=10, num_rounds=20, num_clients=50)
    
    # Run all scenarios
    print("\nStarting experiments")
    runner.run_all_scenarios()
    
    # Generate visualizations
    runner.generate_visualizations()
    
    # Generate report
    runner.generate_report()
    
    # Save results
    runner.save_results_json()
    
    # Print comprehensive quality metrics summary
    print("\n" + "="*70)
    print("COMPREHENSIVE QUALITY METRICS SUMMARY")
    print("="*70)
    
    for scenario_name, algo_results in runner.results.items():
        print(f"\n{scenario_name}:")
        print("-" * 70)
        
        for algo_name, results in algo_results.items():
            print(f"\n  {algo_name}:")
            print(f"    Final Accuracy: {results['final_accuracy']:.4f} ± {results['final_accuracy_std']:.4f}")
            if 'quality_metrics' in results:
                qm = results['quality_metrics']
                print(f"    Quality - Mean: {qm['mean']:.6f}, Median: {qm['median']:.6f}")
                print(f"    Quality - Std: {qm['std']:.6f}, Variance: {qm['variance']:.6f}")
                print(f"    Quality - Range: [{qm['min']:.6f}, {qm['max']:.6f}]")
        
        # Show improvements
        random_acc = algo_results.get('Random', {}).get('final_accuracy', 0)
        greedy_acc = algo_results.get('Greedy', {}).get('final_accuracy', 0)
        dp_acc = algo_results.get('DynamicProgramming', {}).get('final_accuracy', 0)
        
        if random_acc > 0 and greedy_acc > 0:
            improvement = ((greedy_acc - random_acc) / random_acc) * 100
            print(f"\n  Greedy vs Random: {improvement:+.2f}% accuracy improvement")
        
        if greedy_acc > 0 and dp_acc > 0:
            improvement = ((dp_acc - greedy_acc) / greedy_acc) * 100
            print(f"  DP vs Greedy:     {improvement:+.2f}% accuracy improvement")
        
        # Quality metrics improvements
        if 'quality_metrics' in algo_results.get('Random', {}):
            random_qm = algo_results['Random']['quality_metrics']
            greedy_qm = algo_results.get('Greedy', {}).get('quality_metrics', {})
            dp_qm = algo_results.get('DynamicProgramming', {}).get('quality_metrics', {})
            
            if greedy_qm:
                quality_improvement = ((greedy_qm.get('mean', 0) - random_qm.get('mean', 0)) / random_qm.get('mean', 1)) * 100
                print(f"  Quality Improvement (Greedy vs Random): {quality_improvement:+.2f}%")
            
            if dp_qm:
                quality_improvement = ((dp_qm.get('mean', 0) - greedy_qm.get('mean', 0)) / greedy_qm.get('mean', 1)) * 100
                print(f"  Quality Improvement (DP vs Greedy): {quality_improvement:+.2f}%")
    


if __name__ == "__main__":
    main()
