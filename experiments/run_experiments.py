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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import ClientSimulator, Client
from src.algorithms import RandomSelection, GreedySelection, DynamicProgramming
from src.simulator import FederatedLearningSimulator
from src.metrics import MetricsCalculator
from src.visualizations import ResultsVisualizer


class ExperimentRunner:
    """Orchestrates all experiments across scenarios and algorithms"""
    
    def __init__(self, num_runs: int = 10, num_rounds: int = 20, num_clients: int = 50):
        """
        Initialize experiment runner
        
        Args:
            num_runs: Number of independent runs per algorithm-scenario pair
            num_rounds: Number of training rounds per run
            num_clients: Number of clients to simulate
        """
        self.num_runs = num_runs
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        
        self.results = {}
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def calculate_budget(self, clients: List[Client], divisor: float = 2.0) -> float:
        """
        Calculate communication budget for clients
        
        Budget = sum of all communication costs / divisor
        This creates a realistic constraint
        
        Args:
            clients: List of clients
            divisor: Division factor (higher = tighter budget)
            
        Returns:
            Communication budget
        """
        all_costs = [
            MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth
            for c in clients
        ]
        budget = sum(all_costs) / divisor
        return budget
    
    def run_scenario(self, scenario_name: str, clients_generator_func) -> Dict:
        """
        Run a complete scenario (homogeneous, moderate, or highly heterogeneous)
        
        Args:
            scenario_name: Name of scenario
            clients_generator_func: Function that returns list of clients
            
        Returns:
            Dictionary with results for all algorithms
        """
        print(f"\n{'='*70}")
        print(f"Running Scenario: {scenario_name}")
        print(f"{'='*70}")
        
        # Generate clients for this scenario
        clients = clients_generator_func()
        budget = self.calculate_budget(clients, divisor=2.0)
        
        print(f"Clients: {len(clients)}, Budget: {budget:.4f}")
        
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
            
            for run_num in range(self.num_runs):
                # Create simulator
                simulator = FederatedLearningSimulator(
                    clients=clients,
                    selection_algorithm=algorithm,
                    budget=budget,
                    seed=42 + run_num
                )
                
                # Run training
                simulator.run_training(num_rounds=self.num_rounds)
                
                # Collect metrics
                run_accuracies.append(simulator.accuracy_history)
                run_costs.append(simulator.communication_costs)
                run_final_accuracies.append(simulator.accuracy_history[-1])
                convergence = simulator.get_convergence_speed(target_accuracy=0.9)
                run_convergence.append(convergence if convergence > 0 else self.num_rounds)
            
            # Aggregate results across runs
            mean_accuracy = np.mean(run_accuracies, axis=0)
            std_accuracy = np.std(run_accuracies, axis=0)
            mean_cost = np.mean(run_costs, axis=0)
            std_cost = np.std(run_costs, axis=0)
            
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
                'avg_clients_selected': 0  # Will calculate below
            }
            
            print(f"    Final Accuracy: {scenario_results[algo_name]['final_accuracy']:.4f} ± {scenario_results[algo_name]['final_accuracy_std']:.4f}")
            print(f"    Avg Communication Cost: {np.mean(mean_cost):.4f}")
            print(f"    Convergence Speed (90%): {scenario_results[algo_name]['avg_convergence_speed']:.1f} rounds")
        
        return scenario_results
    
    def run_all_scenarios(self):
        """Run all three scenarios"""
        simulator_gen = ClientSimulator(num_clients=self.num_clients)
        
        scenarios = {
            'Homogeneous': simulator_gen.get_homogeneous_clients,
            'ModeratelyHeterogeneous': simulator_gen.get_moderately_heterogeneous_clients,
            'HighlyHeterogeneous': simulator_gen.get_highly_heterogeneous_clients,
        }
        
        for scenario_name, generator_func in scenarios.items():
            self.results[scenario_name] = self.run_scenario(scenario_name, generator_func)
    
    def generate_visualizations(self):
        """Generate all visualization plots"""
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
        """Generate a text report of results"""
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
                
                f.write(f"  Best Accuracy:         {best_accuracy_algo[0]} ({best_accuracy_algo[1]['final_accuracy']:.4f})\n")
                f.write(f"  Best Efficiency:       {best_efficiency_algo[0]} ({best_efficiency_algo[1]['total_communication']:.4f} total cost)\n")
                
                # Greedy vs DP comparison
                greedy_acc = algo_results.get('Greedy', {}).get('final_accuracy', 0)
                dp_acc = algo_results.get('DynamicProgramming', {}).get('final_accuracy', 0)
                
                if greedy_acc > 0 and dp_acc > 0:
                    approximation_ratio = greedy_acc / dp_acc if dp_acc > 0 else 0
                    f.write(f"  Greedy/DP Approximation Ratio: {approximation_ratio:.4f}\n")
        
        print(f"Report saved to: {report_path}")
    
    def save_results_json(self):
        """Save results as JSON for analysis"""
        json_path = "results/results.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {json_path}")


def main():
    """Main entry point"""
    print("Federated Learning Client Selection - Experiment Runner")
    print("="*70)
    
    # Create and run experiment
    runner = ExperimentRunner(num_runs=10, num_rounds=20, num_clients=50)
    
    # Run all scenarios
    print("\nStarting experiments...")
    runner.run_all_scenarios()
    
    # Generate visualizations
    runner.generate_visualizations()
    
    # Generate report
    runner.generate_report()
    
    # Save results
    runner.save_results_json()
    
    print("\n" + "="*70)
    print("Experiments completed!")
    print("Results saved to 'results/' directory")
    print("="*70)


if __name__ == "__main__":
    main()
