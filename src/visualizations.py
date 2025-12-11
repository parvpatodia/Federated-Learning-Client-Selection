"""
Visualization utilities for plotting results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class ResultsVisualizer:
    """Creates visualizations for experiment results"""
    
    @staticmethod
    def plot_accuracy_comparison(results_dict: Dict[str, Dict], output_path: str = "results/accuracy_comparison.png"):
        """
        Plot accuracy over rounds for all algorithms
        
        Args:
            results_dict: Dictionary with algorithm names as keys and their results as values
            output_path: Path to save the plot
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        for algo_name, results in results_dict.items():
            accuracies = results['accuracy_history']
            rounds = range(1, len(accuracies) + 1)
            plt.plot(rounds, accuracies, marker='o', label=algo_name, linewidth=2)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy Over Training Rounds', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
        except Exception as e:
            plt.close()
            print(f"Warning: Could not save {output_path}: {e}")
            # Try alternative path
            try:
                alt_path = os.path.join(os.getcwd(), os.path.basename(output_path))
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                print(f"Saved to alternative path: {alt_path}")
            except:
                print(f"Failed to save visualization")
    
    @staticmethod
    def plot_communication_cost_comparison(results_dict: Dict[str, Dict], output_path: str = "results/communication_cost.png"):
        """
        Plot communication cost per round for all algorithms
        
        Args:
            results_dict: Dictionary with algorithm names as keys and their results as values
            output_path: Path to save the plot
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        for algo_name, results in results_dict.items():
            costs = results['communication_costs']
            rounds = range(1, len(costs) + 1)
            plt.plot(rounds, costs, marker='s', label=algo_name, linewidth=2)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Communication Cost', fontsize=12)
        plt.title('Communication Cost Per Round', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
        except Exception as e:
            plt.close()
            print(f"Warning: Could not save {output_path}: {e}")
            # Try alternative path
            try:
                alt_path = os.path.join(os.getcwd(), os.path.basename(output_path))
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                print(f"Saved to alternative path: {alt_path}")
            except:
                print(f"Failed to save visualization")
    
    @staticmethod
    def plot_summary_comparison(summary_dict: Dict[str, Dict], scenario: str, output_path: str = "results/summary_comparison.png"):
        """
        Plot bar chart comparing key metrics across algorithms
        
        Args:
            summary_dict: Dictionary with algorithm names and their summary stats
            scenario: Name of the scenario (e.g., "Homogeneous")
            output_path: Path to save the plot
        """
        # Ensure directory exists
        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        algorithms = list(summary_dict.keys())
        final_accuracies = [summary_dict[algo]['final_accuracy'] for algo in algorithms]
        
        # Calculate total communication cost for each algorithm
        total_costs = []
        for algo in algorithms:
            if 'total_communication' in summary_dict[algo]:
                # From synthetic experiments
                total_costs.append(summary_dict[algo]['total_communication'])
            else:
                # From real data experiments
                total_costs.append(sum(summary_dict[algo]['communication_costs']))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Final Accuracy
        ax1.bar(algorithms, final_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_ylabel('Final Accuracy', fontsize=12)
        ax1.set_title(f'Final Accuracy - {scenario}', fontsize=12)
        ax1.set_ylim([0, 1])
        for i, v in enumerate(final_accuracies):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Total Communication Cost
        ax2.bar(algorithms, total_costs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_ylabel('Total Communication Cost', fontsize=12)
        ax2.set_title(f'Total Communication Cost - {scenario}', fontsize=12)
        for i, v in enumerate(total_costs):
            ax2.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
        except Exception as e:
            plt.close()
            print(f"Warning: Could not save {output_path}: {e}")
            # Try alternative path
            try:
                alt_path = os.path.join(os.getcwd(), os.path.basename(output_path))
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                print(f"Saved to alternative path: {alt_path}")
            except Exception as e2:
                print(f"Failed to save visualization: {e2}")



if __name__ == "__main__":
    print("Visualization module loaded successfully")
