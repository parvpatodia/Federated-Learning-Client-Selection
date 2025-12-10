"""
Federated Learning Simulator
Orchestrates the simulation of federated learning with different client selection algorithms
"""

import numpy as np
from typing import List, Dict, Tuple
from .data_generator import Client
from .metrics import MetricsCalculator


class FederatedLearningSimulator:
    """Simulates federated learning training with a given client selection algorithm"""
    
    def __init__(self, clients: List[Client], selection_algorithm, budget: float, seed=42, target_k: int = None):
        """
        Initialize simulator
        
        Args:
            clients: List of available clients
            selection_algorithm: Algorithm object with select() method
            budget: Communication budget per round
            seed: Random seed for reproducibility
            target_k: Target number of clients to select per round (for fair comparison)
        """
        self.clients = clients
        self.algorithm = selection_algorithm
        self.budget = budget
        self.seed = seed
        self.target_k = target_k
        
        self.accuracy_history = []
        self.communication_costs = []
        self.selected_clients_history = []
        self.round_num = 0
    
    def run_round(self) -> Dict:
        """
        Run one round of federated learning
        
        Returns:
            Dictionary with round metrics
        """
        # Select clients using the algorithm
        try:
            selected, cost = self.algorithm.select(self.clients, self.budget, k=self.target_k)
        except TypeError:
            # Fallback for algorithms that don't support k
            selected, cost = self.algorithm.select(self.clients, self.budget)
        
        self.selected_clients_history.append(selected)
        self.communication_costs.append(cost)
        
        # Calculate accuracy improvement
        improvement = MetricsCalculator.calculate_accuracy_improvement(selected)
        
        # Add to running accuracy
        if len(self.accuracy_history) == 0:
            current_accuracy = improvement
        else:
            current_accuracy = self.accuracy_history[-1] + improvement
        
        current_accuracy = min(current_accuracy, 1.0)  # Cap at 1.0
        self.accuracy_history.append(current_accuracy)
        
        self.round_num += 1
        
        return {
            'round': self.round_num,
            'num_selected': len(selected),
            'communication_cost': cost,
            'accuracy': current_accuracy,
            'improvement': improvement
        }
    
    def run_training(self, num_rounds: int = 20) -> List[Dict]:
        """
        Run multiple rounds of training
        
        Args:
            num_rounds: Number of training rounds
            
        Returns:
            List of round metrics
        """
        results = []
        for _ in range(num_rounds):
            result = self.run_round()
            results.append(result)
        
        return results
    
    def get_convergence_speed(self, target_accuracy: float = 0.9) -> int:
        """
        Get number of rounds to reach target accuracy
        
        Args:
            target_accuracy: Target accuracy threshold
            
        Returns:
            Number of rounds (or -1 if not reached)
        """
        for round_num, accuracy in enumerate(self.accuracy_history):
            if accuracy >= target_accuracy:
                return round_num + 1
        
        return -1
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of the training run
        
        Returns:
            Dictionary with summary metrics
        """
        if len(self.accuracy_history) == 0:
            return {}
        
        return {
            'final_accuracy': self.accuracy_history[-1],
            'max_accuracy': max(self.accuracy_history),
            'avg_communication_cost': np.mean(self.communication_costs),
            'total_communication_cost': sum(self.communication_costs),
            'avg_clients_selected': np.mean([len(s) for s in self.selected_clients_history]),
            'convergence_speed_90': self.get_convergence_speed(0.9),
            'convergence_speed_80': self.get_convergence_speed(0.8),
            'num_rounds': self.round_num
        }
    
    def reset(self):
        """Reset simulator for new run"""
        self.accuracy_history = []
        self.communication_costs = []
        self.selected_clients_history = []
        self.round_num = 0


if __name__ == "__main__":
    # Test simulator
    from .data_generator import ClientSimulator
    from .algorithms import RandomSelection, GreedySelection, DynamicProgramming
    
    print("Testing FederatedLearningSimulator")
    
    simulator_gen = ClientSimulator(num_clients=50)
    clients = simulator_gen.generate_clients()
    
    # Calculate budget
    all_costs = [MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth for c in clients]
    budget = sum(all_costs) / 2.0
    
    # Test with Greedy
    greedy_algo = GreedySelection()
    simulator = FederatedLearningSimulator(clients, greedy_algo, budget)
    
    results = simulator.run_training(num_rounds=20)
    
    print(f"\nSimulation results (Greedy):")
    for result in results[-5:]:  # Print last 5 rounds
        print(f"  Round {result['round']}: Accuracy={result['accuracy']:.4f}, "
              f"Cost={result['communication_cost']:.4f}")
    
    summary = simulator.get_summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
