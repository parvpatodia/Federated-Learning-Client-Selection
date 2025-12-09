"""
Metrics Calculator for Federated Learning Evaluation
"""

import numpy as np
from typing import List
from .data_generator import Client


class MetricsCalculator:
    """Calculates various performance metrics"""
    
    # Weighting parameters
    ALPHA = 1.0  # Latency weight
    BETA = 1.0   # Bandwidth weight (inverse)
    
    @staticmethod
    def compute_utility_score(client: Client, alpha: float = ALPHA, beta: float = BETA) -> float:
        """
        Compute utility score for a client
        
        Formula: utility = quality / (alpha * latency + beta / bandwidth)
        
        Higher quality and bandwidth, lower latency -> higher utility
        
        Args:
            client: Client object
            alpha: Latency weight
            beta: Bandwidth weight
            
        Returns:
            Utility score (float)
        """
        denominator = alpha * client.latency + beta / client.bandwidth
        if denominator == 0:
            return 0
        
        utility = client.quality / denominator
        return utility
    
    @staticmethod
    def calculate_communication_cost(selected_clients: List[Client], alpha: float = ALPHA, beta: float = BETA) -> float:
        """
        Calculate total communication cost for selected clients in one round
        
        Cost = sum(alpha * latency + beta / bandwidth) for each selected client
        
        Args:
            selected_clients: List of selected Client objects
            alpha: Latency weight
            beta: Bandwidth weight
            
        Returns:
            Total communication cost
        """
        total_cost = 0
        for client in selected_clients:
            cost = alpha * client.latency + beta / client.bandwidth
            total_cost += cost
        
        return total_cost
    
    @staticmethod
    def calculate_accuracy_improvement(selected_clients: List[Client]) -> float:
        """
        Calculate accuracy improvement from selected clients
        
        Simplified model: improvement = mean(quality) * scaling_factor
        
        Args:
            selected_clients: List of selected Client objects
            
        Returns:
            Accuracy improvement (float between 0 and 1)
        """
        if len(selected_clients) == 0:
            return 0
        
        mean_quality = np.mean([c.quality for c in selected_clients])
        
        # Scaling factor: improvement is 5% of mean quality per round
        improvement = mean_quality * 0.05
        
        return improvement
    
    @staticmethod
    def calculate_total_accuracy(selected_clients_per_round: List[List[Client]], num_rounds: int) -> float:
        """
        Calculate cumulative accuracy over multiple rounds
        
        Args:
            selected_clients_per_round: List of selected clients for each round
            num_rounds: Number of rounds
            
        Returns:
            Final accuracy (capped at 1.0)
        """
        accuracy = 0.0
        
        for selected in selected_clients_per_round:
            improvement = MetricsCalculator.calculate_accuracy_improvement(selected)
            accuracy += improvement
        
        # Cap accuracy at 1.0
        return min(accuracy, 1.0)
    
    @staticmethod
    def calculate_convergence_speed(selected_clients_per_round: List[List[Client]], target_accuracy: float = 0.9) -> int:
        """
        Calculate number of rounds to reach target accuracy
        
        Args:
            selected_clients_per_round: List of selected clients for each round
            target_accuracy: Target accuracy threshold
            
        Returns:
            Number of rounds to reach target (or -1 if not reached)
        """
        accuracy = 0.0
        
        for round_num, selected in enumerate(selected_clients_per_round):
            improvement = MetricsCalculator.calculate_accuracy_improvement(selected)
            accuracy += improvement
            
            if accuracy >= target_accuracy:
                return round_num + 1
        
        return -1  # Target not reached


if __name__ == "__main__":
    # Test metrics
    from .data_generator import ClientSimulator
    
    simulator = ClientSimulator(num_clients=50)
    clients = simulator.generate_clients()
    
    print("Testing MetricsCalculator...")
    
    # Test utility score
    print(f"\nUtility scores (first 5 clients):")
    for client in clients[:5]:
        utility = MetricsCalculator.compute_utility_score(client)
        print(f"  Client {client.client_id}: {utility:.4f}")
    
    # Test communication cost
    selected = clients[:10]
    cost = MetricsCalculator.calculate_communication_cost(selected)
    print(f"\nCommunication cost for 10 clients: {cost:.4f}")
    
    # Test accuracy improvement
    improvement = MetricsCalculator.calculate_accuracy_improvement(selected)
    print(f"Accuracy improvement: {improvement:.4f}")
