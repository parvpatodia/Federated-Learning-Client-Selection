"""
Client Selection Algorithms for Federated Learning
Implements: Random, Greedy, and Dynamic Programming approaches
"""

import numpy as np
from typing import List, Tuple
from .data_generator import Client
from .metrics import MetricsCalculator


class RandomSelection:
    """Random client selection baseline"""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
    
    def select(self, clients: List[Client], budget: float, k: int = None) -> Tuple[List[Client], float]:
        """
        Select clients randomly
        
        Args:
            clients: List of available clients
            budget: Communication budget (not used for random)
            k: Number of clients to select (if None, select as many as fit budget)
            
        Returns:
            Tuple of (selected clients, total cost)
        """
        np.random.seed(self.seed)
        
        if k is None:
            # Select clients until budget exhausted (random order)
            shuffled = clients.copy()
            np.random.shuffle(shuffled)
            
            selected = []
            total_cost = 0
            
            for client in shuffled:
                cost = MetricsCalculator.ALPHA * client.latency + MetricsCalculator.BETA / client.bandwidth
                if total_cost + cost <= budget:
                    selected.append(client)
                    total_cost += cost
            
            return selected, total_cost
        else:
            # Select exactly k clients randomly
            selected = list(np.random.choice(clients, size=min(k, len(clients)), replace=False))
            total_cost = MetricsCalculator.calculate_communication_cost(selected)
            return selected, total_cost



class GreedySelection:
    """Greedy client selection algorithm - O(N log N)"""
    
    def select(self, clients: List[Client], budget: float, 
               alpha: float = MetricsCalculator.ALPHA, 
               beta: float = MetricsCalculator.BETA) -> Tuple[List[Client], float]:
        """
        Greedily select clients based on utility score
        
        Algorithm:
        1. Compute utility score for each client: u_i = quality / (alpha*latency + beta/bandwidth)
        2. Sort clients by utility in descending order - O(N log N)
        3. Select clients in order until budget exhausted - O(N)
        
        Time Complexity: O(N log N)
        Space Complexity: O(N)
        
        Args:
            clients: List of available clients
            budget: Communication budget
            alpha: Latency weight
            beta: Bandwidth weight
            
        Returns:
            Tuple of (selected clients, total cost)
        """
        # Step 1: Compute utility scores for all clients
        utilities = []
        for client in clients:
            utility = MetricsCalculator.compute_utility_score(client, alpha, beta)
            utilities.append((client, utility))
        
        # Step 2: Sort by utility in descending order - O(N log N)
        utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Greedily select clients until budget exhausted
        selected = []
        total_cost = 0
        
        for client, utility in utilities:
            cost = alpha * client.latency + beta / client.bandwidth
            
            if total_cost + cost <= budget:
                selected.append(client)
                total_cost += cost
            # Continue checking other clients even if one doesn't fit
        
        return selected, total_cost


class DynamicProgramming:
    """Dynamic Programming client selection - O(N * C) where C is discretized budget"""
    
    def select(self, clients: List[Client], budget: float,
               alpha: float = MetricsCalculator.ALPHA,
               beta: float = MetricsCalculator.BETA,
               discretization_factor: float = 100) -> Tuple[List[Client], float]:
        """
        Optimally select clients using dynamic programming (knapsack variant)
        
        Algorithm:
        1. Discretize budget into steps (to keep DP table manageable)
        2. Create DP table: f[i][c] = max accuracy using first i clients with cost <= c
        3. Fill table with recurrence: f[i][c] = max(f[i-1][c], f[i-1][c-cost_i] + quality_i)
        4. Backtrack to find which clients were selected
        
        Time Complexity: O(N * C) where C is discretized budget
        Space Complexity: O(N * C)
        
        Args:
            clients: List of available clients
            budget: Communication budget
            alpha: Latency weight
            beta: Bandwidth weight
            discretization_factor: Factor to discretize budget (larger = finer granularity, slower)
            
        Returns:
            Tuple of (selected clients, total cost)
        """
        n = len(clients)
        
        # Compute costs for all clients
        costs = []
        for client in clients:
            cost = alpha * client.latency + beta / client.bandwidth
            costs.append(cost)
        
        # Discretize budget
        discretized_budget = int(budget * discretization_factor)
        
        # Initialize DP table: f[i][c] = max accuracy
        # i = 0 to n (number of clients considered)
        # c = 0 to discretized_budget (cost)
        f = np.zeros((n + 1, discretized_budget + 1))
        
        # Fill DP table
        for i in range(1, n + 1):
            client = clients[i - 1]
            quality = client.quality
            cost_discretized = int(costs[i - 1] * discretization_factor)
            
            for c in range(discretized_budget + 1):
                # Option 1: Don't include client i
                f[i][c] = f[i - 1][c]
                
                # Option 2: Include client i (if it fits)
                if cost_discretized <= c:
                    include_value = f[i - 1][c - cost_discretized] + quality
                    f[i][c] = max(f[i][c], include_value)
        
        # Backtrack to find selected clients
        selected = []
        c = discretized_budget
        
        for i in range(n, 0, -1):
            # If this client was included in optimal solution
            if f[i][c] != f[i - 1][c]:
                selected.append(clients[i - 1])
                cost_discretized = int(costs[i - 1] * discretization_factor)
                c -= cost_discretized
        
        selected.reverse()
        
        # Calculate actual total cost
        total_cost = MetricsCalculator.calculate_communication_cost(selected, alpha, beta)
        
        return selected, total_cost


if __name__ == "__main__":
    # Test algorithms
    from .data_generator import ClientSimulator
    
    simulator = ClientSimulator(num_clients=50)
    clients = simulator.generate_clients()
    
    # Calculate budget (sum of all costs / 2 for tighter constraint)
    all_costs = [MetricsCalculator.ALPHA * c.latency + MetricsCalculator.BETA / c.bandwidth for c in clients]
    budget = sum(all_costs) / 2.0
    
    print(f"Testing algorithms with budget={budget:.4f}")
    print(f"{'='*60}\n")
    
    # Test Random
    random_algo = RandomSelection(seed=42)
    random_selected, random_cost = random_algo.select(clients, budget), 0
    random_selected = random_algo.select(clients, budget)
    print(f"Random Selection:")
    print(f"  Selected {len(random_selected)} clients")
    
    # Test Greedy
    greedy_algo = GreedySelection()
    greedy_selected, greedy_cost = greedy_algo.select(clients, budget)
    print(f"\nGreedy Selection:")
    print(f"  Selected {len(greedy_selected)} clients")
    print(f"  Cost: {greedy_cost:.4f} / Budget: {budget:.4f}")
    
    # Test DP
    dp_algo = DynamicProgramming()
    dp_selected, dp_cost = dp_algo.select(clients, budget)
    print(f"\nDynamic Programming:")
    print(f"  Selected {len(dp_selected)} clients")
    print(f"  Cost: {dp_cost:.4f} / Budget: {budget:.4f}")
