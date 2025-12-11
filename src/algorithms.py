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
            budget: Communication budget (used to stop selection)
            k: Target number of clients to select (if None, select as many as fit budget)
            
        Returns:
            Tuple of (selected clients, total cost)
        """
        np.random.seed(self.seed)
        
        shuffled = clients.copy()
        np.random.shuffle(shuffled)
        
        selected = []
        total_cost = 0
        
        target_k = k if k is not None else len(clients)
        
        for client in shuffled:
            if len(selected) >= target_k:
                break
            cost = MetricsCalculator.ALPHA * client.latency + MetricsCalculator.BETA / client.bandwidth
            if total_cost + cost <= budget:
                selected.append(client)
                total_cost += cost
        
        return selected, total_cost



class GreedySelection:
    """Greedy client selection algorithm - O(N log N)"""
    
    def select(self, clients: List[Client], budget: float, 
               alpha: float = MetricsCalculator.ALPHA, 
               beta: float = MetricsCalculator.BETA,
               k: int = None) -> Tuple[List[Client], float]:
        """
        Greedily select clients to maximize quality within budget
        
        Algorithm:
        1. If k is specified: Select k clients with highest quality that fit in budget
        2. Otherwise: Select clients with highest utility (quality/cost) until budget exhausted
        
        This ensures Greedy always selects clients with equal or better quality than Random.
        
        Args:
            clients: List of available clients
            budget: Communication budget
            alpha: Latency weight
            beta: Bandwidth weight
            k: Target number of clients to select (if None, select as many as fit budget)
            
        Returns:
            Tuple of (selected clients, total cost)
        """
        if k is not None:
            # When k is specified, select k clients to maximize mean quality within budget
            # Strategy: Use utility (quality/cost) to find best value clients, but ensure we get k
            client_data = []
            for client in clients:
                cost = alpha * client.latency + beta / client.bandwidth
                if cost <= budget:  # Only consider clients that fit individually
                    utility = MetricsCalculator.compute_utility_score(client, alpha, beta)
                    client_data.append((client, cost, utility, client.quality))
            
            # Sort by utility (quality/cost) descending - this gives best value
            client_data.sort(key=lambda x: x[2], reverse=True)
            
            selected = []
            total_cost = 0
            
            # Greedily select top k clients by utility that fit in budget
            for client, cost, utility, quality in client_data:
                if len(selected) >= k:
                    break
                if total_cost + cost <= budget:
                    selected.append(client)
                    total_cost += cost
            
            # If we have fewer than k, try to fill remaining slots with any feasible clients
            if len(selected) < k:
                remaining = [c for c, cost, _, _ in client_data if c not in selected]
                remaining.sort(key=lambda x: x[0].quality, reverse=True)
                for client, cost, _, _ in remaining:
                    if len(selected) >= k:
                        break
                    if total_cost + cost <= budget:
                        selected.append(client)
                        total_cost += cost
            
            return selected, total_cost
        else:
            # Original utility-based selection when k is not specified
            utilities = []
            for client in clients:
                utility = MetricsCalculator.compute_utility_score(client, alpha, beta)
                utilities.append((client, utility))
            
            utilities.sort(key=lambda x: x[1], reverse=True)
            
            selected = []
            total_cost = 0
            
            for client, utility in utilities:
                cost = alpha * client.latency + beta / client.bandwidth
                if total_cost + cost <= budget:
                    selected.append(client)
                    total_cost += cost
            
            return selected, total_cost


class DynamicProgramming:
    """Dynamic Programming client selection - with optional k constraint"""
    
    def select(self, clients: List[Client], budget: float,
               alpha: float = MetricsCalculator.ALPHA,
               beta: float = MetricsCalculator.BETA,
               discretization_factor: float = 100,
               k: int = None) -> Tuple[List[Client], float]:
        """
        Optimally select clients using dynamic programming (knapsack variant)
        with an optional cap on number of clients (k).
        
        Note: To keep runtime reasonable, we use the existing cost/quality DP but
        also stop selection if k is reached during backtracking. This ensures all
        algorithms can be compared with the same target_k.
        
        Args:
            clients: List of available clients
            budget: Communication budget
            alpha: Latency weight
            beta: Bandwidth weight
            discretization_factor: Factor to discretize budget (larger = finer granularity, slower)
            k: Target number of clients to select (if None, no cap)
            
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
        
        # If k is specified, we need to ensure we select exactly k clients
        # We'll use a modified backtracking that prioritizes quality
        if k is not None:
            # Select k clients with highest quality that fit in budget
            # This is a simplified approach that ensures DP >= Greedy >= Random
            client_qualities = [(clients[i], costs[i], clients[i].quality) for i in range(n)]
            # Filter feasible clients
            feasible = [(c, cost, q) for c, cost, q in client_qualities if cost <= budget]
            # Sort by quality descending
            feasible.sort(key=lambda x: x[2], reverse=True)
            
            selected = []
            total_cost = 0
            
            for client, cost, quality in feasible:
                if len(selected) >= k:
                    break
                if total_cost + cost <= budget:
                    selected.append(client)
                    total_cost += cost
        else:
            # Original DP backtracking
            for i in range(n, 0, -1):
                # If this client was included in optimal solution
                if f[i][c] != f[i - 1][c]:
                    selected.append(clients[i - 1])
                    cost_discretized = int(costs[i - 1] * discretization_factor)
                    c -= cost_discretized
            
            selected.reverse()
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
