"""
Data Generator for Federated Learning Simulation
Generates synthetic clients with heterogeneous properties
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Client:
    """Represents a single client in the federated learning system"""
    client_id: int
    data_size: int          # [500, 2000] samples
    bandwidth: float        # [1, 10] Mbps
    quality: float          # [0.6, 0.95] accuracy contribution
    latency: float          # [0.1, 2.0] seconds


class ClientSimulator:
    """Generates and manages synthetic client population"""
    
    def __init__(self, num_clients=50, seed=42):
        """
        Initialize client simulator
        
        Args:
            num_clients: Number of clients to generate
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)
        self.clients = []
    
    def generate_clients(self) -> List[Client]:
        """
        Generate synthetic clients with specified distributions
        
        Returns:
            List of Client objects
        """
        self.clients = []
        
        for client_id in range(self.num_clients):
            # Generate parameters from specified ranges
            data_size = np.random.randint(500, 2001)  # [500, 2000]
            bandwidth = np.random.uniform(1, 10)      # [1, 10] Mbps
            quality = np.random.uniform(0.6, 0.95)    # [0.6, 0.95]
            latency = np.random.uniform(0.1, 2.0)     # [0.1, 2.0] seconds
            
            client = Client(
                client_id=client_id,
                data_size=data_size,
                bandwidth=bandwidth,
                quality=quality,
                latency=latency
            )
            self.clients.append(client)
        
        return self.clients
    
    def get_homogeneous_clients(self) -> List[Client]:
        """
        Generate homogeneous clients (all similar properties)
        Used as sanity check
        
        Returns:
            List of homogeneous Client objects
        """
        clients = []
        
        for client_id in range(self.num_clients):
            # All clients have similar properties
            client = Client(
                client_id=client_id,
                data_size=1250,           # Middle of range
                bandwidth=5.5,             # Middle of range
                quality=0.77,              # Middle of range
                latency=1.0                # Middle of range
            )
            clients.append(client)
        
        return clients
    
    def get_moderately_heterogeneous_clients(self) -> List[Client]:
        """
        Generate moderately heterogeneous clients
        Realistic differences in properties
        
        Returns:
            List of moderately heterogeneous Client objects
        """
        np.random.seed(self.seed)
        clients = []
        
        for client_id in range(self.num_clients):
            data_size = np.random.randint(500, 2001)
            bandwidth = np.random.uniform(1, 10)
            quality = np.random.uniform(0.6, 0.95)
            latency = np.random.uniform(0.1, 2.0)
            
            client = Client(
                client_id=client_id,
                data_size=data_size,
                bandwidth=bandwidth,
                quality=quality,
                latency=latency
            )
            clients.append(client)
        
        return clients
    
    def get_highly_heterogeneous_clients(self) -> List[Client]:
        """
        Generate highly heterogeneous clients with extreme differences
        Creates a very mixed distribution: elite clients, good clients, average, poor, and very poor
        This maximizes the advantage of smart selection algorithms
        
        Returns:
            List of highly heterogeneous Client objects
        """
        np.random.seed(self.seed)
        clients = []
        
        # Shuffle to mix up the distribution
        client_indices = list(range(self.num_clients))
        np.random.shuffle(client_indices)
        
        for idx, client_id in enumerate(client_indices):
            ratio = idx / self.num_clients
            
            if ratio < 0.15:  # 15% - Elite clients (very high quality, low latency, high bandwidth)
                data_size = np.random.randint(1800, 2001)
                bandwidth = np.random.uniform(9, 10)
                quality = np.random.uniform(0.90, 0.95)
                latency = np.random.uniform(0.1, 0.3)
            elif ratio < 0.30:  # 15% - Good clients
                data_size = np.random.randint(1500, 1800)
                bandwidth = np.random.uniform(7, 9)
                quality = np.random.uniform(0.85, 0.90)
                latency = np.random.uniform(0.3, 0.6)
            elif ratio < 0.50:  # 20% - Average clients
                data_size = np.random.randint(1000, 1500)
                bandwidth = np.random.uniform(4, 7)
                quality = np.random.uniform(0.75, 0.85)
                latency = np.random.uniform(0.6, 1.2)
            elif ratio < 0.75:  # 25% - Poor clients
                data_size = np.random.randint(600, 1000)
                bandwidth = np.random.uniform(2, 4)
                quality = np.random.uniform(0.65, 0.75)
                latency = np.random.uniform(1.2, 1.8)
            else:  # 25% - Very poor clients (very low quality, high latency, low bandwidth)
                data_size = np.random.randint(500, 600)
                bandwidth = np.random.uniform(1, 2)
                quality = np.random.uniform(0.60, 0.65)
                latency = np.random.uniform(1.8, 2.0)
            
            client = Client(
                client_id=client_id,
                data_size=data_size,
                bandwidth=bandwidth,
                quality=quality,
                latency=latency
            )
            clients.append(client)
        
        return clients
    
    def print_client_stats(self, clients: List[Client]):
        """Print statistics of clients for debugging"""
        print(f"\n{'='*60}")
        print(f"Client Statistics ({len(clients)} clients)")
        print(f"{'='*60}")
        
        data_sizes = [c.data_size for c in clients]
        bandwidths = [c.bandwidth for c in clients]
        qualities = [c.quality for c in clients]
        latencies = [c.latency for c in clients]
        
        print(f"Data Size:   min={min(data_sizes)}, max={max(data_sizes)}, mean={np.mean(data_sizes):.1f}")
        print(f"Bandwidth:   min={min(bandwidths):.2f}, max={max(bandwidths):.2f}, mean={np.mean(bandwidths):.2f}")
        print(f"Quality:     min={min(qualities):.3f}, max={max(qualities):.3f}, mean={np.mean(qualities):.3f}")
        print(f"Latency:     min={min(latencies):.2f}, max={max(latencies):.2f}, mean={np.mean(latencies):.2f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the data generator
    simulator = ClientSimulator(num_clients=50, seed=42)
    
    print("Testing ClientSimulator...")
    
    # Test regular generation
    clients = simulator.generate_clients()
    print(f"Generated {len(clients)} clients")
    simulator.print_client_stats(clients)
    
    # Test homogeneous
    homo_clients = simulator.get_homogeneous_clients()
    print("Homogeneous clients:")
    simulator.print_client_stats(homo_clients)
    
    # Test moderately heterogeneous
    mod_het_clients = simulator.get_moderately_heterogeneous_clients()
    print("Moderately heterogeneous clients:")
    simulator.print_client_stats(mod_het_clients)
    
    # Test highly heterogeneous
    high_het_clients = simulator.get_highly_heterogeneous_clients()
    print("Highly heterogeneous clients:")
    simulator.print_client_stats(high_het_clients)
