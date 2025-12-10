from datasets import load_dataset
import numpy as np
import random
from typing import List, Dict

# Import your existing Client class
try:
    from .data_generator import Client
    use_custom_client = True
except ImportError:
    use_custom_client = False

class HFClient:
    """Compatible with your existing algorithms"""
    
    def __init__(self, client_id, num_samples, bandwidth=None, latency=None):
        self.client_id = str(client_id)
        self.id = str(client_id)
        self.num_samples = num_samples
        
        np.random.seed(hash(client_id) % 2**32)
        self.bandwidth = bandwidth if bandwidth else np.random.uniform(5, 50)
        self.latency = latency if latency else np.random.uniform(20, 150)
        self.device_type = ['Xiaomi Mi10', 'Samsung S10e', 'OnePlus 7', 'Pixel 4a'][hash(client_id) % 4]
        self.quality = self._compute_quality()
    
    def _compute_quality(self):
        sample_q = min(1.0, self.num_samples / 1000.0)
        network_q = min(1.0, self.bandwidth / 100.0)
        device_q = 0.75
        quality = 0.2 + (sample_q * 0.3 + network_q * 0.3 + device_q * 0.4)
        return min(0.95, quality)

class HFfemnistLoader:
    _dataset_cache = None
    _train_data_cache = None
    
    @staticmethod
    def _load_dataset():
        """Load and cache the dataset"""
        if HFfemnistLoader._dataset_cache is None:
            try:
                HFfemnistLoader._dataset_cache = load_dataset("flwrlabs/femnist")
                HFfemnistLoader._train_data_cache = HFfemnistLoader._dataset_cache['train']
            except Exception as e:
                print(f" Error loading dataset: {e}")
                print("Run: pip install datasets pillow")
                return None, None
        return HFfemnistLoader._dataset_cache, HFfemnistLoader._train_data_cache
    
    @staticmethod
    def load_femnist(num_clients=100):
        print("Loading REAL FEMNIST from Hugging Face (3,550 users, 805k samples)")
        
        dataset, train_data = HFfemnistLoader._load_dataset()
        if dataset is None:
            return None
        
        # Group by writer_id
        writer_ids = list(set(train_data['writer_id']))[:num_clients]
        
        clients = []
        for writer_id in writer_ids:
            # Count samples for this writer
            indices = [i for i, w in enumerate(train_data['writer_id']) if w == writer_id]
            num_samples = len(indices)
            
            client = HFClient(writer_id, num_samples)
            clients.append(client)
        
        print(f" Loaded {len(clients)} real clients from Hugging Face")
        return clients
    
    @staticmethod
    def get_dataset_statistics(num_clients=100):
        """Get comprehensive statistics about the FEMNIST dataset"""
        dataset, train_data = HFfemnistLoader._load_dataset()
        if dataset is None:
            return None
        
        # Get all unique writer IDs
        all_writer_ids = list(set(train_data['writer_id']))
        
        # Sample writer IDs if needed
        if num_clients < len(all_writer_ids):
            import random
            random.seed(42)
            sampled_writer_ids = random.sample(all_writer_ids, num_clients)
        else:
            sampled_writer_ids = all_writer_ids
        
        # Collect statistics
        samples_per_writer = []
        labels_per_writer = []
        
        # Get all characters/labels from the dataset
        # FEMNIST uses 'character' field instead of 'label'
        try:
            if 'character' in train_data.column_names:
                all_characters = train_data['character']
            elif 'label' in train_data.column_names:
                all_characters = train_data['label']
            else:
                # Fallback: create dummy labels
                all_characters = [0] * len(train_data)
        except:
            all_characters = [0] * len(train_data)
        
        for writer_id in sampled_writer_ids:
            writer_indices = [i for i, w in enumerate(train_data['writer_id']) if w == writer_id]
            samples_per_writer.append(len(writer_indices))
            
            # Get labels (characters) for this writer using column access
            writer_labels = [all_characters[idx] for idx in writer_indices]
            labels_per_writer.append(writer_labels)
        
        # Flatten all labels
        all_labels = [label for labels in labels_per_writer for label in labels]
        
        # Calculate statistics
        stats = {
            'total_clients_available': len(all_writer_ids),
            'clients_analyzed': len(sampled_writer_ids),
            'total_samples': sum(samples_per_writer),
            'samples_per_client': {
                'mean': float(np.mean(samples_per_writer)),
                'median': float(np.median(samples_per_writer)),
                'std': float(np.std(samples_per_writer)),
                'min': int(np.min(samples_per_writer)),
                'max': int(np.max(samples_per_writer)),
                'q25': float(np.percentile(samples_per_writer, 25)),
                'q75': float(np.percentile(samples_per_writer, 75)),
                'data': samples_per_writer  # Include raw data for visualization
            },
            'num_classes': len(set(all_labels)),
            'class_distribution': {int(label): int(all_labels.count(label)) for label in set(all_labels)},
            'labels_per_client': {
                'mean': float(np.mean([len(set(labels)) for labels in labels_per_writer])),
                'std': float(np.std([len(set(labels)) for labels in labels_per_writer])),
                'min': int(np.min([len(set(labels)) for labels in labels_per_writer])),
                'max': int(np.max([len(set(labels)) for labels in labels_per_writer])),
                'data': [len(set(labels)) for labels in labels_per_writer]  # Include raw data
            }
        }
        
        return stats
    
    @staticmethod
    def get_statistics(clients: List[HFClient]) -> Dict:
        qualities = [c.quality for c in clients]
        bandwidths = [c.bandwidth for c in clients]
        samples = [c.num_samples for c in clients]
        
        return {
            'num_clients': len(clients),
            'quality_mean': float(np.mean(qualities)),
            'quality_std': float(np.std(qualities)),
            'bandwidth_mean': float(np.mean(bandwidths)),
            'samples_mean': float(np.mean(samples)),
            'samples_range': (int(min(samples)), int(max(samples)))
        }
    
    @staticmethod
    def print_statistics(clients: List[HFClient]):
        stats = HFfemnistLoader.get_statistics(clients)
        print("\n" + "="*80)
        print("REAL FEMNIST - Hugging Face (LEAF Benchmark)")
        print("="*80)
        print(f"Real clients: {stats['num_clients']}")
        print(f"Quality: {stats['quality_mean']:.4f} ± {stats['quality_std']:.4f}")
        print(f"Bandwidth: {stats['bandwidth_mean']:.1f} Mbps")
        print(f"Samples per client: {stats['samples_mean']:.1f} (range: {stats['samples_range'][0]}-{stats['samples_range'][1]})")
        print("="*80 + "\n")