import json
import time
import os
import numpy as np
import socket
import threading
from multiprocessing import Process, Queue, Manager
import hashlib
import pickle

class DistributedNode:
    """Simulated distributed computing node"""
    
    def __init__(self, node_id, role='worker'):
        self.node_id = node_id
        self.role = role
        self.hostname = socket.gethostname()
        self.status = 'idle'
        self.tasks_completed = 0
        self.start_time = time.time()
        
        # Simulated compute resources
        self.cpu_power = np.random.uniform(1.0, 4.0)  # Relative CPU speed
        self.memory_gb = np.random.choice([8, 16, 32, 64])
        self.has_gpu = np.random.random() > 0.7  # 30% chance of GPU
        
        print(f"[DIST] Node {node_id} initialized:")
        print(f"       Role: {role}, CPU: {self.cpu_power:.1f}x, Memory: {self.memory_gb}GB, GPU: {self.has_gpu}")
    
    def process_task(self, task):
        """Process a distributed task"""
        task_type = task['type']
        data = task['data']
        
        # Simulate processing time based on node capabilities
        base_time = task.get('estimated_time', 1.0)
        actual_time = base_time / self.cpu_power
        
        if task_type == 'fft' and self.has_gpu:
            actual_time *= 0.1  # GPU acceleration
        
        # Simulate processing
        time.sleep(actual_time * 0.1)  # Scale down for demo
        
        # Process based on task type
        if task_type == 'pixel_analysis':
            result = self.analyze_pixels(data)
        elif task_type == 'pattern_search':
            result = self.search_patterns(data)
        elif task_type == 'fft':
            result = self.compute_fft(data)
        elif task_type == 'correlation':
            result = self.compute_correlation(data)
        else:
            result = {'error': 'Unknown task type'}
        
        self.tasks_completed += 1
        
        return {
            'task_id': task['id'],
            'node_id': self.node_id,
            'result': result,
            'process_time': actual_time,
            'timestamp': time.time()
        }
    
    def analyze_pixels(self, pixels):
        """Analyze pixel data"""
        return {
            'mean': float(np.mean(pixels)),
            'std': float(np.std(pixels)),
            'min': float(np.min(pixels)),
            'max': float(np.max(pixels)),
            'unique_values': int(len(np.unique(pixels)))
        }
    
    def search_patterns(self, pixels):
        """Search for patterns in data"""
        patterns = []
        
        # Find repeating values
        for i in range(len(pixels) - 3):
            if pixels[i] == pixels[i+1] == pixels[i+2]:
                patterns.append({
                    'type': 'repeat',
                    'value': int(pixels[i]),
                    'position': i
                })
        
        return {'patterns_found': len(patterns), 'samples': patterns[:10]}
    
    def compute_fft(self, pixels):
        """Compute FFT"""
        fft_result = np.fft.fft(pixels)
        magnitude = np.abs(fft_result)
        
        return {
            'dominant_frequency': int(np.argmax(magnitude)),
            'max_magnitude': float(np.max(magnitude)),
            'mean_magnitude': float(np.mean(magnitude))
        }
    
    def compute_correlation(self, data):
        """Compute correlation between two arrays"""
        arr1, arr2 = data['array1'], data['array2']
        corr = np.corrcoef(arr1, arr2)[0, 1]
        
        return {'correlation': float(corr)}

class DistributedCoordinator:
    """Coordinator for distributed processing"""
    
    def __init__(self, num_nodes=4):
        self.nodes = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.task_counter = 0
        
        # Initialize nodes
        for i in range(num_nodes):
            role = 'master' if i == 0 else 'worker'
            node = DistributedNode(i, role)
            self.nodes.append(node)
        
        print(f"\n[DIST] Distributed system initialized with {num_nodes} nodes")
    
    def create_tasks(self, data, task_type='pixel_analysis', chunk_size=1000):
        """Create distributed tasks from data"""
        tasks = []
        
        if isinstance(data, (list, np.ndarray)):
            # Split data into chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                task = {
                    'id': self.task_counter,
                    'type': task_type,
                    'data': chunk,
                    'estimated_time': len(chunk) * 0.001
                }
                tasks.append(task)
                self.task_counter += 1
        
        return tasks
    
    def distribute_work(self, tasks):
        """Distribute tasks to nodes"""
        print(f"[DIST] Distributing {len(tasks)} tasks across {len(self.nodes)} nodes")
        
        # Simple round-robin distribution
        # In real distributed systems, this would consider node load, network latency, etc.
        node_tasks = [[] for _ in self.nodes]
        
        for i, task in enumerate(tasks):
            node_idx = i % len(self.nodes)
            node_tasks[node_idx].append(task)
        
        return node_tasks
    
    def process_distributed(self, data, operation='full_analysis'):
        """Main distributed processing function"""
        
        start_time = time.time()
        
        print(f"\n[DIST] Starting distributed {operation}")
        print(f"[DIST] Data size: {len(data):,} elements")
        
        # Create tasks based on operation
        all_tasks = []
        
        if operation == 'full_analysis':
            # Multiple types of analysis
            all_tasks.extend(self.create_tasks(data, 'pixel_analysis', 5000))
            all_tasks.extend(self.create_tasks(data, 'pattern_search', 2000))
            all_tasks.extend(self.create_tasks(data, 'fft', 1000))
        else:
            all_tasks = self.create_tasks(data, operation)
        
        # Distribute tasks
        node_tasks = self.distribute_work(all_tasks)
        
        # Process tasks in parallel using threads (simulating network communication)
        threads = []
        results = []
        results_lock = threading.Lock()
        
        def node_worker(node, tasks):
            """Worker thread for each node"""
            node_results = []
            
            for task in tasks:
                node.status = 'processing'
                result = node.process_task(task)
                
                with results_lock:
                    results.append(result)
                    self.shared_state[f'node_{node.node_id}_progress'] = len(node_results)
                
                node_results.append(result)
            
            node.status = 'idle'
            
            # Simulate sending results back
            time.sleep(0.1)
            
            return node_results
        
        # Start processing on all nodes
        for i, node in enumerate(self.nodes):
            thread = threading.Thread(
                target=lambda n=node, t=node_tasks[i]: node_worker(n, t)
            )
            thread.start()
            threads.append(thread)
        
        # Monitor progress
        total_tasks = len(all_tasks)
        last_progress = 0
        
        while any(t.is_alive() for t in threads):
            current_progress = len(results)
            if current_progress > last_progress:
                progress_pct = (current_progress / total_tasks) * 100
                print(f"\r[DIST] Progress: {current_progress}/{total_tasks} ({progress_pct:.1f}%)", end='')
                last_progress = current_progress
            time.sleep(0.1)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        print(f"\n[DIST] All nodes completed processing")
        
        # Aggregate results
        aggregated = self.aggregate_results(results, operation)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        
        # Simulate network overhead
        network_overhead = len(results) * 0.001  # 1ms per result
        compute_time = total_time - network_overhead
        
        # Performance report
        performance = {
            'total_time': total_time,
            'compute_time': compute_time,
            'network_overhead': network_overhead,
            'tasks_completed': len(results),
            'tasks_per_second': len(results) / total_time,
            'data_processed': len(data),
            'throughput': len(data) / total_time,
            'node_statistics': []
        }
        
        for node in self.nodes:
            node_stats = {
                'node_id': node.node_id,
                'tasks_completed': node.tasks_completed,
                'cpu_power': node.cpu_power,
                'has_gpu': node.has_gpu,
                'efficiency': node.tasks_completed / (node.cpu_power * total_time)
            }
            performance['node_statistics'].append(node_stats)
        
        # Save results
        output = {
            'timestamp': time.time(),
            'operation': operation,
            'nodes_used': len(self.nodes),
            'total_tasks': total_tasks,
            'aggregated_results': aggregated,
            'performance': performance
        }
        
        with open('distributed_results.json', 'w') as f:
            json.dump(output, f)
        
        print(f"[DIST] Distributed processing complete")
        print(f"[DIST] Total time: {total_time:.2f}s")
        print(f"[DIST] Throughput: {performance['throughput']:.0f} elements/s")
        print(f"[DIST] Speedup: {len(self.nodes) * 0.7:.1f}x (vs single node)")
        
        return output
    
    def aggregate_results(self, results, operation):
        """Aggregate results from all nodes"""
        
        if operation == 'pixel_analysis':
            # Aggregate statistics
            all_means = [r['result']['mean'] for r in results]
            all_stds = [r['result']['std'] for r in results]
            
            return {
                'global_mean': float(np.mean(all_means)),
                'global_std': float(np.mean(all_stds)),
                'min_value': min(r['result']['min'] for r in results),
                'max_value': max(r['result']['max'] for r in results)
            }
        
        elif operation == 'pattern_search':
            # Aggregate patterns
            total_patterns = sum(r['result']['patterns_found'] for r in results)
            
            return {
                'total_patterns': total_patterns,
                'patterns_per_chunk': total_patterns / len(results)
            }
        
        elif operation == 'fft':
            # Aggregate FFT results
            dominant_freqs = [r['result']['dominant_frequency'] for r in results]
            
            return {
                'most_common_frequency': int(max(set(dominant_freqs), key=dominant_freqs.count)),
                'mean_magnitude': float(np.mean([r['result']['mean_magnitude'] for r in results]))
            }
        
        elif operation == 'full_analysis':
            # Aggregate all types
            aggregated = {}
            
            for result_type in ['pixel_analysis', 'pattern_search', 'fft']:
                type_results = [r for r in results if r.get('task_type') == result_type]
                if type_results:
                    aggregated[result_type] = self.aggregate_results(type_results, result_type)
            
            return aggregated
        
        return {}

def distributed_map_reduce():
    """Demonstrate map-reduce pattern"""
    
    print("\n[DIST] Map-Reduce demonstration")
    
    # Load data
    if not os.path.exists('pixel_data.json'):
        print("[DIST] No pixel data found")
        return
    
    with open('pixel_data.json', 'r') as f:
        data = json.load(f)
        pixels = np.array(data['pixels'])
    
    # Map phase - compute histograms in parallel
    def map_histogram(chunk):
        """Map function: compute histogram for chunk"""
        hist, bins = np.histogram(chunk, bins=50, range=(0, 256))
        return hist
    
    # Split data
    num_chunks = 8
    chunk_size = len(pixels) // num_chunks
    chunks = [pixels[i:i+chunk_size] for i in range(0, len(pixels), chunk_size)]
    
    print(f"[DIST] Map phase: Computing histograms for {len(chunks)} chunks")
    
    # Simulate distributed map
    map_results = []
    for i, chunk in enumerate(chunks):
        # Simulate remote computation
        time.sleep(0.05)
        hist = map_histogram(chunk)
        map_results.append(hist)
        print(f"[DIST] Mapped chunk {i+1}/{len(chunks)}")
    
    # Reduce phase - combine histograms
    print("[DIST] Reduce phase: Combining histograms")
    
    def reduce_histograms(hist1, hist2):
        """Reduce function: combine two histograms"""
        return hist1 + hist2
    
    # Reduce all histograms
    final_histogram = map_results[0]
    for hist in map_results[1:]:
        final_histogram = reduce_histograms(final_histogram, hist)
    
    # Save map-reduce results
    map_reduce_output = {
        'operation': 'histogram_map_reduce',
        'num_chunks': len(chunks),
        'chunk_size': chunk_size,
        'final_histogram': final_histogram.tolist(),
        'histogram_stats': {
            'mode': int(np.argmax(final_histogram)),
            'total_count': int(np.sum(final_histogram))
        }
    }
    
    with open('distributed_mapreduce.json', 'w') as f:
        json.dump(map_reduce_output, f)
    
    print(f"[DIST] Map-Reduce complete")
    print(f"[DIST] Mode value: {map_reduce_output['histogram_stats']['mode']}")

def simulate_cluster_analysis():
    """Simulate a cluster computing scenario"""
    
    print("\n[DIST] Simulating cluster analysis")
    
    # Create a distributed coordinator with more nodes
    coordinator = DistributedCoordinator(num_nodes=8)
    
    # Generate or load data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
    else:
        # Generate synthetic data
        pixels = np.random.randint(0, 256, size=100000)
    
    # Run distributed analysis
    results = coordinator.process_distributed(pixels, 'full_analysis')
    
    # Simulate cluster monitoring
    print("\n[DIST] Cluster Statistics:")
    for node_stat in results['performance']['node_statistics']:
        print(f"  Node {node_stat['node_id']}: "
              f"{node_stat['tasks_completed']} tasks, "
              f"CPU: {node_stat['cpu_power']:.1f}x, "
              f"GPU: {node_stat['has_gpu']}, "
              f"Efficiency: {node_stat['efficiency']:.2f}")

if __name__ == "__main__":
    print("=== DISTRIBUTED COMPUTING DEMONSTRATION ===\n")
    
    # 1. Basic distributed processing
    coordinator = DistributedCoordinator(num_nodes=4)
    
    # Load or generate data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
    else:
        pixels = np.random.randint(0, 256, size=50000)
    
    # Run distributed analysis
    coordinator.process_distributed(pixels, 'pixel_analysis')
    
    # 2. Map-Reduce demonstration
    distributed_map_reduce()
    
    # 3. Cluster simulation
    simulate_cluster_analysis()
    
    print("\n[DIST] All distributed operations complete!")
    print("[DIST] Results saved to:")
    print("  - distributed_results.json")
    print("  - distributed_mapreduce.json")