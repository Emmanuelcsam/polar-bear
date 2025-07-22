import json
import time
import os
import numpy as np
import psutil
import platform
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ComputeResource:
    """Represents available compute resources"""
    cpu_cores: int
    cpu_freq_ghz: float
    memory_gb: float
    memory_available_gb: float
    gpu_available: bool
    gpu_name: str = "None"
    gpu_memory_gb: float = 0.0
    
    def score(self) -> float:
        """Calculate compute power score"""
        cpu_score = self.cpu_cores * self.cpu_freq_ghz
        memory_score = self.memory_available_gb / 8.0  # Normalize to 8GB
        gpu_score = 10.0 if self.gpu_available else 1.0
        return cpu_score * memory_score * gpu_score

class HPCOptimizer:
    """High Performance Computing optimizer"""
    
    def __init__(self):
        self.resources = self.detect_resources()
        self.optimization_history = []
        print("[HPC] HPC Optimizer initialized")
        self.print_resources()
    
    def detect_resources(self) -> ComputeResource:
        """Detect available compute resources"""
        
        # CPU information
        cpu_cores = psutil.cpu_count()
        try:
            cpu_freq = psutil.cpu_freq().current / 1000 if psutil.cpu_freq() else 2.0  # GHz
        except:
            cpu_freq = 2.0  # Default to 2 GHz if detection fails
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # GPU detection
        gpu_available = False
        gpu_name = "None"
        gpu_memory_gb = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return ComputeResource(
            cpu_cores=cpu_cores,
            cpu_freq_ghz=cpu_freq,
            memory_gb=memory_gb,
            memory_available_gb=memory_available_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb
        )
    
    def print_resources(self):
        """Print detected resources"""
        print(f"\n[HPC] System Resources:")
        print(f"  CPU: {self.resources.cpu_cores} cores @ {self.resources.cpu_freq_ghz:.1f} GHz")
        print(f"  Memory: {self.resources.memory_available_gb:.1f}/{self.resources.memory_gb:.1f} GB available")
        print(f"  GPU: {self.resources.gpu_name} ({self.resources.gpu_memory_gb:.1f} GB)" if self.resources.gpu_available else "  GPU: Not available")
        print(f"  Compute Score: {self.resources.score():.1f}")
    
    def optimize_workload(self, data_size: int, operation: str) -> Dict[str, Any]:
        """Optimize workload distribution based on resources and data"""
        
        print(f"\n[HPC] Optimizing {operation} for {data_size:,} elements")
        
        # Calculate optimal parameters
        optimization = {
            'operation': operation,
            'data_size': data_size,
            'use_gpu': False,
            'parallel_workers': 1,
            'chunk_size': data_size,
            'batch_size': 1,
            'algorithm': 'serial',
            'estimated_time': 0
        }
        
        # Decision tree for optimization
        if data_size < 1000:
            # Small data - serial processing
            optimization['algorithm'] = 'serial'
            optimization['estimated_time'] = data_size * 0.0001
            
        elif data_size < 10000:
            # Medium data - parallel CPU
            optimization['algorithm'] = 'parallel_cpu'
            optimization['parallel_workers'] = min(4, self.resources.cpu_cores)
            optimization['chunk_size'] = data_size // optimization['parallel_workers']
            optimization['estimated_time'] = data_size * 0.00005 / optimization['parallel_workers']
            
        else:
            # Large data - GPU or distributed
            if self.resources.gpu_available and operation in ['fft', 'matrix', 'neural']:
                optimization['algorithm'] = 'gpu'
                optimization['use_gpu'] = True
                optimization['batch_size'] = min(10000, data_size // 10)
                optimization['estimated_time'] = data_size * 0.00001
                
            elif self.resources.cpu_cores >= 8:
                optimization['algorithm'] = 'distributed'
                optimization['parallel_workers'] = self.resources.cpu_cores - 2
                optimization['chunk_size'] = data_size // (optimization['parallel_workers'] * 4)
                optimization['estimated_time'] = data_size * 0.00002 / optimization['parallel_workers']
                
            else:
                optimization['algorithm'] = 'parallel_cpu'
                optimization['parallel_workers'] = self.resources.cpu_cores
                optimization['chunk_size'] = data_size // (optimization['parallel_workers'] * 2)
                optimization['estimated_time'] = data_size * 0.00003 / optimization['parallel_workers']
        
        # Memory check
        required_memory_gb = data_size * 8 / (1024**3)  # 8 bytes per element
        if required_memory_gb > self.resources.memory_available_gb * 0.8:
            optimization['algorithm'] = 'streaming'
            optimization['chunk_size'] = int(self.resources.memory_available_gb * 0.5 * (1024**3) / 8)
            optimization['estimated_time'] *= 1.5  # Streaming overhead
        
        print(f"[HPC] Recommended: {optimization['algorithm']}")
        
        return optimization
    
    def benchmark_operation(self, operation: str, data_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark an operation across different data sizes"""
        
        print(f"\n[HPC] Benchmarking {operation}")
        
        results = []
        
        for size in data_sizes:
            # Generate test data
            data = np.random.rand(size).astype(np.float32)
            
            # Get optimization
            opt = self.optimize_workload(size, operation)
            
            # Measure actual time
            start_time = time.time()
            
            if operation == 'fft':
                result = np.fft.fft(data)
            elif operation == 'sort':
                result = np.sort(data)
            elif operation == 'statistics':
                result = (np.mean(data), np.std(data), np.min(data), np.max(data))
            elif operation == 'matrix':
                n = int(np.sqrt(size))
                if n * n == size:
                    matrix = data.reshape(n, n)
                    result = np.linalg.svd(matrix, compute_uv=False)
                else:
                    result = None
            
            actual_time = time.time() - start_time
            
            # Calculate efficiency
            efficiency = opt['estimated_time'] / actual_time if actual_time > 0 else 0
            
            results.append({
                'size': size,
                'algorithm': opt['algorithm'],
                'estimated_time': opt['estimated_time'],
                'actual_time': actual_time,
                'efficiency': efficiency,
                'throughput': size / actual_time if actual_time > 0 else 0
            })
            
            print(f"  Size {size:8}: {actual_time:.4f}s ({opt['algorithm']})")
        
        return {
            'operation': operation,
            'benchmarks': results,
            'timestamp': time.time()
        }
    
    def auto_optimize_pipeline(self):
        """Automatically optimize the entire processing pipeline"""
        
        print("\n[HPC] Auto-optimizing processing pipeline")
        
        # Analyze available data
        data_files = {}
        for file in ['pixel_data.json', 'patterns.json', 'neural_results.json']:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                # Estimate data size
                if 'pixels' in data:
                    size = len(data['pixels'])
                elif 'predictions' in data:
                    size = len(data['predictions'])
                elif isinstance(data, list):
                    size = len(data)
                else:
                    size = 1000  # Default estimate
                    
                data_files[file] = size
        
        # Create optimized pipeline
        pipeline = []
        total_estimated_time = 0
        
        # Stage 1: Data loading (optimized based on size)
        for file, size in data_files.items():
            opt = self.optimize_workload(size, 'load')
            pipeline.append({
                'stage': 'load',
                'file': file,
                'optimization': opt
            })
            total_estimated_time += opt['estimated_time']
        
        # Stage 2: Analysis (choose based on resources)
        if any(size > 10000 for size in data_files.values()):
            # Large data - use HPC techniques
            operations = ['fft', 'statistics', 'pattern_search']
            
            for op in operations:
                max_size = max(data_files.values()) if data_files else 10000
                opt = self.optimize_workload(max_size, op)
                pipeline.append({
                    'stage': 'analysis',
                    'operation': op,
                    'optimization': opt
                })
                total_estimated_time += opt['estimated_time']
        
        # Stage 3: Machine learning (if GPU available)
        if self.resources.gpu_available:
            opt = self.optimize_workload(max(data_files.values(), default=10000), 'neural')
            pipeline.append({
                'stage': 'ml',
                'operation': 'neural_training',
                'optimization': opt
            })
            total_estimated_time += opt['estimated_time']
        
        # Save optimized pipeline
        pipeline_output = {
            'timestamp': time.time(),
            'resources': {
                'cpu_cores': self.resources.cpu_cores,
                'memory_gb': self.resources.memory_gb,
                'gpu': self.resources.gpu_name if self.resources.gpu_available else 'None',
                'compute_score': self.resources.score()
            },
            'data_sizes': data_files,
            'pipeline': pipeline,
            'total_stages': len(pipeline),
            'estimated_total_time': total_estimated_time,
            'recommendations': self.generate_recommendations()
        }
        
        with open('hpc_pipeline.json', 'w') as f:
            json.dump(pipeline_output, f)
        
        print(f"[HPC] Optimized pipeline with {len(pipeline)} stages")
        print(f"[HPC] Estimated total time: {total_estimated_time:.2f}s")
        
        return pipeline_output
    
    def generate_recommendations(self) -> List[str]:
        """Generate HPC recommendations based on resources"""
        
        recommendations = []
        
        # CPU recommendations
        if self.resources.cpu_cores < 4:
            recommendations.append("Consider upgrading to a system with more CPU cores for better parallel performance")
        elif self.resources.cpu_cores >= 16:
            recommendations.append("Excellent CPU resources - use distributed computing for large datasets")
        
        # Memory recommendations
        if self.resources.memory_available_gb < 4:
            recommendations.append("Low memory available - use streaming algorithms for large datasets")
        elif self.resources.memory_available_gb > 32:
            recommendations.append("Ample memory available - can process large datasets in-memory")
        
        # GPU recommendations
        if not self.resources.gpu_available:
            recommendations.append("No GPU detected - consider GPU for 10-100x speedup on parallel operations")
        else:
            recommendations.append(f"GPU available ({self.resources.gpu_name}) - prioritize GPU-accelerated algorithms")
        
        # Algorithm recommendations
        if self.resources.score() > 100:
            recommendations.append("High-performance system detected - suitable for complex ML and HPC workloads")
        
        return recommendations
    
    def simulate_scaling(self):
        """Simulate performance scaling"""
        
        print("\n[HPC] Simulating performance scaling")
        
        data_sizes = [1000, 10000, 100000, 1000000]
        algorithms = ['serial', 'parallel_cpu', 'gpu', 'distributed']
        
        scaling_results = {}
        
        for algo in algorithms:
            results = []
            
            for size in data_sizes:
                # Simulate processing time based on algorithm
                if algo == 'serial':
                    time_estimate = size * 0.0001
                elif algo == 'parallel_cpu':
                    time_estimate = size * 0.0001 / self.resources.cpu_cores
                elif algo == 'gpu':
                    time_estimate = size * 0.00001 if self.resources.gpu_available else size * 0.0001
                elif algo == 'distributed':
                    time_estimate = size * 0.0001 / (self.resources.cpu_cores * 2)
                
                results.append({
                    'size': size,
                    'time': time_estimate,
                    'throughput': size / time_estimate
                })
            
            scaling_results[algo] = results
        
        # Save scaling results
        with open('hpc_scaling.json', 'w') as f:
            json.dump({
                'algorithms': scaling_results,
                'optimal_crossover_points': {
                    'serial_to_parallel': 5000,
                    'parallel_to_gpu': 50000,
                    'gpu_to_distributed': 1000000
                }
            }, f)
        
        print("[HPC] Scaling simulation complete")
        
        # Print scaling summary
        print("\n[HPC] Scaling Summary (throughput elements/s):")
        print(f"{'Size':<10} {'Serial':<15} {'Parallel':<15} {'GPU':<15} {'Distributed':<15}")
        
        for i, size in enumerate(data_sizes):
            row = f"{size:<10}"
            for algo in algorithms:
                throughput = scaling_results[algo][i]['throughput']
                row += f"{throughput:<15.0f}"
            print(row)

def main():
    """Main HPC optimization demonstration"""
    
    print("=== HPC OPTIMIZATION SYSTEM ===")
    
    # Initialize optimizer
    optimizer = HPCOptimizer()
    
    # 1. Benchmark operations
    print("\n--- Performance Benchmarking ---")
    operations = ['fft', 'sort', 'statistics']
    data_sizes = [1000, 10000, 100000]
    
    benchmark_results = []
    for op in operations:
        result = optimizer.benchmark_operation(op, data_sizes)
        benchmark_results.append(result)
    
    # Save benchmark results
    with open('hpc_benchmarks.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'system': platform.platform(),
            'resources': optimizer.resources.__dict__,
            'benchmarks': benchmark_results
        }, f)
    
    # 2. Auto-optimize pipeline
    print("\n--- Pipeline Optimization ---")
    pipeline = optimizer.auto_optimize_pipeline()
    
    # 3. Scaling simulation
    print("\n--- Scaling Analysis ---")
    optimizer.simulate_scaling()
    
    print("\n[HPC] Complete HPC analysis finished!")
    print("[HPC] Results saved to:")
    print("  - hpc_benchmarks.json   (performance benchmarks)")
    print("  - hpc_pipeline.json     (optimized pipeline)")
    print("  - hpc_scaling.json      (scaling analysis)")
    
    # Print recommendations
    print("\n[HPC] Recommendations:")
    for rec in pipeline['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    main()