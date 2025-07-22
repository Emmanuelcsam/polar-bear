#!/usr/bin/env python3
"""
HPC (High Performance Computing) Demo
Demonstrates GPU acceleration, parallel processing, and distributed computing
"""

import subprocess
import time
import os
import sys
import json
import numpy as np

def check_gpu_availability():
    """Check if GPU acceleration is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    
    print("✗ No GPU detected - will use CPU optimizations")
    return False

def generate_large_dataset(size=100000):
    """Generate a large dataset for HPC testing"""
    print(f"\n[DEMO] Generating large dataset ({size:,} elements)...")
    
    # Create large pixel array
    pixels = np.random.randint(0, 256, size=size)
    
    # Save as pixel data
    data = {
        'pixels': pixels.tolist(),
        'size': [int(np.sqrt(size)), int(np.sqrt(size))],
        'timestamp': time.time()
    }
    
    with open('pixel_data.json', 'w') as f:
        json.dump(data, f)
    
    print(f"[DEMO] Generated {size:,} pixel dataset")
    
    # Also create some test images
    from PIL import Image
    for i in range(5):
        img_size = (100, 100)
        img_data = np.random.randint(0, 256, img_size, dtype=np.uint8)
        img = Image.fromarray(img_data)
        img.save(f'hpc_test_{i}.jpg')
    
    print("[DEMO] Created 5 test images for batch processing")

def run_module(module_name, description):
    """Run a module and capture results"""
    print(f"\n[DEMO] {description}")
    print("─" * 60)
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, module_name],
        capture_output=True,
        text=True
    )
    
    elapsed_time = time.time() - start_time
    
    # Print relevant output
    for line in result.stdout.split('\n'):
        if any(keyword in line for keyword in ['[GPU]', '[PARALLEL]', '[DIST]', '[HPC]', 'complete', 'Performance']):
            print(line)
    
    if result.returncode != 0:
        print(f"✗ Error: {result.stderr[:200]}")
    else:
        print(f"✓ Completed in {elapsed_time:.2f}s")
    
    return elapsed_time

def compare_performance():
    """Compare performance across different processing modes"""
    print("\n[DEMO] Performance Comparison")
    print("─" * 60)
    
    # Load results if available
    results = {}
    
    # GPU results
    if os.path.exists('gpu_results.json'):
        with open('gpu_results.json', 'r') as f:
            gpu_data = json.load(f)
            results['GPU'] = {
                'time': gpu_data['performance']['total_time'],
                'throughput': gpu_data['performance']['pixels_per_second'],
                'device': gpu_data['device']
            }
    
    # Parallel results
    if os.path.exists('parallel_results.json'):
        with open('parallel_results.json', 'r') as f:
            parallel_data = json.load(f)
            results['Parallel'] = {
                'time': parallel_data['performance']['total_time'],
                'throughput': parallel_data['performance']['pixels_per_second'],
                'cores': parallel_data['cpu_cores']
            }
    
    # Distributed results
    if os.path.exists('distributed_results.json'):
        with open('distributed_results.json', 'r') as f:
            dist_data = json.load(f)
            results['Distributed'] = {
                'time': dist_data['performance']['total_time'],
                'throughput': dist_data['performance']['throughput'],
                'nodes': dist_data['nodes_used']
            }
    
    # Display comparison
    if results:
        print("\nProcessing Mode    Time(s)    Throughput    Details")
        print("─" * 60)
        
        for mode, data in results.items():
            details = ""
            if 'device' in data:
                details = f"{data['device']}"
            elif 'cores' in data:
                details = f"{data['cores']} cores"
            elif 'nodes' in data:
                details = f"{data['nodes']} nodes"
            
            print(f"{mode:<15} {data['time']:>8.2f} {data['throughput']:>12.0f}/s    {details}")
    
    # Calculate speedups
    if 'Parallel' in results and results:
        baseline = results.get('Parallel', {}).get('time', 1)
        print("\nSpeedup vs Parallel CPU:")
        
        for mode, data in results.items():
            if mode != 'Parallel':
                speedup = baseline / data['time'] if data['time'] > 0 else 0
                print(f"  {mode}: {speedup:.1f}x faster")

def show_optimization_recommendations():
    """Display HPC optimization recommendations"""
    print("\n[DEMO] Optimization Recommendations")
    print("─" * 60)
    
    if os.path.exists('hpc_pipeline.json'):
        with open('hpc_pipeline.json', 'r') as f:
            pipeline = json.load(f)
        
        print("\nDetected Resources:")
        res = pipeline['resources']
        print(f"  CPU: {res['cpu_cores']} cores")
        print(f"  Memory: {res['memory_gb']:.1f} GB")
        print(f"  GPU: {res['gpu']}")
        print(f"  Compute Score: {res['compute_score']:.1f}")
        
        print("\nRecommendations:")
        for rec in pipeline['recommendations']:
            print(f"  • {rec}")
        
        print("\nOptimized Pipeline:")
        for stage in pipeline['pipeline'][:5]:  # Show first 5 stages
            opt = stage['optimization']
            print(f"  {stage['stage']}: {opt['algorithm']} "
                  f"(workers: {opt['parallel_workers']}, "
                  f"chunk: {opt['chunk_size']})")

def main():
    print("=== HPC (HIGH PERFORMANCE COMPUTING) DEMO ===")
    print("This demonstrates GPU acceleration, parallel processing,")
    print("and distributed computing capabilities\n")
    
    # Check system capabilities
    gpu_available = check_gpu_availability()
    
    # Generate test data
    generate_large_dataset(100000)
    
    # Phase 1: GPU Processing
    if gpu_available or True:  # Run anyway to show CPU fallback
        run_module('gpu_accelerator.py', 
                  "Phase 1: GPU-Accelerated Processing")
    
    # Phase 2: Parallel Processing
    run_module('parallel_processor.py', 
              "Phase 2: Multi-Core Parallel Processing")
    
    # Phase 3: Distributed Computing
    run_module('distributed_analyzer.py', 
              "Phase 3: Distributed Computing Simulation")
    
    # Phase 4: HPC Optimization
    run_module('hpc_optimizer.py', 
              "Phase 4: HPC Optimization & Benchmarking")
    
    # Performance comparison
    compare_performance()
    
    # Show recommendations
    show_optimization_recommendations()
    
    # Final summary
    print("\n" + "=" * 60)
    print("HPC DEMO COMPLETE!")
    print("=" * 60)
    
    print("\nGenerated Files:")
    hpc_files = [
        ('gpu_results.json', 'GPU processing results'),
        ('gpu_batch_results.json', 'GPU batch processing'),
        ('parallel_results.json', 'Parallel processing results'),
        ('parallel_correlations.json', 'Parallel correlation analysis'),
        ('parallel_batch.json', 'Parallel batch processing'),
        ('distributed_results.json', 'Distributed computing results'),
        ('distributed_mapreduce.json', 'Map-Reduce results'),
        ('hpc_benchmarks.json', 'Performance benchmarks'),
        ('hpc_pipeline.json', 'Optimized processing pipeline'),
        ('hpc_scaling.json', 'Scaling analysis')
    ]
    
    for filename, description in hpc_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename:<25} - {description}")
    
    print("\nKey Insights:")
    print("  1. GPU acceleration provides 10-100x speedup for parallel operations")
    print("  2. Multi-core processing scales linearly with CPU cores")
    print("  3. Distributed computing enables processing of massive datasets")
    print("  4. HPC optimizer automatically selects best algorithm for your hardware")
    
    print("\nNext Steps:")
    print("  • View 'hpc_benchmarks.json' for detailed performance metrics")
    print("  • Check 'hpc_pipeline.json' for optimized processing recommendations")
    print("  • Run individual modules with larger datasets")
    print("  • Enable GPU support for maximum performance")

if __name__ == "__main__":
    main()