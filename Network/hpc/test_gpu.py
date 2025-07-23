#!/usr/bin/env python3
"""
GPU and Distributed Training Test Script
Tests CUDA availability and distributed communication
"""

import torch
import torch.distributed as dist
import os
import sys
from datetime import datetime

def test_cuda():
    """Test CUDA availability and properties"""
    print("=" * 60)
    print("CUDA AVAILABILITY TEST")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Test tensor operations
        print("\nTesting tensor operations on GPU...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"Matrix multiplication successful. Result shape: {z.shape}")
            
            # Test memory allocation
            try:
                large_tensor = torch.randn(10000, 10000).cuda()
                print(f"Large tensor allocation successful: {large_tensor.shape}")
                del large_tensor
            except RuntimeError as e:
                print(f"Large tensor allocation failed: {e}")
            
        except Exception as e:
            print(f"GPU tensor operations failed: {e}")
    else:
        print("CUDA is not available. Please check your installation.")
    
    return cuda_available

def test_distributed():
    """Test distributed training setup"""
    print("\n" + "=" * 60)
    print("DISTRIBUTED TRAINING TEST")
    print("=" * 60)
    
    # Check environment variables
    env_vars = ['SLURM_PROCID', 'SLURM_NTASKS', 'MASTER_ADDR', 'MASTER_PORT']
    print("Environment variables:")
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Try to initialize distributed
    if 'SLURM_PROCID' in os.environ:
        try:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
            
            print(f"\nInitializing distributed with rank {rank}/{world_size}")
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
            
            print("Distributed initialization successful!")
            
            # Test all_reduce
            tensor = torch.ones(1).cuda() * (rank + 1)
            print(f"Before all_reduce: {tensor.item()}")
            
            dist.all_reduce(tensor)
            print(f"After all_reduce: {tensor.item()}")
            
            dist.destroy_process_group()
            print("Distributed test completed successfully!")
            
        except Exception as e:
            print(f"Distributed initialization failed: {e}")
    else:
        print("\nNot running under SLURM. Skipping distributed test.")

def test_performance():
    """Basic performance test"""
    if not torch.cuda.is_available():
        print("\nSkipping performance test (no GPU available)")
        return
    
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    # Test different operations
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create random matrices
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(a, b)
        
        # Time the operation
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            c = torch.matmul(a, b)
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 10  # Average time in ms
        
        flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
        tflops = (flops / elapsed / 1e9)  # TFLOPs
        
        print(f"  Average time: {elapsed:.2f} ms")
        print(f"  Performance: {tflops:.2f} TFLOPs")
        
        del a, b, c

def main():
    """Run all tests"""
    print(f"Starting GPU and Distributed Tests - {datetime.now()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test distributed if available
    test_distributed()
    
    # Test performance
    if cuda_ok:
        test_performance()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()