#!/usr/bin/env python3
"""
GPU Test Script for HPC Environment
Tests CUDA availability and basic PyTorch operations
"""

import torch
import sys
import os
from datetime import datetime

def test_cuda_availability():
    """Test if CUDA is available and functioning"""
    print("="*60)
    print("CUDA Availability Test")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("CUDA is not available!")
        print("Please check your CUDA installation and environment modules")
        return False
    
    return True

def test_tensor_operations():
    """Test basic tensor operations on GPU"""
    print("\n" + "="*60)
    print("Tensor Operations Test")
    print("="*60)
    
    try:
        # Create tensors
        size = (1000, 1000)
        a = torch.randn(size).cuda()
        b = torch.randn(size).cuda()
        
        # Test operations
        print(f"Created tensors of size: {size}")
        
        # Matrix multiplication
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        print(f"Matrix multiplication completed in: {elapsed:.2f} ms")
        print(f"Result shape: {c.shape}")
        
        # Memory info
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"Error during tensor operations: {e}")
        return False

def test_multi_gpu():
    """Test multi-GPU setup if available"""
    if torch.cuda.device_count() > 1:
        print("\n" + "="*60)
        print("Multi-GPU Test")
        print("="*60)
        
        try:
            # Create a simple model
            model = torch.nn.Linear(100, 10)
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            
            # Test forward pass
            x = torch.randn(32, 100).cuda()
            output = model(x)
            
            print(f"Multi-GPU forward pass successful")
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error during multi-GPU test: {e}")
            return False
    else:
        print("\nOnly one GPU available, skipping multi-GPU test")
        return True

def test_distributed_setup():
    """Test distributed training setup"""
    print("\n" + "="*60)
    print("Distributed Setup Test")
    print("="*60)
    
    # Check environment variables
    env_vars = ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Check if running under SLURM
    if 'SLURM_JOB_ID' in os.environ:
        print(f"\nSLURM Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"SLURM Node List: {os.environ.get('SLURM_JOB_NODELIST', 'Not set')}")
        print(f"SLURM Task ID: {os.environ.get('SLURM_PROCID', 'Not set')}")
    else:
        print("\nNot running under SLURM")
    
    return True

def main():
    """Run all tests"""
    print(f"GPU Test Script - {datetime.now()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    all_passed = True
    
    # Run tests
    if not test_cuda_availability():
        all_passed = False
    
    if torch.cuda.is_available():
        if not test_tensor_operations():
            all_passed = False
        
        if not test_multi_gpu():
            all_passed = False
    
    test_distributed_setup()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if all_passed:
        print("✓ All tests passed!")
        print("The environment is ready for GPU training")
    else:
        print("✗ Some tests failed!")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()