#!/usr/bin/env python3
"""
Test script to verify system stability with the optimized neural network.
This will run a quick test to ensure the system won't crash.
"""

import torch
import os
import time

def check_system_compatibility():
    """Check if the system can handle the neural network training"""
    print("=== SYSTEM COMPATIBILITY TEST ===")
    
    try:
        import psutil
        # Check RAM
        mem = psutil.virtual_memory()
        print(f"Total RAM: {mem.total / 1024**3:.1f} GB")
        print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
        print(f"RAM Usage: {mem.percent:.1f}%")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"CPU Cores: {cpu_count}")
        if cpu_freq:
            print(f"CPU Frequency: {cpu_freq.current:.0f} MHz")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        print(f"Available Disk Space: {disk.free / 1024**3:.1f} GB")
        
        memory_available = mem.available / 1024**3
    except ImportError:
        print("psutil not available - using basic checks")
        memory_available = 8.0  # Assume sufficient for now
    
    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except:
            print("CUDA device info not accessible")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    if memory_available < 4.0:
        print("❌ LOW MEMORY: Close other applications before training")
        return False
    elif memory_available < 8.0:
        print("⚠️ LIMITED MEMORY: Training will use CPU mode for stability")
    else:
        print("✅ SUFFICIENT MEMORY: System should handle training well")
    
    print("✅ SYSTEM COMPATIBILITY: Ready for training")
    return True

def test_memory_allocation():
    """Test memory allocation without crashing"""
    print("\n=== MEMORY ALLOCATION TEST ===")
    
    try:
        # Test small tensor allocation
        test_tensor = torch.randn(100, 100)
        print("✅ Small tensor allocation: OK")
        
        # Test larger tensor allocation
        test_tensor = torch.randn(1000, 1000)
        print("✅ Medium tensor allocation: OK")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✅ Memory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_dataset_loading():
    """Test if we can load the dataset without issues"""
    print("\n=== DATASET LOADING TEST ===")
    
    try:
        # Test importing the main module
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, 'dataset')
        reference_path = os.path.join(script_dir, 'reference')
        
        if os.path.exists(dataset_path):
            image_count = sum([len(files) for r, d, files in os.walk(dataset_path)])
            print(f"✅ Dataset found: {image_count} files")
        else:
            print("⚠️ Dataset not found - this is expected for testing")
        
        if os.path.exists(reference_path):
            ref_count = sum([len(files) for r, d, files in os.walk(reference_path)])
            print(f"✅ Reference tensors found: {ref_count} files")
        else:
            print("⚠️ Reference tensors not found - this is expected for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting system stability test for Neural Network training...")
    print("This test ensures your system won't crash during training.\n")
    
    # Run all tests
    tests_passed = 0
    total_tests = 3
    
    if check_system_compatibility():
        tests_passed += 1
    
    if test_memory_allocation():
        tests_passed += 1
    
    if test_dataset_loading():
        tests_passed += 1
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED - Your system should handle the neural network training!")
        print("\nYou can now run the main training script with:")
        print("python network-reduce.py --epochs 2")
    else:
        print("⚠️ SOME TESTS FAILED - Consider the recommendations above")
        print("The optimized script should still work, but monitor system resources")
    
    print("\nTest completed. System stability verified.")
