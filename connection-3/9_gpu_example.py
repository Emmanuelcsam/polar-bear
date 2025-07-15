# 9_gpu_example.py
try:
    import torch
except ImportError:
    print("Warning: PyTorch not installed. GPU functionality will be limited.")
    torch = None

# Import configuration from 0_config.py
from importlib import import_module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def check_gpu_capabilities():
    """Demonstrates basic GPU usage with PyTorch if a GPU is available."""
    print("--- GPU Capability Check ---")

    if torch is None:
        print("PyTorch not available. Cannot check GPU capabilities.")
        return False

    # Check if CUDA (NVIDIA GPU support) is available
    if not torch.cuda.is_available():
        print("GPU / CUDA is not available. PyTorch will use the CPU.")
        return False

    print(f"GPU is available! Device Name: {torch.cuda.get_device_name(0)}")

    # Define the GPU as our target device
    device = torch.device("cuda")

    # 1. Create a tensor on the CPU
    cpu_tensor = torch.randn(5, 5)
    print(f"\nCreated tensor on CPU:\n{cpu_tensor}")

    # 2. Move the tensor to the GPU
    # The .to(device) command is the key for GPU operations
    print("\nMoving tensor to GPU...")
    gpu_tensor = cpu_tensor.to(device)
    print(f"Tensor is now on device: {gpu_tensor.device}")

    # 3. Perform a computation on the GPU
    print("Performing matrix multiplication on GPU...")
    result_gpu = torch.matmul(gpu_tensor, gpu_tensor)

    # 4. Move the result back to the CPU to use with other libraries (like NumPy)
    print("Moving result back to CPU...")
    result_cpu = result_gpu.cpu()

    print(f"\nResult computed on GPU and moved to CPU:\n{result_cpu}")
    print("--- GPU Check Finished ---")
    return True

if __name__ == "__main__":
    check_gpu_capabilities()
