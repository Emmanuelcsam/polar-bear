#!/usr/bin/env python3
"""
Multiply an array on the GPU (CuPy).
"""
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[HPC] CuPy not available, falling back to NumPy")
    import numpy as np

def gpu_multiply(data, multiplier=2):
    """Multiply array by given multiplier using GPU if available."""
    if CUPY_AVAILABLE:
        try:
            arr = cp.array(data)
            out = arr * multiplier
            result = cp.asnumpy(out)
            print("[HPC] GPU Result:", result)
            return result
        except Exception as e:
            print(f"[HPC] GPU error: {e}, falling back to CPU")
            return cpu_multiply(data, multiplier)
    else:
        return cpu_multiply(data, multiplier)

def cpu_multiply(data, multiplier=2):
    """Multiply array by given multiplier using CPU."""
    arr = np.array(data)
    result = arr * multiplier
    print("[HPC] CPU Result:", result)
    return result

if __name__ == "__main__":
    gpu_multiply([1, 2, 3])
