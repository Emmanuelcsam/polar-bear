#!/usr/bin/env python3
"""Step by step debug to find where the hang occurs"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("1. Testing TensorProcessor...")
from fiber_tensor_processor import TensorProcessor
tp = TensorProcessor()
x = torch.randn(1, 3, 224, 224)
print("  - calculate_gradient_intensity...")
grad_info = tp.calculate_gradient_intensity(x)
print(f"    ✓ Result shape: {grad_info['gradient_map'].shape}")

print("  - calculate_pixel_positions...")
pos_info = tp.calculate_pixel_positions(x.shape)
print(f"    ✓ Result shape: {pos_info['radial_positions'].shape}")

print("\n2. Testing feature extractor directly...")
from fiber_feature_extractor import SimultaneousFeatureExtractor
fe = SimultaneousFeatureExtractor(in_channels=3, out_channels=64)
print("  - forward pass...")
try:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation took too long")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    result = fe(x)
    signal.alarm(0)
    print(f"    ✓ Result keys: {list(result.keys())}")
except TimeoutError:
    print("    ✗ TIMEOUT in SimultaneousFeatureExtractor forward")
except Exception as e:
    print(f"    ✗ ERROR: {e}")

print("\n3. Testing MultiScaleFeatureExtractor...")
from fiber_feature_extractor import MultiScaleFeatureExtractor
mfe = MultiScaleFeatureExtractor()
print("  - forward pass...")
try:
    signal.alarm(5)
    result = mfe(x, grad_info, pos_info)
    signal.alarm(0)
    print(f"    ✓ Result keys: {list(result.keys())}")
except TimeoutError:
    print("    ✗ TIMEOUT in MultiScaleFeatureExtractor forward")
except Exception as e:
    print(f"    ✗ ERROR: {e}")