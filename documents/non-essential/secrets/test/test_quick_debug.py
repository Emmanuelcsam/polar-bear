#!/usr/bin/env python3
"""Quick debug test to isolate the hanging issue"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_integrated_network import FiberOpticsIntegratedNetwork

print("Creating network...")
network = FiberOpticsIntegratedNetwork()
network.eval()
print("Network created")

print("Creating test input...")
x = torch.randn(1, 3, 224, 224)
print(f"Input shape: {x.shape}")

print("Starting forward pass...")
with torch.no_grad():
    try:
        # Add timeout using alarm (Linux only)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Forward pass took too long")
        
        # Set alarm for 10 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        output = network(x)
        
        # Cancel alarm
        signal.alarm(0)
        
        print("Forward pass completed!")
        print(f"Output keys: {list(output.keys())}")
    except TimeoutError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()