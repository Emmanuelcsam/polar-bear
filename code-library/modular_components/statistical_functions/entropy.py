#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def compute_entropy(data, bins=256):
    """Compute Shannon entropy."""
    # Create histogram with specified bins in range 0-255
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    # Normalize histogram to get probability distribution
    hist = hist / (hist.sum() + 1e-10)
    # Remove zero bins to avoid log(0)
    hist = hist[hist > 0]
    # Compute Shannon entropy: -Σ(p * log2(p))
    return -np.sum(hist * np.log2(hist + 1e-10))


def main():
    """Standalone script to test entropy computation."""
    parser = argparse.ArgumentParser(description='Compute entropy of data')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--data', nargs='+', type=float, help='Input data values')
    parser.add_argument('--bins', type=int, default=256, help='Number of histogram bins')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test data
        print("Testing entropy computation...")
        
        # Uniform distribution (high entropy)
        uniform_data = np.random.uniform(0, 255, 1000)
        entropy_uniform = compute_entropy(uniform_data)
        print(f"Uniform distribution entropy: {entropy_uniform:.4f}")
        
        # Normal distribution (medium entropy)
        normal_data = np.random.normal(128, 50, 1000)
        normal_data = np.clip(normal_data, 0, 255)
        entropy_normal = compute_entropy(normal_data)
        print(f"Normal distribution entropy: {entropy_normal:.4f}")
        
        # Constant data (low entropy)
        constant_data = np.full(1000, 128)
        entropy_constant = compute_entropy(constant_data)
        print(f"Constant data entropy: {entropy_constant:.4f}")
        
        print("✓ Entropy computation test completed!")
        
    elif args.data:
        # Compute entropy for provided data
        data = np.array(args.data)
        entropy = compute_entropy(data, args.bins)
        print(f"Entropy: {entropy:.4f}")
        
    else:
        print("Usage:")
        print("  python entropy.py --test")
        print("  python entropy.py --data 1 2 3 4 5")
        sys.exit(1)


if __name__ == "__main__":
    main() 