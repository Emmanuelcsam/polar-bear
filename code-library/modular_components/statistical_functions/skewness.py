#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def compute_skewness(data):
    """Compute skewness of data."""
    # Calculate mean of data
    mean = np.mean(data)
    # Calculate standard deviation
    std = np.std(data)
    # Handle zero standard deviation case
    if std == 0:
        return 0.0
    # Compute third standardized moment (skewness)
    return np.mean(((data - mean) / std) ** 3)


def main():
    """Standalone script to test skewness computation."""
    parser = argparse.ArgumentParser(description='Compute skewness of data')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--data', nargs='+', type=float, help='Input data values')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test data
        print("Testing skewness computation...")
        
        # Normal distribution (should be close to 0)
        normal_data = np.random.normal(0, 1, 1000)
        skew_normal = compute_skewness(normal_data)
        print(f"Normal distribution skewness: {skew_normal:.4f}")
        
        # Exponential distribution (should be positive)
        exp_data = np.random.exponential(1, 1000)
        skew_exp = compute_skewness(exp_data)
        print(f"Exponential distribution skewness: {skew_exp:.4f}")
        
        # Uniform distribution (should be close to 0)
        uniform_data = np.random.uniform(0, 1, 1000)
        skew_uniform = compute_skewness(uniform_data)
        print(f"Uniform distribution skewness: {skew_uniform:.4f}")
        
        print("✓ Skewness computation test completed!")
        
    elif args.data:
        # Compute skewness for provided data
        data = np.array(args.data)
        skewness = compute_skewness(data)
        print(f"Skewness: {skewness:.4f}")
        
    else:
        print("Usage:")
        print("  python skewness.py --test")
        print("  python skewness.py --data 1 2 3 4 5")
        sys.exit(1)


if __name__ == "__main__":
    main() 