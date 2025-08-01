#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def compute_kurtosis(data):
    """Compute kurtosis of data."""
    # Calculate mean of data
    mean = np.mean(data)
    # Calculate standard deviation
    std = np.std(data)
    # Handle zero standard deviation case
    if std == 0:
        return 0.0
    # Compute fourth standardized moment minus 3 (excess kurtosis)
    return np.mean(((data - mean) / std) ** 4) - 3


def main():
    """Standalone script to test kurtosis computation."""
    parser = argparse.ArgumentParser(description='Compute kurtosis of data')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--data', nargs='+', type=float, help='Input data values')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test data
        print("Testing kurtosis computation...")
        
        # Normal distribution (should be close to 0)
        normal_data = np.random.normal(0, 1, 1000)
        kurt_normal = compute_kurtosis(normal_data)
        print(f"Normal distribution kurtosis: {kurt_normal:.4f}")
        
        # Laplace distribution (should be positive)
        laplace_data = np.random.laplace(0, 1, 1000)
        kurt_laplace = compute_kurtosis(laplace_data)
        print(f"Laplace distribution kurtosis: {kurt_laplace:.4f}")
        
        # Uniform distribution (should be negative)
        uniform_data = np.random.uniform(0, 1, 1000)
        kurt_uniform = compute_kurtosis(uniform_data)
        print(f"Uniform distribution kurtosis: {kurt_uniform:.4f}")
        
        print("✓ Kurtosis computation test completed!")
        
    elif args.data:
        # Compute kurtosis for provided data
        data = np.array(args.data)
        kurtosis = compute_kurtosis(data)
        print(f"Kurtosis: {kurtosis:.4f}")
        
    else:
        print("Usage:")
        print("  python kurtosis.py --test")
        print("  python kurtosis.py --data 1 2 3 4 5")
        sys.exit(1)


if __name__ == "__main__":
    main() 