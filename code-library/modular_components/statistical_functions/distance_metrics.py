#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def compute_ks_statistic(x, y):
    """Compute Kolmogorov-Smirnov statistic."""
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Combine and sort all values
    combined = np.concatenate([x_sorted, y_sorted])
    combined_sorted = np.sort(combined)
    
    # Compute empirical CDFs and find maximum difference
    max_diff = 0
    for val in combined_sorted:
        # Compute CDF at this value for both distributions
        cdf_x = np.sum(x_sorted <= val) / len(x_sorted)
        cdf_y = np.sum(y_sorted <= val) / len(y_sorted)
        # Update maximum difference
        max_diff = max(max_diff, abs(cdf_x - cdf_y))
    
    return max_diff


def compute_wasserstein_distance(x, y):
    """Compute 1D Wasserstein distance."""
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Interpolate to same size for comparison
    n = max(len(x_sorted), len(y_sorted))
    # Create interpolation points
    x_interp = np.interp(np.linspace(0, 1, n), 
                         np.linspace(0, 1, len(x_sorted)), x_sorted)
    y_interp = np.interp(np.linspace(0, 1, n), 
                         np.linspace(0, 1, len(y_sorted)), y_sorted)
    
    # Compute average absolute difference
    return np.mean(np.abs(x_interp - y_interp))


def main():
    """Standalone script to test distance metrics computation."""
    parser = argparse.ArgumentParser(description='Compute distance metrics')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--x', nargs='+', type=float, help='First distribution values')
    parser.add_argument('--y', nargs='+', type=float, help='Second distribution values')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test data
        print("Testing distance metrics computation...")
        
        # Same distribution (should have low distance)
        x1 = np.random.normal(0, 1, 100)
        y1 = np.random.normal(0, 1, 100)
        ks_same = compute_ks_statistic(x1, y1)
        wasserstein_same = compute_wasserstein_distance(x1, y1)
        print(f"Same distribution - KS: {ks_same:.4f}, Wasserstein: {wasserstein_same:.4f}")
        
        # Different distributions (should have high distance)
        x2 = np.random.normal(0, 1, 100)
        y2 = np.random.normal(5, 1, 100)
        ks_diff = compute_ks_statistic(x2, y2)
        wasserstein_diff = compute_wasserstein_distance(x2, y2)
        print(f"Different distributions - KS: {ks_diff:.4f}, Wasserstein: {wasserstein_diff:.4f}")
        
        # Uniform vs normal
        x3 = np.random.uniform(0, 1, 100)
        y3 = np.random.normal(0.5, 0.2, 100)
        ks_uniform = compute_ks_statistic(x3, y3)
        wasserstein_uniform = compute_wasserstein_distance(x3, y3)
        print(f"Uniform vs Normal - KS: {ks_uniform:.4f}, Wasserstein: {wasserstein_uniform:.4f}")
        
        print("✓ Distance metrics computation test completed!")
        
    elif args.x and args.y:
        # Compute distance metrics for provided data
        x_data = np.array(args.x)
        y_data = np.array(args.y)
        
        ks_stat = compute_ks_statistic(x_data, y_data)
        wasserstein_dist = compute_wasserstein_distance(x_data, y_data)
        print(f"KS statistic: {ks_stat:.4f}")
        print(f"Wasserstein distance: {wasserstein_dist:.4f}")
        
    else:
        print("Usage:")
        print("  python distance_metrics.py --test")
        print("  python distance_metrics.py --x 1 2 3 4 5 --y 2 4 6 8 10")
        sys.exit(1)


if __name__ == "__main__":
    main() 