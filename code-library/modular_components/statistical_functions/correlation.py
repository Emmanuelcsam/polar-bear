#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def compute_correlation(x, y):
    """Compute Pearson correlation coefficient."""
    # Need at least 2 points for correlation
    if len(x) < 2:
        return 0.0
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Calculate covariance
    cov = np.mean((x - x_mean) * (y - y_mean))
    # Calculate standard deviations
    std_x = np.std(x)
    std_y = np.std(y)
    # Handle zero standard deviation
    if std_x == 0 or std_y == 0:
        return 0.0
    # Return correlation coefficient
    return cov / (std_x * std_y)


def compute_spearman_correlation(x, y):
    """Compute Spearman rank correlation."""
    # Need at least 2 points
    if len(x) < 2:
        return 0.0
    # Convert values to ranks using double argsort trick
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    # Compute Pearson correlation on ranks
    return compute_correlation(rank_x, rank_y)


def main():
    """Standalone script to test correlation computation."""
    parser = argparse.ArgumentParser(description='Compute correlation coefficients')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--x', nargs='+', type=float, help='First variable values')
    parser.add_argument('--y', nargs='+', type=float, help='Second variable values')
    
    args = parser.parse_args()
    
    if args.test:
        # Generate test data
        print("Testing correlation computation...")
        
        # Perfect positive correlation
        x1 = np.arange(10)
        y1 = x1 * 2 + 1
        corr_pos = compute_correlation(x1, y1)
        print(f"Perfect positive correlation: {corr_pos:.4f}")
        
        # Perfect negative correlation
        y2 = -x1 + 10
        corr_neg = compute_correlation(x1, y2)
        print(f"Perfect negative correlation: {corr_neg:.4f}")
        
        # No correlation
        y3 = np.random.normal(0, 1, 10)
        corr_none = compute_correlation(x1, y3)
        print(f"No correlation: {corr_none:.4f}")
        
        # Spearman correlation test
        spearman_test = compute_spearman_correlation(x1, y1)
        print(f"Spearman correlation: {spearman_test:.4f}")
        
        print("✓ Correlation computation test completed!")
        
    elif args.x and args.y:
        # Compute correlation for provided data
        x_data = np.array(args.x)
        y_data = np.array(args.y)
        
        if len(x_data) != len(y_data):
            print("Error: x and y must have the same length")
            sys.exit(1)
        
        pearson = compute_correlation(x_data, y_data)
        spearman = compute_spearman_correlation(x_data, y_data)
        print(f"Pearson correlation: {pearson:.4f}")
        print(f"Spearman correlation: {spearman:.4f}")
        
    else:
        print("Usage:")
        print("  python correlation.py --test")
        print("  python correlation.py --x 1 2 3 4 5 --y 2 4 6 8 10")
        sys.exit(1)


if __name__ == "__main__":
    main() 