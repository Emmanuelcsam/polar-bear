#!/usr/bin/env python3
"""
Statistical Analysis Toolkit - Advanced Statistical Functions
Extracted from detection.py - Standalone modular script
"""

import numpy as np
import json
import sys
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatisticalAnalysisToolkit:
    """Advanced statistical analysis functions for data analysis."""
    
    def __init__(self):
        self.logger = logger
    
    def compute_skewness(self, data):
        """Compute skewness of data."""
        data = np.array(data).flatten()
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def compute_kurtosis(self, data):
        """Compute kurtosis of data."""
        data = np.array(data).flatten()
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def compute_entropy(self, data, bins=256):
        """Compute Shannon entropy."""
        data = np.array(data).flatten()
        hist, _ = np.histogram(data, bins=bins)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def compute_correlation(self, x, y):
        """Compute Pearson correlation coefficient."""
        x, y = np.array(x).flatten(), np.array(y).flatten()
        if len(x) < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.mean((x - x_mean) * (y - y_mean))
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)
    
    def compute_spearman_correlation(self, x, y):
        """Compute Spearman rank correlation."""
        x, y = np.array(x).flatten(), np.array(y).flatten()
        if len(x) < 2:
            return 0.0
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))
        return self.compute_correlation(rank_x, rank_y)
    
    def compute_ks_statistic(self, x, y):
        """Compute Kolmogorov-Smirnov statistic."""
        x_sorted = np.sort(np.array(x).flatten())
        y_sorted = np.sort(np.array(y).flatten())
        
        combined = np.concatenate([x_sorted, y_sorted])
        combined_sorted = np.sort(combined)
        
        max_diff = 0
        for val in combined_sorted:
            cdf_x = np.sum(x_sorted <= val) / len(x_sorted)
            cdf_y = np.sum(y_sorted <= val) / len(y_sorted)
            max_diff = max(max_diff, abs(cdf_x - cdf_y))
        
        return max_diff
    
    def compute_wasserstein_distance(self, x, y):
        """Compute 1D Wasserstein distance."""
        x_sorted = np.sort(np.array(x).flatten())
        y_sorted = np.sort(np.array(y).flatten())
        
        n = max(len(x_sorted), len(y_sorted))
        x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y_sorted)), y_sorted)
        
        return np.mean(np.abs(x_interp - y_interp))
    
    def compute_robust_statistics(self, data):
        """Compute robust mean and covariance using custom implementation."""
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        # Use median as initial robust mean estimate
        robust_mean = np.median(data, axis=0)
        
        # Compute deviations from median
        deviations = data - robust_mean
        mad = np.median(np.abs(deviations), axis=0)
        
        # Scale MAD to approximate standard deviation
        mad_scaled = mad * 1.4826
        mad_scaled[mad_scaled < 1e-6] = 1.0
        
        # Compute robust covariance using weighted approach
        normalized_deviations = deviations / mad_scaled
        distances = np.sqrt(np.sum(normalized_deviations**2, axis=1))
        distances = np.clip(distances, 0, 10)
        
        # Compute weights using Gaussian kernel
        weights = np.exp(-0.5 * distances)
        weight_sum = weights.sum()
        
        if weight_sum < 1e-10 or n_samples < 2:
            robust_cov = np.cov(data, rowvar=False)
            if robust_cov.ndim == 0:
                robust_cov = np.array([[robust_cov]])
        else:
            weights = weights / weight_sum
            weighted_data = data * np.sqrt(weights[:, np.newaxis])
            robust_cov = np.dot(weighted_data.T, weighted_data)
            
            effective_n = 1.0 / np.sum(weights**2)
            if effective_n > 1:
                robust_cov = robust_cov * effective_n / (effective_n - 1)
        
        # Ensure positive semi-definite
        reg_value = np.trace(robust_cov) / n_features * 1e-4
        if reg_value < 1e-6:
            reg_value = 1e-6
        robust_cov = robust_cov + np.eye(n_features) * reg_value
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            robust_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except np.linalg.LinAlgError:
            var_scale = np.var(data)
            robust_cov = np.eye(n_features) * var_scale
        
        try:
            robust_inv_cov = np.linalg.pinv(robust_cov + np.eye(n_features) * 1e-4)
        except np.linalg.LinAlgError:
            diag_values = np.diag(robust_cov)
            diag_values[diag_values < 1e-6] = 1e-6
            robust_inv_cov = np.diag(1.0 / diag_values)
        
        return robust_mean, robust_cov, robust_inv_cov
    
    def compute_comprehensive_statistics(self, data):
        """Compute comprehensive statistical summary of data."""
        data = np.array(data).flatten()
        
        if len(data) == 0:
            return {}
        
        percentiles = np.percentile(data, [5, 10, 25, 50, 75, 90, 95])
        
        stats = {
            'count': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'variance': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'median': float(np.median(data)),
            'mad': float(np.median(np.abs(data - np.median(data)))),
            'skewness': float(self.compute_skewness(data)),
            'kurtosis': float(self.compute_kurtosis(data)),
            'entropy': float(self.compute_entropy(data)),
            'energy': float(np.sum(data**2)),
            'p05': float(percentiles[0]),
            'p10': float(percentiles[1]),
            'p25': float(percentiles[2]),
            'p50': float(percentiles[3]),
            'p75': float(percentiles[4]),
            'p90': float(percentiles[5]),
            'p95': float(percentiles[6]),
            'iqr': float(percentiles[4] - percentiles[2]),
        }
        
        return stats
    
    def compare_distributions(self, data1, data2):
        """Compare two data distributions comprehensively."""
        data1 = np.array(data1).flatten()
        data2 = np.array(data2).flatten()
        
        if len(data1) == 0 or len(data2) == 0:
            return {}
        
        comparison = {
            'euclidean_distance': float(np.linalg.norm(data1[:min(len(data1), len(data2))] - data2[:min(len(data1), len(data2))])),
            'manhattan_distance': float(np.sum(np.abs(data1[:min(len(data1), len(data2))] - data2[:min(len(data1), len(data2))]))),
            'pearson_correlation': float(self.compute_correlation(data1, data2)),
            'spearman_correlation': float(self.compute_spearman_correlation(data1, data2)),
            'ks_statistic': float(self.compute_ks_statistic(data1, data2)),
            'wasserstein_distance': float(self.compute_wasserstein_distance(data1, data2)),
        }
        
        # Information theoretic measures
        bins = min(30, min(len(data1), len(data2)) // 2)
        if bins > 2:
            min_val = min(data1.min(), data2.min())
            max_val = max(data1.max(), data2.max())
            
            hist1, bin_edges = np.histogram(data1, bins=bins, range=(min_val, max_val))
            hist2, _ = np.histogram(data2, bins=bin_edges)
            
            hist1 = hist1 / (hist1.sum() + 1e-10)
            hist2 = hist2 / (hist2.sum() + 1e-10)
            
            # KL divergence
            kl_div = 0
            for i in range(len(hist1)):
                if hist1[i] > 0:
                    kl_div += hist1[i] * np.log((hist1[i] + 1e-10) / (hist2[i] + 1e-10))
            comparison['kl_divergence'] = float(kl_div)
            
            # JS divergence
            m = 0.5 * (hist1 + hist2)
            js_div = 0.5 * sum(hist1[i] * np.log((hist1[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist1)) if hist1[i] > 0)
            js_div += 0.5 * sum(hist2[i] * np.log((hist2[i] + 1e-10) / (m[i] + 1e-10)) for i in range(len(hist2)) if hist2[i] > 0)
            comparison['js_divergence'] = float(js_div)
            
            # Chi-square distance
            chi_sq = 0.5 * np.sum(np.where(hist1 + hist2 > 0, (hist1 - hist2)**2 / (hist1 + hist2 + 1e-10), 0))
            comparison['chi_square'] = float(chi_sq)
        else:
            comparison['kl_divergence'] = float('inf')
            comparison['js_divergence'] = 1.0
            comparison['chi_square'] = float('inf')
        
        return comparison
    
    def analyze_data_from_file(self, file_path, output_path=None):
        """Analyze data from CSV/JSON file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = list(data.values())
        elif file_path.suffix.lower() == '.csv':
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                data = []
                for row in reader:
                    try:
                        data.extend([float(x) for x in row if x.strip()])
                    except ValueError:
                        continue
        else:
            # Try to read as text file with numbers
            with open(file_path, 'r') as f:
                content = f.read()
                data = []
                for line in content.split('\n'):
                    try:
                        data.append(float(line.strip()))
                    except ValueError:
                        continue
        
        if not data:
            self.logger.error(f"No numeric data found in {file_path}")
            return None
        
        self.logger.info(f"Analyzing {len(data)} data points from {file_path}")
        
        # Compute comprehensive statistics
        stats = self.compute_comprehensive_statistics(data)
        
        # Compute robust statistics if enough data
        if len(data) >= 3:
            robust_mean, robust_cov, robust_inv_cov = self.compute_robust_statistics(data)
            stats['robust_mean'] = float(robust_mean[0] if robust_mean.ndim > 0 else robust_mean)
            stats['robust_variance'] = float(robust_cov[0, 0] if robust_cov.ndim > 1 else robust_cov)
        
        result = {
            'file_path': str(file_path),
            'data_points': len(data),
            'statistics': stats,
            'analyzed_at': str(np.datetime64('now')),
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Analysis saved to: {output_path}")
        
        return result
    
    def create_statistical_plots(self, data, output_dir=None):
        """Create comprehensive statistical plots."""
        data = np.array(data).flatten()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(data)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel('Value')
        
        # Q-Q plot (against normal distribution)
        from scipy import stats
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, y_vals, 'b-', linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = output_dir / 'statistical_plots.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Plots saved to: {plot_path}")
        
        plt.show()


def main():
    """Command line interface for statistical analysis."""
    parser = argparse.ArgumentParser(description='Advanced statistical analysis toolkit')
    parser.add_argument('input_file', help='Input data file (CSV, JSON, or text)')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--plots', help='Directory to save statistical plots')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    toolkit = StatisticalAnalysisToolkit()
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_stats.json"
    
    result = toolkit.analyze_data_from_file(args.input_file, output_path)
    
    if result:
        print(f"Successfully analyzed {result['data_points']} data points")
        print(f"Statistics saved to: {output_path}")
        
        if args.plots:
            # Read data again for plotting
            with open(args.input_file, 'r') as f:
                if args.input_file.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = list(data.values())
                else:
                    data = [float(line.strip()) for line in f if line.strip().replace('.', '').replace('-', '').isdigit()]
            
            toolkit.create_statistical_plots(data, args.plots)
    else:
        print("Statistical analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    import os
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available, some features may be limited")
    
    main()
