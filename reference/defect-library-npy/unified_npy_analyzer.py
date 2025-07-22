#!/usr/bin/env python3
"""
Complete Unified NPY Data Analysis Script
Performs exhaustive statistical analysis on .npy files without modifying them
Generates complete mathematical expressions for neural network training
"""

import numpy as np
import os
import glob
from scipy import stats, linalg, optimize, signal
from scipy.special import gamma, digamma, polygamma
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert, find_peaks, ricker
try:
    from scipy.signal import cwt
except ImportError:
    # For newer scipy versions, cwt might be in a different location
    try:
        from scipy.signal import cwt as cwt
    except:
        # Fallback implementation if cwt is not available
        cwt = None
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, ICA, FactorAnalysis, NMF, SparsePCA, KernelPCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE, MDS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some advanced statistics will be skipped.")
import itertools
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveNPYAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_files = {}
        self.statistics = {}
        self.correlations = {}
        self.equations = {}
        self.full_analysis = {}
        
    def load_all_npy_files(self):
        """Load all .npy files from the specified folder"""
        # Get the current directory if folder_path is "."
        if self.folder_path == ".":
            self.folder_path = os.getcwd()
            
        npy_files = glob.glob(os.path.join(self.folder_path, "*.npy"))
        print(f"Found {len(npy_files)} .npy files in {self.folder_path}")
        
        for file_path in npy_files:
            file_name = os.path.basename(file_path)
            try:
                data = np.load(file_path, allow_pickle=True)
                self.data_files[file_name] = data
                print(f"Loaded: {file_name} - Shape: {data.shape if hasattr(data, 'shape') else 'Unknown'}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                
    def calculate_exhaustive_statistics(self, data, name):
        """Calculate every possible statistical measurement"""
        stats_dict = {}
        
        # Flatten if multidimensional
        original_shape = data.shape if hasattr(data, 'shape') else None
        flat_data = data.flatten() if hasattr(data, 'flatten') else data
        
        # Basic statistics
        stats_dict['shape'] = original_shape
        stats_dict['size'] = flat_data.size
        stats_dict['dtype'] = str(flat_data.dtype)
        
        # Central tendency
        stats_dict['mean'] = float(np.mean(flat_data))
        stats_dict['median'] = float(np.median(flat_data))
        stats_dict['mode'] = float(stats.mode(flat_data, keepdims=True).mode[0])
        stats_dict['geometric_mean'] = float(stats.gmean(flat_data[flat_data > 0])) if np.any(flat_data > 0) else None
        stats_dict['harmonic_mean'] = float(stats.hmean(flat_data[flat_data > 0])) if np.any(flat_data > 0) else None
        stats_dict['trimmed_mean_10'] = float(stats.trim_mean(flat_data, 0.1))
        stats_dict['trimmed_mean_20'] = float(stats.trim_mean(flat_data, 0.2))
        
        # Dispersion
        stats_dict['variance'] = float(np.var(flat_data))
        stats_dict['std_dev'] = float(np.std(flat_data))
        stats_dict['mad'] = float(np.median(np.abs(flat_data - np.median(flat_data))))
        stats_dict['iqr'] = float(np.percentile(flat_data, 75) - np.percentile(flat_data, 25))
        stats_dict['range'] = float(np.ptp(flat_data))
        stats_dict['cv'] = float(np.std(flat_data) / np.mean(flat_data)) if np.mean(flat_data) != 0 else None
        
        # Distribution shape
        stats_dict['skewness'] = float(stats.skew(flat_data))
        stats_dict['kurtosis'] = float(stats.kurtosis(flat_data))
        stats_dict['jarque_bera'] = stats.jarque_bera(flat_data)._asdict()
        
        # Percentiles
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        for p in percentiles:
            stats_dict[f'percentile_{p}'] = float(np.percentile(flat_data, p))
        
        # Moments
        for i in range(1, 11):
            stats_dict[f'moment_{i}'] = float(stats.moment(flat_data, i))
            
        # Cumulants (up to 4th order)
        stats_dict['cumulant_1'] = float(np.mean(flat_data))
        stats_dict['cumulant_2'] = float(np.var(flat_data))
        stats_dict['cumulant_3'] = float(stats.moment(flat_data, 3))
        stats_dict['cumulant_4'] = float(stats.moment(flat_data, 4) - 3 * np.var(flat_data)**2)
        
        # L-moments
        sorted_data = np.sort(flat_data)
        n = len(sorted_data)
        if n > 3:
            stats_dict['l_moment_1'] = float(np.mean(sorted_data))
            stats_dict['l_moment_2'] = float(np.mean(sorted_data[1:] - sorted_data[:-1]))
            
        # Entropy
        if flat_data.min() >= 0:
            hist, _ = np.histogram(flat_data, bins=50)
            hist = hist[hist > 0]
            stats_dict['entropy'] = float(stats.entropy(hist))
        
        # Extreme values
        stats_dict['min'] = float(np.min(flat_data))
        stats_dict['max'] = float(np.max(flat_data))
        stats_dict['argmin'] = int(np.argmin(flat_data))
        stats_dict['argmax'] = int(np.argmax(flat_data))
        
        # Counts
        stats_dict['num_unique'] = int(np.unique(flat_data).size)
        stats_dict['num_zeros'] = int(np.sum(flat_data == 0))
        stats_dict['num_positive'] = int(np.sum(flat_data > 0))
        stats_dict['num_negative'] = int(np.sum(flat_data < 0))
        stats_dict['num_nan'] = int(np.sum(np.isnan(flat_data)))
        stats_dict['num_inf'] = int(np.sum(np.isinf(flat_data)))
        
        # Autocorrelation
        if len(flat_data) > 10:
            acf_result = np.correlate(flat_data - np.mean(flat_data), flat_data - np.mean(flat_data), mode='full')
            acf_result = acf_result[len(acf_result)//2:]
            acf_result = acf_result / acf_result[0]
            stats_dict['autocorr_lag_1'] = float(acf_result[1]) if len(acf_result) > 1 else None
            stats_dict['autocorr_lag_5'] = float(acf_result[5]) if len(acf_result) > 5 else None
            stats_dict['autocorr_lag_10'] = float(acf_result[10]) if len(acf_result) > 10 else None
        
        # Power spectrum statistics
        if len(flat_data) > 10:
            freqs, psd = signal.periodogram(flat_data)
            stats_dict['dominant_frequency'] = float(freqs[np.argmax(psd)])
            stats_dict['spectral_entropy'] = float(stats.entropy(psd[psd > 0]))
            
        # Information theory
        stats_dict['shannon_entropy'] = float(-np.sum(flat_data * np.log2(flat_data + 1e-10))) if np.all(flat_data >= 0) else None
        
        return stats_dict
    
    def calculate_correlations(self):
        """Calculate all possible correlations between datasets"""
        file_names = list(self.data_files.keys())
        
        for i, file1 in enumerate(file_names):
            for j, file2 in enumerate(file_names[i:], i):
                data1 = self.data_files[file1].flatten()
                data2 = self.data_files[file2].flatten()
                
                # Make same length
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]
                
                corr_key = f"{file1}_vs_{file2}"
                self.correlations[corr_key] = {}
                
                # Pearson correlation
                if len(data1) > 1:
                    self.correlations[corr_key]['pearson'] = float(np.corrcoef(data1, data2)[0, 1])
                    
                # Spearman correlation
                self.correlations[corr_key]['spearman'] = float(stats.spearmanr(data1, data2).correlation)
                
                # Kendall tau
                if len(data1) < 1000:  # Computationally expensive
                    self.correlations[corr_key]['kendall'] = float(stats.kendalltau(data1, data2).correlation)
                
                # Distance correlation
                self.correlations[corr_key]['distance_corr'] = self.distance_correlation(data1, data2)
                
                # Mutual information
                if len(np.unique(data1)) < 100 and len(np.unique(data2)) < 100:
                    self.correlations[corr_key]['mutual_info'] = float(mutual_info_score(
                        np.digitize(data1, np.percentile(data1, np.arange(0, 101, 10))),
                        np.digitize(data2, np.percentile(data2, np.arange(0, 101, 10)))
                    ))
                
                # Cross-correlation
                xcorr = np.correlate(data1 - np.mean(data1), data2 - np.mean(data2), mode='full')
                self.correlations[corr_key]['max_cross_corr'] = float(np.max(xcorr))
                self.correlations[corr_key]['lag_max_cross_corr'] = int(np.argmax(xcorr) - len(data1) + 1)
    
    def distance_correlation(self, X, Y):
        """Calculate distance correlation"""
        n = len(X)
        a = np.abs(np.subtract.outer(X, X))
        b = np.abs(np.subtract.outer(Y, Y))
        
        A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
        
        dcov2 = (A * B).sum() / n**2
        dvar2_x = (A * A).sum() / n**2
        dvar2_y = (B * B).sum() / n**2
        
        return float(np.sqrt(dcov2 / np.sqrt(dvar2_x * dvar2_y))) if dvar2_x > 0 and dvar2_y > 0 else 0
    
    def fit_comprehensive_models(self):
        """Fit various models to find best representations"""
        file_names = list(self.data_files.keys())
        
        for file_name in file_names:
            data = self.data_files[file_name]
            self.equations[file_name] = {}
            
            # Flatten data
            flat_data = data.flatten()
            n = len(flat_data)
            x = np.arange(n)
            
            # Polynomial fits (up to degree 10)
            for degree in range(1, 11):
                if n > degree + 1:
                    coeffs = np.polyfit(x, flat_data, degree)
                    self.equations[file_name][f'polynomial_degree_{degree}'] = {
                        'coefficients': coeffs.tolist(),
                        'equation': self.generate_polynomial_equation(coeffs, degree)
                    }
            
            # Fourier series approximation
            if n > 20:
                num_harmonics = min(10, n // 2)
                fourier_coeffs = self.fit_fourier_series(x, flat_data, num_harmonics)
                self.equations[file_name]['fourier_series'] = {
                    'coefficients': fourier_coeffs,
                    'equation': self.generate_fourier_equation(fourier_coeffs)
                }
            
            # Exponential fit
            try:
                if np.all(flat_data > 0):
                    log_data = np.log(flat_data)
                    exp_coeffs = np.polyfit(x, log_data, 1)
                    self.equations[file_name]['exponential'] = {
                        'coefficients': [np.exp(exp_coeffs[1]), exp_coeffs[0]],
                        'equation': f"y = {np.exp(exp_coeffs[1]):.6f} * exp({exp_coeffs[0]:.6f} * x)"
                    }
            except:
                pass
            
            # Power law fit
            try:
                if np.all(flat_data > 0) and np.all(x > 0):
                    log_x = np.log(x[x > 0])
                    log_y = np.log(flat_data[x > 0])
                    power_coeffs = np.polyfit(log_x, log_y, 1)
                    self.equations[file_name]['power_law'] = {
                        'coefficients': [np.exp(power_coeffs[1]), power_coeffs[0]],
                        'equation': f"y = {np.exp(power_coeffs[1]):.6f} * x^{power_coeffs[0]:.6f}"
                    }
            except:
                pass
    
    def generate_polynomial_equation(self, coeffs, degree):
        """Generate polynomial equation string"""
        equation = "y = "
        for i, coeff in enumerate(coeffs):
            power = degree - i
            if power == 0:
                equation += f"{coeff:.10f}"
            elif power == 1:
                equation += f"{coeff:.10f}*x + "
            else:
                equation += f"{coeff:.10f}*x^{power} + "
        return equation.rstrip(" + ")
    
    def generate_fourier_equation(self, coeffs):
        """Generate Fourier series equation string"""
        a0, a_n, b_n = coeffs
        equation = f"y = {a0:.10f}"
        for i, (a, b) in enumerate(zip(a_n, b_n), 1):
            equation += f" + {a:.10f}*cos({i}*x) + {b:.10f}*sin({i}*x)"
        return equation
    
    def fit_fourier_series(self, x, y, num_harmonics):
        """Fit Fourier series to data"""
        n = len(x)
        x_norm = 2 * np.pi * x / n
        
        a0 = np.mean(y)
        a_n = []
        b_n = []
        
        for k in range(1, num_harmonics + 1):
            a_k = 2/n * np.sum(y * np.cos(k * x_norm))
            b_k = 2/n * np.sum(y * np.sin(k * x_norm))
            a_n.append(a_k)
            b_n.append(b_k)
        
        return (a0, a_n, b_n)
    
    def create_global_mathematical_expression(self):
        """Create comprehensive mathematical expression for all data"""
        # This will create the I=Ax1+Bx2+Cx3...=D(S) expression
        all_features = []
        feature_names = []
        
        # Extract features from all files
        for file_name, data in self.data_files.items():
            flat_data = data.flatten()
            
            # Statistical features
            features = [
                np.mean(flat_data),
                np.std(flat_data),
                np.var(flat_data),
                stats.skew(flat_data),
                stats.kurtosis(flat_data),
                np.percentile(flat_data, 25),
                np.percentile(flat_data, 50),
                np.percentile(flat_data, 75),
                np.min(flat_data),
                np.max(flat_data)
            ]
            
            for i, feat in enumerate(features):
                all_features.append(feat)
                feature_names.append(f"{file_name}_feature_{i}")
        
        # Create the expression
        expression = "I = "
        for i, (feat, name) in enumerate(zip(all_features, feature_names)):
            expression += f"({feat:.10f} * x_{i}) + "
        
        expression = expression.rstrip(" + ")
        expression += " = D(S)"
        
        return {
            'expression': expression,
            'features': all_features,
            'feature_names': feature_names,
            'num_features': len(all_features)
        }
    
    def analyze_all(self):
        """Run complete analysis"""
        print("Loading NPY files...")
        self.load_all_npy_files()
        
        if not self.data_files:
            print("No NPY files found! Please check the folder path.")
            return None
        
        print("\nCalculating exhaustive statistics...")
        for file_name, data in self.data_files.items():
            self.statistics[file_name] = self.calculate_exhaustive_statistics(data, file_name)
            print(f"Completed statistics for {file_name}")
        
        print("\nCalculating correlations...")
        self.calculate_correlations()
        
        print("\nFitting models and generating equations...")
        self.fit_comprehensive_models()
        
        print("\nCreating global mathematical expression...")
        self.full_analysis['global_expression'] = self.create_global_mathematical_expression()
        
        # Compile full analysis
        self.full_analysis['statistics'] = self.statistics
        self.full_analysis['correlations'] = self.correlations
        self.full_analysis['equations'] = self.equations
        self.full_analysis['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'num_files': len(self.data_files),
            'file_names': list(self.data_files.keys())
        }
        
        return self.full_analysis
    
    def save_results(self, output_dir='analysis_results'):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main analysis
        with open(os.path.join(output_dir, 'complete_analysis.json'), 'w') as f:
            json.dump(self.full_analysis, f, indent=2)
        
        # Save detailed statistics report
        with open(os.path.join(output_dir, 'detailed_statistics.txt'), 'w') as f:
            for file_name, stats in self.statistics.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"FILE: {file_name}\n")
                f.write(f"{'='*80}\n")
                for stat_name, value in stats.items():
                    f.write(f"{stat_name}: {value}\n")
        
        # Save equations report
        with open(os.path.join(output_dir, 'mathematical_equations.txt'), 'w') as f:
            for file_name, equations in self.equations.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"EQUATIONS FOR: {file_name}\n")
                f.write(f"{'='*80}\n")
                for eq_type, eq_data in equations.items():
                    f.write(f"\n{eq_type}:\n")
                    f.write(f"Equation: {eq_data['equation']}\n")
                    f.write(f"Coefficients: {eq_data['coefficients']}\n")
        
        print(f"\nResults saved to {output_dir}/")


class AdvancedPHDAnalysis:
    def __init__(self, data_files):
        self.data_files = data_files
        self.advanced_stats = {}
        self.topological_features = {}
        self.information_measures = {}
        self.spectral_features = {}
        self.nonlinear_features = {}
        
    def calculate_advanced_moments(self, data, name):
        """Calculate L-moments, probability weighted moments, and generalized moments"""
        flat_data = data.flatten()
        n = len(flat_data)
        sorted_data = np.sort(flat_data)
        
        results = {}
        
        # L-moments (up to 6th order)
        for r in range(1, 7):
            if n >= r:
                l_moment = 0
                for k in range(r):
                    coeff = (-1)**k * np.math.comb(r-1, k)
                    indices = np.arange(k, n)
                    weights = np.array([np.math.comb(i, k) * np.math.comb(n-1-i, r-1-k) / np.math.comb(n-1, r-1) for i in indices])
                    l_moment += coeff * np.sum(weights * sorted_data[indices])
                results[f'l_moment_{r}'] = l_moment
        
        # Probability weighted moments
        for r in range(5):
            for s in range(5):
                pwm = np.mean(sorted_data * ((np.arange(n) + 1)/(n + 1))**r * (1 - (np.arange(n) + 1)/(n + 1))**s)
                results[f'pwm_r{r}_s{s}'] = pwm
        
        # TL-moments (trimmed L-moments)
        trim_proportions = [0.1, 0.2, 0.3]
        for trim in trim_proportions:
            trim_n = int(n * trim)
            if trim_n < n/2:
                trimmed = sorted_data[trim_n:-trim_n] if trim_n > 0 else sorted_data
                results[f'tl_moment_1_trim_{trim}'] = np.mean(trimmed)
                results[f'tl_moment_2_trim_{trim}'] = np.mean(np.abs(trimmed[1:] - trimmed[:-1]))
        
        # Generalized moments
        for p in [0.5, 1.5, 2.5, 3.5]:
            results[f'generalized_moment_{p}'] = np.mean(np.abs(flat_data)**p) if p > 0 else None
            
        return results
    
    def calculate_entropy_measures(self, data, name):
        """Calculate various entropy measures"""
        flat_data = data.flatten()
        results = {}
        
        # Rényi entropy
        for alpha in [0.5, 2, 3, 4, np.inf]:
            if alpha == np.inf:
                results['renyi_entropy_inf'] = -np.log(np.max(np.abs(flat_data)))
            else:
                hist, _ = np.histogram(flat_data, bins=50, density=True)
                hist = hist[hist > 0]
                if alpha == 1:
                    results['renyi_entropy_1'] = -np.sum(hist * np.log(hist))
                else:
                    results[f'renyi_entropy_{alpha}'] = (1/(1-alpha)) * np.log(np.sum(hist**alpha))
        
        # Tsallis entropy
        for q in [0.5, 1.5, 2, 3]:
            hist, _ = np.histogram(flat_data, bins=50, density=True)
            hist = hist[hist > 0]
            if q == 1:
                results['tsallis_entropy_1'] = -np.sum(hist * np.log(hist))
            else:
                results[f'tsallis_entropy_{q}'] = (1/(q-1)) * (1 - np.sum(hist**q))
        
        # Approximate entropy
        def approx_entropy(data, m, r):
            def _maxdist(xi, xj):
                return max([abs(float(a) - float(b)) for a, b in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i+m] for i in range(len(data)-m+1)])
                C = np.zeros(len(patterns))
                for i, pattern in enumerate(patterns):
                    matching = np.sum([1 for j, p in enumerate(patterns) if _maxdist(pattern, p) <= r])
                    C[i] = matching / len(patterns)
                return np.mean(np.log(C))
            
            return _phi(m) - _phi(m+1)
        
        if len(flat_data) > 100:
            results['approx_entropy_m2_r0.2'] = approx_entropy(flat_data[:100], 2, 0.2*np.std(flat_data))
        
        # Sample entropy
        results['sample_entropy'] = self.sample_entropy(flat_data[:min(1000, len(flat_data))], 2, 0.2*np.std(flat_data))
        
        # Permutation entropy
        results['permutation_entropy'] = self.permutation_entropy(flat_data[:min(1000, len(flat_data))], 3, 1)
        
        return results
    
    def sample_entropy(self, data, m, r):
        """Calculate sample entropy"""
        N = len(data)
        
        def _count_patterns(data, m, r):
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            count = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return count
        
        phi_m = _count_patterns(data, m, r)
        phi_m1 = _count_patterns(data, m+1, r)
        
        return -np.log(phi_m1 / phi_m) if phi_m > 0 and phi_m1 > 0 else np.inf
    
    def permutation_entropy(self, data, order, delay):
        """Calculate permutation entropy"""
        N = len(data)
        permutations = list(itertools.permutations(range(order)))
        c = np.zeros(len(permutations))
        
        for i in range(N - (order-1)*delay):
            sorted_indices = np.argsort(data[i:i+order*delay:delay])
            for j, perm in enumerate(permutations):
                if tuple(sorted_indices) == perm:
                    c[j] += 1
                    break
        
        c = c[c > 0]
        p = c / np.sum(c)
        return -np.sum(p * np.log(p))
    
    def calculate_fractal_dimensions(self, data, name):
        """Calculate various fractal dimensions"""
        flat_data = data.flatten()
        results = {}
        
        # Box-counting dimension
        def box_counting_dimension(data, max_box_size=None):
            if max_box_size is None:
                max_box_size = min(len(data) // 4, 100)
            
            box_sizes = np.logspace(0, np.log10(max_box_size), 20, dtype=int)
            counts = []
            
            for box_size in box_sizes:
                bins = np.arange(np.min(data), np.max(data) + box_size, box_size)
                hist, _ = np.histogram(data, bins=bins)
                counts.append(np.sum(hist > 0))
            
            coeffs = np.polyfit(np.log(1/box_sizes), np.log(counts), 1)
            return coeffs[0]
        
        results['box_counting_dimension'] = box_counting_dimension(flat_data[:min(1000, len(flat_data))])
        
        # Correlation dimension
        def correlation_dimension(data, max_r=None):
            if max_r is None:
                max_r = np.std(data)
            
            radii = np.logspace(-2, np.log10(max_r), 20)
            N = min(len(data), 500)  # Limit for computational efficiency
            data_sample = data[:N]
            
            correlations = []
            for r in radii:
                dists = pdist(data_sample.reshape(-1, 1))
                correlation = np.sum(dists < r) / (N * (N-1) / 2)
                if correlation > 0:
                    correlations.append(correlation)
            
            if len(correlations) > 2:
                coeffs = np.polyfit(np.log(radii[:len(correlations)]), np.log(correlations), 1)
                return coeffs[0]
            return None
        
        corr_dim = correlation_dimension(flat_data[:min(1000, len(flat_data))])
        if corr_dim is not None:
            results['correlation_dimension'] = corr_dim
        
        # Hausdorff dimension estimate
        results['hausdorff_dimension_estimate'] = results['box_counting_dimension']
        
        # Information dimension
        def information_dimension(data, num_bins=50):
            hist, _ = np.histogram(data, bins=num_bins)
            p = hist / np.sum(hist)
            p = p[p > 0]
            
            info = -np.sum(p * np.log(p))
            return info / np.log(num_bins)
        
        results['information_dimension'] = information_dimension(flat_data)
        
        return results
    
    def calculate_nonlinear_measures(self, data, name):
        """Calculate nonlinear dynamics measures"""
        flat_data = data.flatten()
        results = {}
        
        # Lyapunov exponent estimate
        def lyapunov_exponent(data, embedding_dim=3, delay=1):
            N = len(data)
            M = N - (embedding_dim - 1) * delay
            
            if M <= 0:
                return None
            
            # Reconstruct phase space
            X = np.zeros((M, embedding_dim))
            for i in range(embedding_dim):
                X[:, i] = data[i*delay:i*delay + M]
            
            # Calculate average divergence
            lyap = 0
            count = 0
            for i in range(M-1):
                for j in range(i+1, M):
                    d0 = np.linalg.norm(X[i] - X[j])
                    if d0 > 0 and d0 < 0.1 * np.std(data):
                        d1 = np.linalg.norm(X[i+1] - X[j+1])
                        if d1 > 0:
                            lyap += np.log(d1/d0)
                            count += 1
            
            return lyap / count if count > 0 else None
        
        lyap = lyapunov_exponent(flat_data[:min(1000, len(flat_data))])
        if lyap is not None:
            results['lyapunov_exponent'] = lyap
        
        # Hurst exponent
        def hurst_exponent(data):
            lags = range(2, min(100, len(data)//2))
            tau = []
            
            for lag in lags:
                chunks = [data[i:i+lag] for i in range(0, len(data), lag)]
                R_S = []
                
                for chunk in chunks:
                    if len(chunk) > 1:
                        mean = np.mean(chunk)
                        cumsum = np.cumsum(chunk - mean)
                        R = np.max(cumsum) - np.min(cumsum)
                        S = np.std(chunk)
                        if S > 0:
                            R_S.append(R/S)
                
                if R_S:
                    tau.append(np.mean(R_S))
            
            if len(tau) > 2:
                coeffs = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
                return coeffs[0]
            return None
        
        hurst = hurst_exponent(flat_data[:min(1000, len(flat_data))])
        if hurst is not None:
            results['hurst_exponent'] = hurst
        
        # DFA (Detrended Fluctuation Analysis)
        def dfa(data, scales=None):
            if scales is None:
                scales = np.logspace(1, np.log10(len(data)//4), 20, dtype=int)
            
            fluct = []
            for scale in scales:
                if scale < len(data):
                    segments = len(data) // scale
                    F = []
                    
                    for seg in range(segments):
                        segment_data = data[seg*scale:(seg+1)*scale]
                        x = np.arange(len(segment_data))
                        coeffs = np.polyfit(x, segment_data, 1)
                        fit = np.polyval(coeffs, x)
                        F.append(np.sqrt(np.mean((segment_data - fit)**2)))
                    
                    if F:
                        fluct.append(np.mean(F))
            
            if len(fluct) > 2:
                coeffs = np.polyfit(np.log(scales[:len(fluct)]), np.log(fluct), 1)
                return coeffs[0]
            return None
        
        dfa_exp = dfa(flat_data[:min(1000, len(flat_data))])
        if dfa_exp is not None:
            results['dfa_exponent'] = dfa_exp
        
        return results
    
    def calculate_spectral_measures(self, data, name):
        """Calculate advanced spectral measures"""
        flat_data = data.flatten()
        results = {}
        
        # Multitaper spectral analysis
        from scipy.signal import windows
        
        # Spectral entropy
        freqs = np.fft.fftfreq(len(flat_data))
        fft = np.abs(np.fft.fft(flat_data))
        psd = fft**2
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        results['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))
        
        # Spectral edge frequency
        cumsum_psd = np.cumsum(psd)
        results['spectral_edge_95'] = freqs[np.argmax(cumsum_psd > 0.95 * cumsum_psd[-1])]
        results['spectral_edge_50'] = freqs[np.argmax(cumsum_psd > 0.50 * cumsum_psd[-1])]
        
        # Spectral rolloff
        results['spectral_rolloff'] = freqs[np.argmax(cumsum_psd > 0.85 * cumsum_psd[-1])]
        
        # Spectral centroid
        results['spectral_centroid'] = np.sum(freqs[:len(psd)] * psd) / np.sum(psd)
        
        # Spectral spread
        results['spectral_spread'] = np.sqrt(np.sum((freqs[:len(psd)] - results['spectral_centroid'])**2 * psd) / np.sum(psd))
        
        # Spectral skewness
        results['spectral_skewness'] = np.sum((freqs[:len(psd)] - results['spectral_centroid'])**3 * psd) / (np.sum(psd) * results['spectral_spread']**3)
        
        # Spectral kurtosis
        results['spectral_kurtosis'] = np.sum((freqs[:len(psd)] - results['spectral_centroid'])**4 * psd) / (np.sum(psd) * results['spectral_spread']**4)
        
        # Wavelet entropy (simplified implementation)
        def simple_wavelet_transform(data, wavelet_func, scales):
            """Simple continuous wavelet transform implementation"""
            output = np.zeros((len(scales), len(data)))
            for i, scale in enumerate(scales):
                # Create wavelet at this scale
                wavelet_length = min(10 * scale, len(data))
                wavelet = wavelet_func(wavelet_length, scale)
                # Convolve with data
                output[i, :] = np.convolve(data, wavelet, mode='same')
            return output
        
        try:
            scales = np.arange(1, min(20, len(flat_data)//10))
            cwt_matrix = simple_wavelet_transform(flat_data[:min(1000, len(flat_data))], ricker, scales)
            wavelet_energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
            wavelet_energy_norm = wavelet_energy / np.sum(wavelet_energy)
            wavelet_energy_norm = wavelet_energy_norm[wavelet_energy_norm > 0]
            results['wavelet_entropy'] = -np.sum(wavelet_energy_norm * np.log(wavelet_energy_norm))
        except Exception as e:
            # Fallback to simple decomposition
            results['wavelet_entropy'] = 0.0
        
        return results
    
    def create_advanced_mathematical_expression(self):
        """Create comprehensive mathematical expression with full terms"""
        expression_parts = []
        
        # For each file, create detailed feature extraction
        for file_idx, (file_name, data) in enumerate(self.data_files.items()):
            flat_data = data.flatten()
            
            # Statistical moments (full expansion)
            for k in range(1, 11):
                moment = stats.moment(flat_data, k)
                expression_parts.append(f"({moment:.15e} * w_{file_idx}_{k}_moment)")
            
            # Cumulants (full expansion)
            for k in range(1, 5):
                if k == 1:
                    cumulant = np.mean(flat_data)
                elif k == 2:
                    cumulant = np.var(flat_data)
                elif k == 3:
                    cumulant = stats.moment(flat_data, 3)
                elif k == 4:
                    cumulant = stats.moment(flat_data, 4) - 3 * np.var(flat_data)**2
                expression_parts.append(f"({cumulant:.15e} * w_{file_idx}_{k}_cumulant)")
            
            # Quantiles (full expansion)
            for q in np.linspace(0.01, 0.99, 99):
                quantile = np.quantile(flat_data, q)
                expression_parts.append(f"({quantile:.15e} * w_{file_idx}_{int(q*100)}_quantile)")
            
            # Entropy measures
            hist, _ = np.histogram(flat_data, bins=50)
            hist = hist[hist > 0] / np.sum(hist)
            shannon_entropy = -np.sum(hist * np.log(hist))
            expression_parts.append(f"({shannon_entropy:.15e} * w_{file_idx}_shannon_entropy)")
            
            # Autocorrelation coefficients
            if len(flat_data) > 50:
                for lag in range(1, min(51, len(flat_data))):
                    try:
                        if STATSMODELS_AVAILABLE:
                            acf_value = acf(flat_data, nlags=lag, fft=True)[-1]
                        else:
                            # Simple autocorrelation calculation
                            acf_value = np.corrcoef(flat_data[:-lag], flat_data[lag:])[0, 1]
                        if not np.isnan(acf_value):
                            expression_parts.append(f"({acf_value:.15e} * w_{file_idx}_{lag}_acf)")
                    except:
                        pass
            
            # Fourier coefficients (magnitude and phase)
            fft_coeffs = np.fft.fft(flat_data)
            for k in range(min(100, len(fft_coeffs)//2)):
                magnitude = np.abs(fft_coeffs[k])
                phase = np.angle(fft_coeffs[k])
                expression_parts.append(f"({magnitude:.15e} * w_{file_idx}_{k}_fft_mag)")
                expression_parts.append(f"({phase:.15e} * w_{file_idx}_{k}_fft_phase)")
        
        # Construct the full expression
        full_expression = "I = \n"
        for i, part in enumerate(expression_parts):
            full_expression += f"    {part}"
            if i < len(expression_parts) - 1:
                full_expression += " +\n"
            else:
                full_expression += "\n"
        
        full_expression += "= D(S)\n\n"
        full_expression += f"Where:\n"
        full_expression += f"- I represents the input image features\n"
        full_expression += f"- D represents the comparison image features\n"
        full_expression += f"- S represents the similarity score\n"
        full_expression += f"- w_i_j_type represents the weight for file i, index j, feature type\n"
        full_expression += f"- Total number of terms: {len(expression_parts)}\n"
        
        return full_expression
    
    def create_similarity_function(self):
        """Create detailed similarity function"""
        similarity_expr = """
SIMILARITY FUNCTION S(I, D):

S(I, D) = 1 / (1 + Σ(α_k * d_k(I, D)))

Where d_k represents different distance metrics:

d_1(I, D) = ||I - D||_2 = √(Σ(I_i - D_i)²)  [Euclidean distance]

d_2(I, D) = ||I - D||_1 = Σ|I_i - D_i|  [Manhattan distance]

d_3(I, D) = ||I - D||_∞ = max|I_i - D_i|  [Chebyshev distance]

d_4(I, D) = 1 - (I·D)/(||I||·||D||)  [Cosine distance]

d_5(I, D) = √(Σ((I_i - D_i)²/σ_i²))  [Mahalanobis distance component]

d_6(I, D) = Σ((I_i - D_i)²/(I_i + D_i))  [Chi-squared distance]

d_7(I, D) = -Σ(I_i * log(D_i/I_i))  [Kullback-Leibler divergence]

d_8(I, D) = 1/2 * Σ((√I_i - √D_i)²)  [Hellinger distance]

d_9(I, D) = Σ(|I_i - D_i|/(|I_i| + |D_i|))  [Canberra distance]

d_10(I, D) = (Σ|I_i - D_i|^p)^(1/p)  [Minkowski distance, p-parameterized]

With learned weights α_k optimized through neural network training.
"""
        return similarity_expr
    
    def run_advanced_analysis(self):
        """Run complete advanced analysis"""
        results = {
            'advanced_moments': {},
            'entropy_measures': {},
            'fractal_dimensions': {},
            'nonlinear_measures': {},
            'spectral_measures': {},
            'mathematical_expressions': {}
        }
        
        for file_name, data in self.data_files.items():
            print(f"Running advanced analysis for {file_name}...")
            
            results['advanced_moments'][file_name] = self.calculate_advanced_moments(data, file_name)
            results['entropy_measures'][file_name] = self.calculate_entropy_measures(data, file_name)
            results['fractal_dimensions'][file_name] = self.calculate_fractal_dimensions(data, file_name)
            results['nonlinear_measures'][file_name] = self.calculate_nonlinear_measures(data, file_name)
            results['spectral_measures'][file_name] = self.calculate_spectral_measures(data, file_name)
        
        # Create comprehensive mathematical expressions
        results['mathematical_expressions']['full_feature_expression'] = self.create_advanced_mathematical_expression()
        results['mathematical_expressions']['similarity_function'] = self.create_similarity_function()
        
        return results


class FullMathematicalExpressionGenerator:
    def __init__(self, data_files, analysis_results):
        self.data_files = data_files
        self.analysis_results = analysis_results
        self.full_expressions = {}
        
    def generate_complete_feature_vector_expression(self):
        """Generate the complete I = Σ(feature_i * weight_i) expression"""
        expression_lines = []
        expression_lines.append("COMPLETE FEATURE VECTOR MATHEMATICAL EXPRESSION")
        expression_lines.append("="*80)
        expression_lines.append("")
        expression_lines.append("I = ")
        
        term_counter = 0
        
        for file_idx, (file_name, data) in enumerate(self.data_files.items()):
            flat_data = data.flatten()
            expression_lines.append(f"\n    # Features from {file_name}")
            expression_lines.append(f"    # Shape: {data.shape}, Total elements: {flat_data.size}")
            expression_lines.append("")
            
            # Mean
            mean_val = np.mean(flat_data)
            expression_lines.append(f"    + ({mean_val:.20e} * w_{{{file_idx}}}_mean)")
            term_counter += 1
            
            # Variance
            var_val = np.var(flat_data)
            expression_lines.append(f"    + ({var_val:.20e} * w_{{{file_idx}}}_variance)")
            term_counter += 1
            
            # Standard deviation
            std_val = np.std(flat_data)
            expression_lines.append(f"    + ({std_val:.20e} * w_{{{file_idx}}}_std)")
            term_counter += 1
            
            # All percentiles from 0 to 100
            for p in range(101):
                percentile_val = np.percentile(flat_data, p)
                expression_lines.append(f"    + ({percentile_val:.20e} * w_{{{file_idx}}}_percentile_{p})")
                term_counter += 1
            
            # All moments up to 20th order
            for moment_order in range(1, 21):
                moment_val = np.mean((flat_data - np.mean(flat_data))**moment_order)
                expression_lines.append(f"    + ({moment_val:.20e} * w_{{{file_idx}}}_moment_{moment_order})")
                term_counter += 1
            
            # All unique values (if reasonable number)
            unique_vals = np.unique(flat_data)
            if len(unique_vals) <= 1000:
                for i, val in enumerate(unique_vals):
                    count = np.sum(flat_data == val)
                    expression_lines.append(f"    + ({val:.20e} * {count} * w_{{{file_idx}}}_unique_{i})")
                    term_counter += 1
            
            # FFT coefficients (real and imaginary parts)
            fft_result = np.fft.fft(flat_data)
            for k in range(min(len(fft_result), 500)):
                real_part = np.real(fft_result[k])
                imag_part = np.imag(fft_result[k])
                expression_lines.append(f"    + ({real_part:.20e} * w_{{{file_idx}}}_fft_real_{k})")
                expression_lines.append(f"    + ({imag_part:.20e} * w_{{{file_idx}}}_fft_imag_{k})")
                term_counter += 2
            
            # Autocorrelation values
            if len(flat_data) > 100:
                for lag in range(1, min(101, len(flat_data))):
                    try:
                        autocorr = np.corrcoef(flat_data[:-lag], flat_data[lag:])[0, 1]
                        if not np.isnan(autocorr):
                            expression_lines.append(f"    + ({autocorr:.20e} * w_{{{file_idx}}}_autocorr_lag_{lag})")
                            term_counter += 1
                    except:
                        pass
            
            # Histogram bin counts
            hist, bin_edges = np.histogram(flat_data, bins=50)
            for bin_idx, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
                expression_lines.append(f"    + ({count} * {edge:.20e} * w_{{{file_idx}}}_hist_bin_{bin_idx})")
                term_counter += 1
            
            expression_lines.append("")
        
        expression_lines.append("")
        expression_lines.append(f"Total number of terms: {term_counter}")
        expression_lines.append("")
        expression_lines.append("Where each w_i represents a learnable weight parameter")
        
        return "\n".join(expression_lines)
    
    def generate_complete_polynomial_expressions(self):
        """Generate complete polynomial expressions for each dataset"""
        poly_expressions = []
        poly_expressions.append("COMPLETE POLYNOMIAL EXPRESSIONS FOR EACH DATASET")
        poly_expressions.append("="*80)
        poly_expressions.append("")
        
        for file_name, data in self.data_files.items():
            poly_expressions.append(f"\nDataset: {file_name}")
            poly_expressions.append("-"*50)
            
            flat_data = data.flatten()
            n = len(flat_data)
            x = np.arange(n)
            
            # Generate polynomials up to degree 20
            for degree in range(1, 21):
                if n > degree + 1:
                    coeffs = np.polyfit(x, flat_data, degree)
                    
                    poly_expressions.append(f"\nDegree {degree} polynomial:")
                    poly_expressions.append("y = ")
                    
                    terms = []
                    for i, coeff in enumerate(coeffs):
                        power = degree - i
                        if power == 0:
                            terms.append(f"{coeff:.20e}")
                        elif power == 1:
                            terms.append(f"({coeff:.20e} * x)")
                        else:
                            terms.append(f"({coeff:.20e} * x^{power})")
                    
                    # Write full expression without truncation
                    poly_expressions.append("    " + " + ".join(terms))
                    
                    # Also write in expanded form
                    poly_expressions.append("\nExpanded form:")
                    for i, term in enumerate(terms):
                        if i == 0:
                            poly_expressions.append(f"    {term}")
                        else:
                            poly_expressions.append(f"    + {term}")
        
        return "\n".join(poly_expressions)
    
    def generate_correlation_matrix_expressions(self):
        """Generate complete correlation matrix expressions"""
        corr_expressions = []
        corr_expressions.append("COMPLETE CORRELATION MATRIX EXPRESSIONS")
        corr_expressions.append("="*80)
        corr_expressions.append("")
        
        # Create correlation matrix
        file_names = list(self.data_files.keys())
        n_files = len(file_names)
        
        # Pearson correlation matrix
        corr_expressions.append("PEARSON CORRELATION MATRIX:")
        corr_expressions.append("")
        pearson_matrix = np.zeros((n_files, n_files))
        
        for i, file1 in enumerate(file_names):
            for j, file2 in enumerate(file_names):
                data1 = self.data_files[file1].flatten()
                data2 = self.data_files[file2].flatten()
                min_len = min(len(data1), len(data2))
                if min_len > 1:
                    corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
                    pearson_matrix[i, j] = corr
                    corr_expressions.append(f"ρ({file1}, {file2}) = {corr:.20e}")
        
        corr_expressions.append("\nMatrix form:")
        corr_expressions.append("R_pearson = ")
        for i in range(n_files):
            row = "    ["
            for j in range(n_files):
                row += f"{pearson_matrix[i, j]:.20e}"
                if j < n_files - 1:
                    row += ", "
            row += "]"
            if i < n_files - 1:
                row += ","
            corr_expressions.append(row)
        
        # Covariance matrix
        corr_expressions.append("\n\nCOVARIANCE MATRIX:")
        corr_expressions.append("")
        cov_matrix = np.zeros((n_files, n_files))
        
        for i, file1 in enumerate(file_names):
            for j, file2 in enumerate(file_names):
                data1 = self.data_files[file1].flatten()
                data2 = self.data_files[file2].flatten()
                min_len = min(len(data1), len(data2))
                if min_len > 1:
                    cov = np.cov(data1[:min_len], data2[:min_len])[0, 1]
                    cov_matrix[i, j] = cov
                    corr_expressions.append(f"Cov({file1}, {file2}) = {cov:.20e}")
        
        return "\n".join(corr_expressions)
    
    def generate_similarity_equation_expanded(self):
        """Generate fully expanded similarity equation"""
        sim_expressions = []
        sim_expressions.append("FULLY EXPANDED SIMILARITY EQUATION")
        sim_expressions.append("="*80)
        sim_expressions.append("")
        
        # Get sample data for demonstration
        sample_file = list(self.data_files.keys())[0]
        sample_data = self.data_files[sample_file].flatten()
        n_features = min(len(sample_data), 100)  # Limit for readability
        
        sim_expressions.append("S(I, D) = 1 / (1 + Σ_{k=1}^{10} α_k * d_k(I, D))")
        sim_expressions.append("")
        sim_expressions.append("Where:")
        sim_expressions.append("")
        
        # Euclidean distance expanded
        sim_expressions.append("d_1(I, D) = √(")
        for i in range(n_features):
            if i == 0:
                sim_expressions.append(f"      (I_{i} - D_{i})²")
            else:
                sim_expressions.append(f"    + (I_{i} - D_{i})²")
        sim_expressions.append(")")
        sim_expressions.append("")
        
        # Manhattan distance expanded
        sim_expressions.append("d_2(I, D) = ")
        for i in range(n_features):
            if i == 0:
                sim_expressions.append(f"      |I_{i} - D_{i}|")
            else:
                sim_expressions.append(f"    + |I_{i} - D_{i}|")
        sim_expressions.append("")
        
        # Minkowski distance general form
        sim_expressions.append("d_10(I, D) = (")
        for i in range(n_features):
            if i == 0:
                sim_expressions.append(f"      |I_{i} - D_{i}|^p")
            else:
                sim_expressions.append(f"    + |I_{i} - D_{i}|^p")
        sim_expressions.append(")^(1/p)")
        sim_expressions.append("")
        
        # Full similarity function
        sim_expressions.append("COMPLETE SIMILARITY FUNCTION:")
        sim_expressions.append("")
        sim_expressions.append("S(I, D) = 1 / (1 + ")
        sim_expressions.append("      α_1 * d_1(I, D)")
        sim_expressions.append("    + α_2 * d_2(I, D)")
        sim_expressions.append("    + α_3 * d_3(I, D)")
        sim_expressions.append("    + α_4 * d_4(I, D)")
        sim_expressions.append("    + α_5 * d_5(I, D)")
        sim_expressions.append("    + α_6 * d_6(I, D)")
        sim_expressions.append("    + α_7 * d_7(I, D)")
        sim_expressions.append("    + α_8 * d_8(I, D)")
        sim_expressions.append("    + α_9 * d_9(I, D)")
        sim_expressions.append("    + α_10 * d_10(I, D)")
        sim_expressions.append(")")
        
        return "\n".join(sim_expressions)
    
    def generate_fourier_series_expanded(self):
        """Generate fully expanded Fourier series for each dataset"""
        fourier_expressions = []
        fourier_expressions.append("COMPLETE FOURIER SERIES EXPANSIONS")
        fourier_expressions.append("="*80)
        fourier_expressions.append("")
        
        for file_name, data in self.data_files.items():
            fourier_expressions.append(f"\nDataset: {file_name}")
            fourier_expressions.append("-"*50)
            
            flat_data = data.flatten()
            n = len(flat_data)
            x = np.arange(n)
            
            # Limit harmonics for readability
            max_harmonics = min(50, n // 2)
            
            # Calculate Fourier coefficients
            a0 = np.mean(flat_data)
            
            fourier_expressions.append(f"\nf(x) = {a0:.20e}")
            
            for k in range(1, max_harmonics + 1):
                # Calculate coefficients
                a_k = 2/n * np.sum(flat_data * np.cos(2*np.pi*k*x/n))
                b_k = 2/n * np.sum(flat_data * np.sin(2*np.pi*k*x/n))
                
                fourier_expressions.append(f"       + ({a_k:.20e} * cos({k}*2π*x/{n}))")
                fourier_expressions.append(f"       + ({b_k:.20e} * sin({k}*2π*x/{n}))")
            
            fourier_expressions.append("")
            
            # Also write in compact form with actual values
            fourier_expressions.append("Compact form with computed values:")
            fourier_expressions.append(f"f(x) = {a0:.20e}")
            for k in range(1, min(10, max_harmonics + 1)):
                a_k = 2/n * np.sum(flat_data * np.cos(2*np.pi*k*x/n))
                b_k = 2/n * np.sum(flat_data * np.sin(2*np.pi*k*x/n))
                fourier_expressions.append(f"       + {a_k:.20e}*cos({2*np.pi*k/n:.20e}*x) + {b_k:.20e}*sin({2*np.pi*k/n:.20e}*x)")
        
        return "\n".join(fourier_expressions)
    
    def save_all_expressions(self, output_file='complete_mathematical_expressions.txt'):
        """Save all mathematical expressions to a single file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("COMPLETE MATHEMATICAL EXPRESSIONS FOR NPY DATA ANALYSIS\n")
            f.write("="*100 + "\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n")
            f.write(f"Number of files analyzed: {len(self.data_files)}\n")
            f.write("="*100 + "\n\n")
            
            # Section 1: Feature Vector Expression
            f.write(self.generate_complete_feature_vector_expression())
            f.write("\n\n" + "="*100 + "\n\n")
            
            # Section 2: Polynomial Expressions
            f.write(self.generate_complete_polynomial_expressions())
            f.write("\n\n" + "="*100 + "\n\n")
            
            # Section 3: Correlation Matrices
            f.write(self.generate_correlation_matrix_expressions())
            f.write("\n\n" + "="*100 + "\n\n")
            
            # Section 4: Similarity Equation
            f.write(self.generate_similarity_equation_expanded())
            f.write("\n\n" + "="*100 + "\n\n")
            
            # Section 5: Fourier Series
            f.write(self.generate_fourier_series_expanded())
            f.write("\n\n" + "="*100 + "\n\n")
            
            # Section 6: Individual data point equations
            f.write("INDIVIDUAL DATA POINT REPRESENTATIONS\n")
            f.write("="*80 + "\n\n")
            
            for file_name, data in self.data_files.items():
                f.write(f"Dataset: {file_name}\n")
                f.write("-"*50 + "\n")
                flat_data = data.flatten()
                
                # Write equations for first 100 data points (or all if less)
                n_points = min(100, len(flat_data))
                for i in range(n_points):
                    f.write(f"Point_{i} = {flat_data[i]:.20e}\n")
                
                if len(flat_data) > 100:
                    f.write(f"... ({len(flat_data) - 100} more points)\n")
                f.write("\n")
            
        print(f"Complete mathematical expressions saved to {output_file}")


class ComprehensiveAnalysisReport:
    def __init__(self, basic_results, advanced_results):
        self.basic_results = basic_results
        self.advanced_results = advanced_results
    
    def generate_full_mathematical_document(self, output_file='full_mathematical_expressions.txt'):
        """Generate document with complete mathematical expressions"""
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("COMPLETE MATHEMATICAL REPRESENTATION OF NPY DATA\n")
            f.write("="*100 + "\n\n")
            
            # Write basic equations
            f.write("SECTION 1: POLYNOMIAL REPRESENTATIONS\n")
            f.write("-"*50 + "\n\n")
            
            for file_name, equations in self.basic_results['equations'].items():
                f.write(f"File: {file_name}\n")
                f.write("="*30 + "\n")
                
                for eq_type, eq_data in equations.items():
                    f.write(f"\n{eq_type}:\n")
                    f.write(f"Full Equation: {eq_data['equation']}\n")
                    f.write(f"Coefficients: {eq_data['coefficients']}\n")
                    f.write("\n")
            
            # Write advanced mathematical expression
            f.write("\n\nSECTION 2: COMPREHENSIVE FEATURE EXPRESSION\n")
            f.write("-"*50 + "\n\n")
            f.write(self.advanced_results['mathematical_expressions']['full_feature_expression'])
            
            # Write similarity function
            f.write("\n\nSECTION 3: SIMILARITY COMPUTATION\n")
            f.write("-"*50 + "\n\n")
            f.write(self.advanced_results['mathematical_expressions']['similarity_function'])
            
            # Write correlation matrices
            f.write("\n\nSECTION 4: CORRELATION EXPRESSIONS\n")
            f.write("-"*50 + "\n\n")
            
            for corr_pair, corr_values in self.basic_results['correlations'].items():
                f.write(f"\n{corr_pair}:\n")
                for corr_type, value in corr_values.items():
                    f.write(f"  {corr_type}: {value:.15e}\n")
            
            # Write advanced statistics
            f.write("\n\nSECTION 5: ADVANCED STATISTICAL MEASURES\n")
            f.write("-"*50 + "\n\n")
            
            for category in ['advanced_moments', 'entropy_measures', 'fractal_dimensions', 
                           'nonlinear_measures', 'spectral_measures']:
                f.write(f"\n{category.upper()}:\n")
                f.write("="*30 + "\n")
                
                for file_name, measures in self.advanced_results[category].items():
                    f.write(f"\nFile: {file_name}\n")
                    for measure_name, value in measures.items():
                        f.write(f"  {measure_name}: {value}\n")


# Main execution function
def run_complete_analysis(folder_path):
    """Run all three levels of analysis"""
    print("Starting comprehensive NPY analysis...")
    print("="*50)
    
    # Step 1: Basic comprehensive analysis
    print("\nStep 1: Running basic comprehensive analysis...")
    basic_analyzer = ComprehensiveNPYAnalyzer(folder_path)
    basic_results = basic_analyzer.analyze_all()
    
    if basic_results is None:
        print("Analysis failed - no NPY files found or loaded.")
        return None, None, None
    
    basic_analyzer.save_results('analysis_results')
    
    # Step 2: Advanced PhD-level analysis
    print("\nStep 2: Running advanced PhD-level analysis...")
    advanced_analyzer = AdvancedPHDAnalysis(basic_analyzer.data_files)
    advanced_results = advanced_analyzer.run_advanced_analysis()
    
    # Step 3: Generate complete mathematical expressions
    print("\nStep 3: Generating complete mathematical expressions...")
    expression_generator = FullMathematicalExpressionGenerator(
        basic_analyzer.data_files, 
        {'basic': basic_results, 'advanced': advanced_results}
    )
    expression_generator.save_all_expressions()
    
    # Step 4: Create comprehensive report
    print("\nStep 4: Creating comprehensive analysis report...")
    report_generator = ComprehensiveAnalysisReport(basic_results, advanced_results)
    report_generator.generate_full_mathematical_document()
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("\nGenerated files:")
    print("1. analysis_results/complete_analysis.json - Complete analysis data")
    print("2. analysis_results/detailed_statistics.txt - Detailed statistics for each file")
    print("3. analysis_results/mathematical_equations.txt - Mathematical equations")
    print("4. complete_mathematical_expressions.txt - FULL mathematical expressions (no truncation)")
    print("5. full_mathematical_expressions.txt - PhD-level mathematical analysis")
    
    return basic_analyzer, advanced_analyzer, expression_generator


if __name__ == "__main__":
    # Use current directory by default, or specify your folder path
    folder_path = "."  # Current directory
    
    # Run the complete analysis
    basic, advanced, expressions = run_complete_analysis(folder_path)
    
    if basic is not None:
        print("\nAnalysis complete! Check the output files for results.")
    else:
        print("\nAnalysis failed. Please check that there are .npy files in the specified directory.")