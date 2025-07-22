import numpy as np
import os
import glob
from scipy import stats, linalg, optimize, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, ICA, FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
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
        npy_files = glob.glob(os.path.join(self.folder_path, "*.npy"))
        print(f"Found {len(npy_files)} .npy files")
        
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
            acf = np.correlate(flat_data - np.mean(flat_data), flat_data - np.mean(flat_data), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]
            stats_dict['autocorr_lag_1'] = float(acf[1]) if len(acf) > 1 else None
            stats_dict['autocorr_lag_5'] = float(acf[5]) if len(acf) > 5 else None
            stats_dict['autocorr_lag_10'] = float(acf[10]) if len(acf) > 10 else None
        
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

# Usage
if __name__ == "__main__":
    # Replace with your folder path
    folder_path = r"C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\defect-library-npyr"
    
    analyzer = ComprehensiveNPYAnalyzer(folder_path)
    results = analyzer.analyze_all()
    analyzer.save_results()
    
    print("\nAnalysis complete!")
