import numpy as np
import scipy.stats as stats
from scipy.special import gamma, digamma, polygamma
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import svd, eig, inv, pinv
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert, find_peaks, cwt, ricker
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import NMF, SparsePCA, KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import itertools
from datetime import datetime

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
        
        # Wavelet entropy
        scales = np.arange(1, min(128, len(flat_data)//2))
        cwt_matrix = cwt(flat_data[:min(1000, len(flat_data))], ricker, scales)
        wavelet_energy = np.sum(np.abs(cwt_matrix)**2, axis=1)
        wavelet_energy_norm = wavelet_energy / np.sum(wavelet_energy)
        wavelet_energy_norm = wavelet_energy_norm[wavelet_energy_norm > 0]
        results['wavelet_entropy'] = -np.sum(wavelet_energy_norm * np.log(wavelet_energy_norm))
        
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
                for lag in range(1, 51):
                    acf_value = acf(flat_data, nlags=lag, fft=True)[-1]
                    expression_parts.append(f"({acf_value:.15e} * w_{file_idx}_{lag}_acf)")
            
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

# Extension to main analyzer
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

if __name__ == "__main__":
    # This would be run after the basic analysis
    print("Run the basic analysis first, then use the results here")
