#!/usr/bin/env python3
"""
Comprehensive JSON Data Analyzer for Consensus Reports
This script performs exhaustive statistical analysis on all JSON files
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveJSONAnalyzer:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.data = []
        self.df = None
        self.statistics = {}
        self.correlations = {}
        self.equations = {}
        
    def load_all_json_files(self):
        """Load all JSON files from the directory"""
        print("Loading JSON files...")
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith('.json')]
        
        for file in json_files:
            try:
                with open(os.path.join(self.directory_path, file), 'r') as f:
                    data = json.load(f)
                    data['filename'] = file
                    self.data.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        print(f"Loaded {len(self.data)} JSON files")
        
    def prepare_dataframe(self):
        """Convert JSON data to a structured DataFrame"""
        print("Preparing DataFrame...")
        
        rows = []
        for item in self.data:
            row = {
                'filename': item['filename'],
                'center_x': item['center'][0] if 'center' in item else np.nan,
                'center_y': item['center'][1] if 'center' in item else np.nan,
                'core_radius': item.get('core_radius', np.nan),
                'cladding_radius': item.get('cladding_radius', np.nan),
                'num_valid_results': item.get('num_valid_results', np.nan),
            }
            
            # Add method accuracies
            if 'method_accuracies' in item:
                for method, accuracy in item['method_accuracies'].items():
                    row[f'accuracy_{method}'] = accuracy
            
            # Add contributing methods count
            if 'contributing_methods' in item:
                row['num_contributing_methods'] = len(item['contributing_methods'])
                for i, method in enumerate(item['contributing_methods']):
                    row[f'contributing_method_{i}'] = method
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        
        # Create derived features
        self.df['core_cladding_ratio'] = self.df['core_radius'] / self.df['cladding_radius']
        self.df['core_area'] = np.pi * self.df['core_radius']**2
        self.df['cladding_area'] = np.pi * self.df['cladding_radius']**2
        self.df['cladding_core_area_diff'] = self.df['cladding_area'] - self.df['core_area']
        self.df['center_distance_from_origin'] = np.sqrt(self.df['center_x']**2 + self.df['center_y']**2)
        self.df['aspect_ratio'] = self.df['center_x'] / self.df['center_y']
        
        print(f"DataFrame prepared with shape: {self.df.shape}")
        
    def calculate_basic_statistics(self):
        """Calculate basic statistical measurements"""
        print("Calculating basic statistics...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        self.statistics['basic'] = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                self.statistics['basic'][col] = {
                    'count': len(data),
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'mode': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else np.nan,
                    'std': np.std(data),
                    'variance': np.var(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'range': np.max(data) - np.min(data),
                    'q1': np.percentile(data, 25),
                    'q3': np.percentile(data, 75),
                    'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                    'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else np.nan,
                    'sum': np.sum(data),
                    'geometric_mean': stats.gmean(data[data > 0]) if any(data > 0) else np.nan,
                    'harmonic_mean': stats.hmean(data[data > 0]) if any(data > 0) else np.nan,
                }
    
    def calculate_advanced_statistics(self):
        """Calculate advanced statistical measurements"""
        print("Calculating advanced statistics...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        self.statistics['advanced'] = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 2:
                self.statistics['advanced'][col] = {
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'jarque_bera': stats.jarque_bera(data),
                    'shapiro_wilk': stats.shapiro(data) if len(data) <= 5000 else None,
                    'anderson_darling': stats.anderson(data),
                    'sem': stats.sem(data),
                    'mad': stats.median_abs_deviation(data),
                    'entropy': stats.entropy(pd.cut(data, bins=10).value_counts()),
                    'moment_2': stats.moment(data, 2),
                    'moment_3': stats.moment(data, 3),
                    'moment_4': stats.moment(data, 4),
                    'trimmed_mean_10': stats.trim_mean(data, 0.1),
                    'trimmed_mean_20': stats.trim_mean(data, 0.2),
                    'winsorized_mean_10': stats.mstats.winsorize(data, limits=[0.1, 0.1]).mean(),
                    'gini_coefficient': self._calculate_gini(data),
                    'coefficient_of_dispersion': self._calculate_cod(data),
                }
    
    def _calculate_gini(self, data):
        """Calculate Gini coefficient"""
        sorted_data = np.sort(data)
        n = len(data)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n
    
    def _calculate_cod(self, data):
        """Calculate coefficient of dispersion"""
        median = np.median(data)
        if median == 0:
            return np.nan
        return np.mean(np.abs(data - median)) / median
    
    def calculate_correlations(self):
        """Calculate all possible correlations"""
        print("Calculating correlations...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Pearson correlation
        self.correlations['pearson'] = self.df[numeric_cols].corr(method='pearson')
        
        # Spearman correlation
        self.correlations['spearman'] = self.df[numeric_cols].corr(method='spearman')
        
        # Kendall correlation
        self.correlations['kendall'] = self.df[numeric_cols].corr(method='kendall')
        
        # Partial correlations
        self.correlations['partial'] = {}
        
        # Distance correlation
        self.correlations['distance'] = {}
        
        # Mutual information
        self.correlations['mutual_info'] = {}
        
        # Calculate pairwise correlations with significance
        self.correlations['pairwise'] = {}
        for col1, col2 in combinations(numeric_cols, 2):
            data1 = self.df[col1].dropna()
            data2 = self.df[col2].dropna()
            
            # Find common indices
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) > 2:
                d1 = data1.loc[common_idx]
                d2 = data2.loc[common_idx]
                
                pearson_r, pearson_p = stats.pearsonr(d1, d2)
                spearman_r, spearman_p = stats.spearmanr(d1, d2)
                
                self.correlations['pairwise'][f"{col1}_vs_{col2}"] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_samples': len(common_idx),
                    'covariance': np.cov(d1, d2)[0, 1],
                }
    
    def fit_regression_models(self):
        """Fit various regression models"""
        print("Fitting regression models...")
        
        self.equations['regression'] = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # For each column as target
        for target_col in numeric_cols:
            target_data = self.df[target_col].dropna()
            if len(target_data) < 10:
                continue
                
            # Get features (all other numeric columns)
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Prepare data
            common_idx = target_data.index
            for col in feature_cols:
                common_idx = common_idx.intersection(self.df[col].dropna().index)
            
            if len(common_idx) < 10:
                continue
                
            X = self.df.loc[common_idx, feature_cols].values
            y = self.df.loc[common_idx, target_col].values
            
            # Remove columns with zero variance
            valid_features = []
            valid_X = []
            for i, col in enumerate(feature_cols):
                if np.std(X[:, i]) > 0:
                    valid_features.append(col)
                    valid_X.append(X[:, i])
            
            if len(valid_X) == 0:
                continue
                
            X = np.array(valid_X).T
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            self.equations['regression'][target_col] = {
                'features': valid_features,
                'linear': {
                    'coefficients': lr.coef_.tolist(),
                    'intercept': lr.intercept_,
                    'r2': r2_score(y, y_pred),
                    'rmse': np.sqrt(np.mean((y - y_pred)**2)),
                    'mae': np.mean(np.abs(y - y_pred)),
                    'equation': self._format_linear_equation(lr.coef_, lr.intercept_, valid_features),
                }
            }
            
            # Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, y)
            y_pred_ridge = ridge.predict(X)
            
            self.equations['regression'][target_col]['ridge'] = {
                'coefficients': ridge.coef_.tolist(),
                'intercept': ridge.intercept_,
                'r2': r2_score(y, y_pred_ridge),
                'alpha': 1.0,
            }
            
            # Lasso regression
            lasso = Lasso(alpha=0.1)
            lasso.fit(X, y)
            y_pred_lasso = lasso.predict(X)
            
            self.equations['regression'][target_col]['lasso'] = {
                'coefficients': lasso.coef_.tolist(),
                'intercept': lasso.intercept_,
                'r2': r2_score(y, y_pred_lasso),
                'alpha': 0.1,
            }
    
    def _format_linear_equation(self, coefficients, intercept, features):
        """Format linear equation as string"""
        equation = f"{intercept:.6f}"
        for coef, feature in zip(coefficients, features):
            if coef >= 0:
                equation += f" + {coef:.6f}*{feature}"
            else:
                equation += f" - {abs(coef):.6f}*{feature}"
        return equation
    
    def fit_nonlinear_models(self):
        """Fit nonlinear models to data"""
        print("Fitting nonlinear models...")
        
        self.equations['nonlinear'] = {}
        
        # Define nonlinear functions
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c
        
        def power_law(x, a, b, c):
            return a * (x ** b) + c
        
        def logarithmic(x, a, b, c):
            return a * np.log(b * x + 1) + c
        
        def sigmoid(x, a, b, c, d):
            return a / (1 + np.exp(-b * (x - c))) + d
        
        def polynomial_2(x, a, b, c):
            return a * x**2 + b * x + c
        
        def polynomial_3(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Fit models for pairs of variables
        for col1, col2 in combinations(numeric_cols, 2):
            data1 = self.df[col1].dropna()
            data2 = self.df[col2].dropna()
            
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) < 10:
                continue
                
            x = data1.loc[common_idx].values
            y = data2.loc[common_idx].values
            
            # Remove infinite or nan values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) < 10:
                continue
            
            pair_key = f"{col1}_vs_{col2}"
            self.equations['nonlinear'][pair_key] = {}
            
            # Try different models
            models = {
                'exponential': (exponential, 3),
                'power_law': (power_law, 3),
                'logarithmic': (logarithmic, 3),
                'sigmoid': (sigmoid, 4),
                'polynomial_2': (polynomial_2, 3),
                'polynomial_3': (polynomial_3, 4),
            }
            
            for model_name, (func, n_params) in models.items():
                try:
                    # Initial guess
                    if model_name == 'exponential':
                        p0 = [1, 0.01, np.mean(y)]
                    elif model_name == 'power_law':
                        p0 = [1, 1, 0]
                    elif model_name == 'logarithmic':
                        p0 = [1, 1, 0]
                    elif model_name == 'sigmoid':
                        p0 = [np.max(y) - np.min(y), 1, np.mean(x), np.min(y)]
                    elif model_name == 'polynomial_2':
                        p0 = [1, 1, 0]
                    else:  # polynomial_3
                        p0 = [1, 1, 1, 0]
                    
                    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=5000)
                    y_pred = func(x, *popt)
                    
                    self.equations['nonlinear'][pair_key][model_name] = {
                        'parameters': popt.tolist(),
                        'r2': r2_score(y, y_pred),
                        'rmse': np.sqrt(np.mean((y - y_pred)**2)),
                        'equation': self._format_nonlinear_equation(model_name, popt),
                    }
                except:
                    pass
    
    def _format_nonlinear_equation(self, model_name, params):
        """Format nonlinear equation as string"""
        if model_name == 'exponential':
            return f"{params[0]:.6f} * exp({params[1]:.6f} * x) + {params[2]:.6f}"
        elif model_name == 'power_law':
            return f"{params[0]:.6f} * x^{params[1]:.6f} + {params[2]:.6f}"
        elif model_name == 'logarithmic':
            return f"{params[0]:.6f} * log({params[1]:.6f} * x + 1) + {params[2]:.6f}"
        elif model_name == 'sigmoid':
            return f"{params[0]:.6f} / (1 + exp(-{params[1]:.6f} * (x - {params[2]:.6f}))) + {params[3]:.6f}"
        elif model_name == 'polynomial_2':
            return f"{params[0]:.6f} * x^2 + {params[1]:.6f} * x + {params[2]:.6f}"
        elif model_name == 'polynomial_3':
            return f"{params[0]:.6f} * x^3 + {params[1]:.6f} * x^2 + {params[2]:.6f} * x + {params[3]:.6f}"
    
    def perform_pca_analysis(self):
        """Perform PCA analysis"""
        print("Performing PCA analysis...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        data = self.df[numeric_cols].dropna()
        
        if len(data) < 10:
            return
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        self.statistics['pca'] = {
            'explained_variance': pca.explained_variance_.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1,
            'feature_importance': {
                col: pca.components_[0][i] for i, col in enumerate(numeric_cols)
            }
        }
    
    def calculate_distribution_fits(self):
        """Fit various probability distributions to data"""
        print("Fitting probability distributions...")
        
        self.statistics['distributions'] = {}
        
        distributions = [
            ('normal', stats.norm),
            ('lognormal', stats.lognorm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('beta', stats.beta),
            ('weibull', stats.weibull_min),
            ('uniform', stats.uniform),
            ('pareto', stats.pareto),
            ('chi2', stats.chi2),
        ]
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 20:
                continue
                
            self.statistics['distributions'][col] = {}
            
            for dist_name, dist in distributions:
                try:
                    params = dist.fit(data)
                    
                    # Kolmogorov-Smirnov test
                    ks_statistic, ks_pvalue = stats.kstest(data, lambda x: dist.cdf(x, *params))
                    
                    self.statistics['distributions'][col][dist_name] = {
                        'parameters': params,
                        'ks_statistic': ks_statistic,
                        'ks_pvalue': ks_pvalue,
                        'aic': 2 * len(params) - 2 * np.sum(dist.logpdf(data, *params)),
                        'bic': len(params) * np.log(len(data)) - 2 * np.sum(dist.logpdf(data, *params)),
                    }
                except:
                    pass
    
    def create_master_equation(self):
        """Create master equation representing all data"""
        print("Creating master equation...")
        
        # Get all numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data for master equation
        data_matrix = self.df[numeric_cols].fillna(0).values
        n_samples, n_features = data_matrix.shape
        
        # Normalize data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_matrix)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Euclidean distance
                euclidean_dist = np.sqrt(np.sum((normalized_data[i] - normalized_data[j])**2))
                
                # Manhattan distance
                manhattan_dist = np.sum(np.abs(normalized_data[i] - normalized_data[j]))
                
                # Cosine similarity
                cosine_sim = np.dot(normalized_data[i], normalized_data[j]) / (
                    np.linalg.norm(normalized_data[i]) * np.linalg.norm(normalized_data[j]) + 1e-10
                )
                
                # Combined similarity
                similarity = np.exp(-euclidean_dist) * (1 + cosine_sim) / 2
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Create master equation coefficients
        coefficients = {}
        for i, col in enumerate(numeric_cols):
            # Calculate feature importance based on variance and correlations
            variance_weight = np.var(data_matrix[:, i])
            correlation_weight = np.mean(np.abs(self.correlations['pearson'][col].values))
            
            coefficients[col] = {
                'weight': variance_weight * correlation_weight,
                'mean': np.mean(data_matrix[:, i]),
                'std': np.std(data_matrix[:, i]),
                'scaling_factor': scaler.scale_[i],
                'index': i,
            }
        
        self.equations['master'] = {
            'features': numeric_cols,
            'coefficients': coefficients,
            'similarity_matrix_stats': {
                'mean': np.mean(similarity_matrix),
                'std': np.std(similarity_matrix),
                'min': np.min(similarity_matrix),
                'max': np.max(similarity_matrix),
            },
            'equation': self._format_master_equation(coefficients, numeric_cols),
        }
    
    def _format_master_equation(self, coefficients, features):
        """Format master equation"""
        equation = "S(I, D) = exp(-sqrt("
        
        terms = []
        for feature in features:
            coef = coefficients[feature]
            weight = coef['weight']
            if weight > 0:
                terms.append(f"{weight:.6f}*(I_{feature} - D_{feature})^2")
        
        equation += " + ".join(terms[:5])  # Show first 5 terms
        equation += " + ...)) * (1 + cos_sim(I, D)) / 2"
        
        # Full equation
        full_equation = "S(I, D) = exp(-sqrt("
        full_equation += " + ".join([f"{coefficients[f]['weight']:.6f}*(I_{f} - D_{f})^2" for f in features])
        full_equation += ")) * (1 + cos_sim(I, D)) / 2"
        
        return {'abbreviated': equation, 'full': full_equation}
    
    def save_results(self):
        """Save all results to files"""
        print("Saving results...")
        
        # Save basic statistics
        with open('basic_statistics.json', 'w') as f:
            json.dump(self.statistics['basic'], f, indent=2)
        
        # Save advanced statistics
        with open('advanced_statistics.json', 'w') as f:
            json.dump(self.statistics['advanced'], f, indent=2, default=str)
        
        # Save correlations
        with open('correlations.json', 'w') as f:
            correlations_serializable = {
                'pearson': self.correlations['pearson'].to_dict(),
                'spearman': self.correlations['spearman'].to_dict(),
                'kendall': self.correlations['kendall'].to_dict(),
                'pairwise': self.correlations['pairwise'],
            }
            json.dump(correlations_serializable, f, indent=2)
        
        # Save equations
        with open('equations.json', 'w') as f:
            json.dump(self.equations, f, indent=2)
        
        # Create markdown report
        self.create_markdown_report()
        
        # Create advanced report
        self.create_advanced_report()
        
        # Create full equations document
        self.create_full_equations_document()
    
    def create_markdown_report(self, filename='statistical_analysis_report.md'):
        """Create comprehensive markdown report"""
        with open(filename, 'w') as f:
            f.write("# Comprehensive Statistical Analysis Report\n\n")
            
            f.write(f"## Overview\n")
            f.write(f"- Total files analyzed: {len(self.data)}\n")
            f.write(f"- Total features: {len(self.df.columns)}\n")
            f.write(f"- Numeric features: {len(self.df.select_dtypes(include=[np.number]).columns)}\n\n")
            
            f.write("## Basic Statistics Summary\n\n")
            for col, stats in self.statistics['basic'].items():
                f.write(f"### {col}\n")
                for stat, value in stats.items():
                    f.write(f"- {stat}: {value}\n")
                f.write("\n")
            
            f.write("## Correlation Analysis\n\n")
            f.write("### Top Correlations\n")
            
            # Get top correlations
            corr_matrix = self.correlations['pearson']
            upper_tri = np.triu(corr_matrix.values, k=1)
            high_corr_indices = np.where(np.abs(upper_tri) > 0.5)
            
            for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                f.write(f"- {col1} vs {col2}: {corr_value:.4f}\n")
            
            f.write("\n## Regression Models\n\n")
            for target, models in self.equations.get('regression', {}).items():
                f.write(f"### Target: {target}\n")
                if 'linear' in models:
                    f.write(f"- Linear R²: {models['linear']['r2']:.4f}\n")
                    f.write(f"- Equation: {models['linear']['equation']}\n\n")
            
            f.write("## Master Equation\n\n")
            if 'master' in self.equations:
                f.write(f"```\n{self.equations['master']['equation']['abbreviated']}\n```\n\n")
    
    def create_advanced_report(self, filename='advanced_statistical_analysis.md'):
        """Create PhD-level advanced statistical report"""
        with open(filename, 'w') as f:
            f.write("# Advanced Statistical Analysis Report\n\n")
            
            f.write("## Advanced Statistical Measurements\n\n")
            
            for col, stats in self.statistics.get('advanced', {}).items():
                f.write(f"### {col}\n")
                f.write(f"#### Distribution Properties\n")
                f.write(f"- Skewness: {stats.get('skewness', 'N/A')}\n")
                f.write(f"- Kurtosis: {stats.get('kurtosis', 'N/A')}\n")
                f.write(f"- Jarque-Bera statistic: {stats.get('jarque_bera', 'N/A')}\n")
                
                f.write(f"#### Robust Statistics\n")
                f.write(f"- MAD: {stats.get('mad', 'N/A')}\n")
                f.write(f"- Trimmed Mean (10%): {stats.get('trimmed_mean_10', 'N/A')}\n")
                f.write(f"- Winsorized Mean (10%): {stats.get('winsorized_mean_10', 'N/A')}\n")
                
                f.write(f"#### Higher Moments\n")
                f.write(f"- 2nd Moment: {stats.get('moment_2', 'N/A')}\n")
                f.write(f"- 3rd Moment: {stats.get('moment_3', 'N/A')}\n")
                f.write(f"- 4th Moment: {stats.get('moment_4', 'N/A')}\n")
                
                f.write(f"#### Information Theory\n")
                f.write(f"- Entropy: {stats.get('entropy', 'N/A')}\n")
                f.write(f"- Gini Coefficient: {stats.get('gini_coefficient', 'N/A')}\n\n")
            
            f.write("## Principal Component Analysis\n\n")
            if 'pca' in self.statistics:
                pca_stats = self.statistics['pca']
                f.write(f"- Components needed for 95% variance: {pca_stats['n_components_95']}\n")
                f.write(f"- Explained variance ratios: {pca_stats['explained_variance_ratio'][:5]}\n\n")
            
            f.write("## Distribution Fitting Results\n\n")
            if 'distributions' in self.statistics:
                for col, dist_fits in self.statistics['distributions'].items():
                    f.write(f"### {col}\n")
                    best_dist = min(dist_fits.items(), key=lambda x: x[1].get('aic', float('inf')))
                    f.write(f"- Best fit distribution: {best_dist[0]}\n")
                    f.write(f"- AIC: {best_dist[1].get('aic', 'N/A')}\n")
                    f.write(f"- KS p-value: {best_dist[1].get('ks_pvalue', 'N/A')}\n\n")
            
            f.write("## Nonlinear Model Fitting\n\n")
            if 'nonlinear' in self.equations:
                for pair, models in self.equations['nonlinear'].items():
                    f.write(f"### {pair}\n")
                    best_model = max(models.items(), key=lambda x: x[1].get('r2', -float('inf')))
                    f.write(f"- Best model: {best_model[0]}\n")
                    f.write(f"- R²: {best_model[1].get('r2', 'N/A')}\n")
                    f.write(f"- Equation: {best_model[1].get('equation', 'N/A')}\n\n")
    
    def create_full_equations_document(self, filename='full_mathematical_expressions.md'):
        """Create document with all full mathematical expressions"""
        with open(filename, 'w') as f:
            f.write("# Full Mathematical Expressions\n\n")
            
            f.write("## Linear Regression Equations\n\n")
            
            for target, models in self.equations.get('regression', {}).items():
                f.write(f"### Target Variable: {target}\n\n")
                
                if 'linear' in models:
                    f.write("#### Linear Model\n")
                    f.write("```\n")
                    f.write(f"{target} = {models['linear']['equation']}\n")
                    f.write("```\n\n")
                    
                    # Write full coefficient list
                    f.write("##### Coefficients:\n")
                    for feature, coef in zip(models['features'], models['linear']['coefficients']):
                        f.write(f"- {feature}: {coef}\n")
                    f.write(f"- Intercept: {models['linear']['intercept']}\n\n")
            
            f.write("## Nonlinear Equations\n\n")
            
            for pair, models in self.equations.get('nonlinear', {}).items():
                f.write(f"### {pair}\n\n")
                
                for model_name, model_data in models.items():
                    f.write(f"#### {model_name}\n")
                    f.write("```\n")
                    f.write(f"{model_data['equation']}\n")
                    f.write("```\n")
                    f.write(f"Parameters: {model_data['parameters']}\n\n")
            
            f.write("## Master Similarity Equation\n\n")
            
            if 'master' in self.equations:
                f.write("### Full Expression\n")
                f.write("```\n")
                f.write(self.equations['master']['equation']['full'])
                f.write("\n```\n\n")
                
                f.write("### Feature Weights\n")
                for feature, coef_data in self.equations['master']['coefficients'].items():
                    f.write(f"- {feature}:\n")
                    f.write(f"  - Weight: {coef_data['weight']}\n")
                    f.write(f"  - Mean: {coef_data['mean']}\n")
                    f.write(f"  - Std: {coef_data['std']}\n")
                    f.write(f"  - Scaling Factor: {coef_data['scaling_factor']}\n\n")
                
                f.write("### Interpretation\n")
                f.write("Where:\n")
                f.write("- I = Input image feature vector\n")
                f.write("- D = Database image feature vector\n")
                f.write("- S(I,D) = Similarity score between I and D\n")
                f.write("- I_feature = Value of specific feature in input image\n")
                f.write("- D_feature = Value of specific feature in database image\n")
                f.write("- cos_sim = Cosine similarity between normalized feature vectors\n\n")
                
                f.write("The equation combines:\n")
                f.write("1. Weighted Euclidean distance in feature space\n")
                f.write("2. Cosine similarity for angular relationships\n")
                f.write("3. Exponential transformation for bounded similarity scores\n")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        self.load_all_json_files()
        self.prepare_dataframe()
        self.calculate_basic_statistics()
        self.calculate_advanced_statistics()
        self.calculate_correlations()
        self.fit_regression_models()
        self.fit_nonlinear_models()
        self.perform_pca_analysis()
        self.calculate_distribution_fits()
        self.create_master_equation()
        self.save_results()
        
        print("\nAnalysis complete! Check the generated files:")
        print("- basic_statistics.json")
        print("- advanced_statistics.json")
        print("- correlations.json")
        print("- equations.json")
        print("- statistical_analysis_report.md")
        print("- advanced_statistical_analysis.md")
        print("- full_mathematical_expressions.md")


if __name__ == "__main__":
    # Set the directory path
    directory_path = "/media/jarvis/New Volume/GitHub/polar-bear/Network/statistics/consensus"
    
    # Create analyzer and run
    analyzer = ComprehensiveJSONAnalyzer(directory_path)
    analyzer.run_analysis()