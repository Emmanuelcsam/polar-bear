#!/usr/bin/env python3
"""
Optimized Comprehensive JSON Data Analyzer for Consensus Reports
This script performs exhaustive statistical analysis on all JSON files with progress tracking
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
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import sys

class OptimizedJSONAnalyzer:
    def __init__(self, directory_path, batch_size=100):
        self.directory_path = directory_path
        self.batch_size = batch_size
        self.data = []
        self.df = None
        self.statistics = {}
        self.correlations = {}
        self.equations = {}
        self.start_time = datetime.now()
        
    def log_progress(self, message):
        """Log progress with timestamp"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"[{elapsed:.1f}s] {message}")
        sys.stdout.flush()
        
    def load_all_json_files(self):
        """Load all JSON files from the directory in batches"""
        self.log_progress("Starting to load JSON files...")
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith('.json')]
        total_files = len(json_files)
        self.log_progress(f"Found {total_files} JSON files")
        
        # Process in batches
        for i in range(0, total_files, self.batch_size):
            batch = json_files[i:i+self.batch_size]
            for file in batch:
                try:
                    with open(os.path.join(self.directory_path, file), 'r') as f:
                        data = json.load(f)
                        data['filename'] = file
                        self.data.append(data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            self.log_progress(f"Loaded {min(i+self.batch_size, total_files)}/{total_files} files")
        
        self.log_progress(f"Successfully loaded {len(self.data)} JSON files")
        
    def prepare_dataframe(self):
        """Convert JSON data to a structured DataFrame"""
        self.log_progress("Preparing DataFrame...")
        
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
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        
        # Create derived features
        self.df['core_cladding_ratio'] = self.df['core_radius'] / self.df['cladding_radius']
        self.df['core_area'] = np.pi * self.df['core_radius']**2
        self.df['cladding_area'] = np.pi * self.df['cladding_radius']**2
        self.df['cladding_core_area_diff'] = self.df['cladding_area'] - self.df['core_area']
        self.df['center_distance_from_origin'] = np.sqrt(self.df['center_x']**2 + self.df['center_y']**2)
        
        self.log_progress(f"DataFrame prepared with shape: {self.df.shape}")
        
    def calculate_basic_statistics(self):
        """Calculate basic statistical measurements"""
        self.log_progress("Calculating basic statistics...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.statistics['basic'] = {}
        
        for i, col in enumerate(numeric_cols):
            data = self.df[col].dropna()
            if len(data) > 0:
                self.statistics['basic'][col] = {
                    'count': int(len(data)),
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                    'variance': float(np.var(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'range': float(np.max(data) - np.min(data)),
                    'q1': float(np.percentile(data, 25)),
                    'q3': float(np.percentile(data, 75)),
                    'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
                    'cv': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else None,
                }
            
            if (i + 1) % 10 == 0:
                self.log_progress(f"Processed {i+1}/{len(numeric_cols)} columns for basic stats")
    
    def calculate_advanced_statistics(self):
        """Calculate advanced statistical measurements"""
        self.log_progress("Calculating advanced statistics...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.statistics['advanced'] = {}
        
        for i, col in enumerate(numeric_cols):
            data = self.df[col].dropna()
            if len(data) > 2:
                self.statistics['advanced'][col] = {
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'jarque_bera_statistic': float(stats.jarque_bera(data)[0]),
                    'jarque_bera_pvalue': float(stats.jarque_bera(data)[1]),
                    'sem': float(stats.sem(data)),
                    'mad': float(stats.median_abs_deviation(data)),
                    'trimmed_mean_10': float(stats.trim_mean(data, 0.1)),
                    'gini_coefficient': float(self._calculate_gini(data)),
                }
            
            if (i + 1) % 10 == 0:
                self.log_progress(f"Processed {i+1}/{len(numeric_cols)} columns for advanced stats")
    
    def _calculate_gini(self, data):
        """Calculate Gini coefficient"""
        sorted_data = np.sort(data)
        n = len(data)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n
    
    def calculate_correlations(self):
        """Calculate correlations efficiently"""
        self.log_progress("Calculating correlations...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrices
        self.correlations['pearson'] = self.df[numeric_cols].corr(method='pearson')
        self.correlations['spearman'] = self.df[numeric_cols].corr(method='spearman')
        
        self.log_progress("Correlations calculated")
    
    def fit_regression_models(self):
        """Fit regression models for key variables"""
        self.log_progress("Fitting regression models...")
        
        self.equations['regression'] = {}
        
        # Focus on key target variables
        key_targets = ['core_radius', 'cladding_radius', 'core_cladding_ratio']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for target_col in key_targets:
            if target_col not in numeric_cols:
                continue
                
            target_data = self.df[target_col].dropna()
            if len(target_data) < 10:
                continue
                
            # Get features
            feature_cols = [col for col in numeric_cols if col != target_col and 'accuracy_' in col]
            
            if len(feature_cols) == 0:
                continue
            
            # Prepare data
            common_idx = target_data.index
            for col in feature_cols:
                common_idx = common_idx.intersection(self.df[col].dropna().index)
            
            if len(common_idx) < 10:
                continue
                
            X = self.df.loc[common_idx, feature_cols].values
            y = self.df.loc[common_idx, target_col].values
            
            # Linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            self.equations['regression'][target_col] = {
                'features': feature_cols,
                'coefficients': lr.coef_.tolist(),
                'intercept': lr.intercept_,
                'r2': r2_score(y, y_pred),
                'equation': self._format_linear_equation(lr.coef_, lr.intercept_, feature_cols),
            }
            
            self.log_progress(f"Fitted regression model for {target_col}")
    
    def _format_linear_equation(self, coefficients, intercept, features):
        """Format linear equation as string"""
        equation = f"{intercept:.6f}"
        for coef, feature in zip(coefficients, features):
            if coef >= 0:
                equation += f" + {coef:.6f}*{feature}"
            else:
                equation += f" - {abs(coef):.6f}*{feature}"
        return equation
    
    def create_master_equation(self):
        """Create simplified master equation"""
        self.log_progress("Creating master equation...")
        
        # Get key numeric columns
        key_cols = ['center_x', 'center_y', 'core_radius', 'cladding_radius', 
                   'core_cladding_ratio', 'num_valid_results']
        
        # Filter to existing columns
        existing_cols = [col for col in key_cols if col in self.df.columns]
        
        # Calculate weights based on variance
        weights = {}
        for col in existing_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                weights[col] = np.var(data)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for col in weights:
            weights[col] = weights[col] / total_weight if total_weight > 0 else 0
        
        self.equations['master'] = {
            'features': existing_cols,
            'weights': weights,
            'equation': self._format_master_equation(weights, existing_cols),
        }
        
        self.log_progress("Master equation created")
    
    def _format_master_equation(self, weights, features):
        """Format master equation"""
        equation = "S(I, D) = exp(-sqrt("
        
        terms = []
        for feature in features:
            weight = weights.get(feature, 0)
            if weight > 0:
                terms.append(f"{weight:.6f}*(I_{feature} - D_{feature})^2")
        
        equation += " + ".join(terms)
        equation += "))"
        
        return equation
    
    def save_results(self):
        """Save all results to files"""
        self.log_progress("Saving results...")
        
        # Save basic statistics
        with open('basic_statistics.json', 'w') as f:
            json.dump(self.statistics['basic'], f, indent=2)
        
        # Save advanced statistics
        with open('advanced_statistics.json', 'w') as f:
            json.dump(self.statistics.get('advanced', {}), f, indent=2, default=str)
        
        # Save correlations
        with open('correlations.json', 'w') as f:
            correlations_serializable = {
                'pearson': self.correlations['pearson'].to_dict(),
                'spearman': self.correlations['spearman'].to_dict(),
            }
            json.dump(correlations_serializable, f, indent=2)
        
        # Save equations
        with open('equations.json', 'w') as f:
            json.dump(self.equations, f, indent=2)
        
        # Create reports
        self.create_markdown_report()
        self.create_advanced_report()
        self.create_full_equations_document()
        
        self.log_progress("All results saved")
    
    def create_markdown_report(self, filename='statistical_analysis_report.md'):
        """Create comprehensive markdown report"""
        with open(filename, 'w') as f:
            f.write("# Comprehensive Statistical Analysis Report\n\n")
            
            f.write(f"## Overview\n")
            f.write(f"- Total files analyzed: {len(self.data)}\n")
            f.write(f"- Total features: {len(self.df.columns)}\n")
            f.write(f"- Numeric features: {len(self.df.select_dtypes(include=[np.number]).columns)}\n\n")
            
            f.write("## Key Statistics Summary\n\n")
            
            # Focus on key columns
            key_cols = ['center_x', 'center_y', 'core_radius', 'cladding_radius', 
                       'core_cladding_ratio', 'num_valid_results']
            
            for col in key_cols:
                if col in self.statistics['basic']:
                    stats = self.statistics['basic'][col]
                    f.write(f"### {col}\n")
                    f.write(f"- Mean: {stats['mean']:.4f}\n")
                    f.write(f"- Std: {stats['std']:.4f}\n")
                    f.write(f"- Min: {stats['min']:.4f}\n")
                    f.write(f"- Max: {stats['max']:.4f}\n")
                    f.write(f"- Median: {stats['median']:.4f}\n\n")
            
            f.write("## Method Accuracy Statistics\n\n")
            
            accuracy_cols = [col for col in self.df.columns if col.startswith('accuracy_')]
            for col in accuracy_cols:
                if col in self.statistics['basic']:
                    stats = self.statistics['basic'][col]
                    method_name = col.replace('accuracy_', '')
                    f.write(f"### {method_name}\n")
                    f.write(f"- Mean accuracy: {stats['mean']:.4f}\n")
                    f.write(f"- Std: {stats['std']:.4f}\n\n")
            
            f.write("## Correlation Analysis\n\n")
            
            # Get high correlations
            corr_matrix = self.correlations['pearson']
            f.write("### High Correlations (|r| > 0.5)\n")
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        f.write(f"- {col1} vs {col2}: {corr_value:.4f}\n")
            
            f.write("\n## Regression Models\n\n")
            for target, model in self.equations.get('regression', {}).items():
                f.write(f"### Target: {target}\n")
                f.write(f"- R²: {model['r2']:.4f}\n")
                f.write(f"- Equation: `{model['equation']}`\n\n")
            
            f.write("## Master Equation\n\n")
            if 'master' in self.equations:
                f.write("```\n")
                f.write(self.equations['master']['equation'])
                f.write("\n```\n")
    
    def create_advanced_report(self, filename='advanced_statistical_analysis.md'):
        """Create PhD-level advanced statistical report"""
        with open(filename, 'w') as f:
            f.write("# Advanced Statistical Analysis Report\n\n")
            
            f.write("## Distribution Analysis\n\n")
            
            key_cols = ['core_radius', 'cladding_radius', 'core_cladding_ratio']
            
            for col in key_cols:
                if col in self.statistics.get('advanced', {}):
                    stats = self.statistics['advanced'][col]
                    f.write(f"### {col}\n")
                    f.write(f"- Skewness: {stats['skewness']:.4f}\n")
                    f.write(f"- Kurtosis: {stats['kurtosis']:.4f}\n")
                    f.write(f"- Jarque-Bera statistic: {stats['jarque_bera_statistic']:.4f}\n")
                    f.write(f"- Jarque-Bera p-value: {stats['jarque_bera_pvalue']:.4f}\n")
                    f.write(f"- Standard Error of Mean: {stats['sem']:.4f}\n")
                    f.write(f"- Median Absolute Deviation: {stats['mad']:.4f}\n")
                    f.write(f"- Trimmed Mean (10%): {stats['trimmed_mean_10']:.4f}\n")
                    f.write(f"- Gini Coefficient: {stats['gini_coefficient']:.4f}\n\n")
            
            f.write("## Correlation Matrix Analysis\n\n")
            
            # Analyze correlation structure
            pearson_corr = self.correlations['pearson']
            spearman_corr = self.correlations['spearman']
            
            f.write("### Correlation Comparison (Pearson vs Spearman)\n")
            f.write("Differences indicate non-linear relationships:\n\n")
            
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    pearson_val = pearson_corr.iloc[i, j]
                    spearman_val = spearman_corr.iloc[i, j]
                    diff = abs(pearson_val - spearman_val)
                    
                    if diff > 0.1:
                        col1 = pearson_corr.columns[i]
                        col2 = pearson_corr.columns[j]
                        f.write(f"- {col1} vs {col2}:\n")
                        f.write(f"  - Pearson: {pearson_val:.4f}\n")
                        f.write(f"  - Spearman: {spearman_val:.4f}\n")
                        f.write(f"  - Difference: {diff:.4f}\n\n")
    
    def create_full_equations_document(self, filename='full_mathematical_expressions.md'):
        """Create document with all full mathematical expressions"""
        with open(filename, 'w') as f:
            f.write("# Full Mathematical Expressions\n\n")
            
            f.write("## Linear Regression Equations\n\n")
            
            for target, model in self.equations.get('regression', {}).items():
                f.write(f"### Target Variable: {target}\n\n")
                f.write("#### Full Equation:\n")
                f.write("```\n")
                f.write(f"{target} = {model['intercept']:.10f}")
                
                for feature, coef in zip(model['features'], model['coefficients']):
                    if coef >= 0:
                        f.write(f"\n        + {coef:.10f} * {feature}")
                    else:
                        f.write(f"\n        - {abs(coef):.10f} * {feature}")
                
                f.write("\n```\n\n")
                
                f.write("#### Coefficient Table:\n")
                f.write("| Feature | Coefficient |\n")
                f.write("|---------|-------------|\n")
                f.write(f"| Intercept | {model['intercept']:.10f} |\n")
                for feature, coef in zip(model['features'], model['coefficients']):
                    f.write(f"| {feature} | {coef:.10f} |\n")
                f.write("\n")
                
                f.write(f"#### Model Performance:\n")
                f.write(f"- R² Score: {model['r2']:.6f}\n\n")
            
            f.write("## Master Similarity Equation\n\n")
            
            if 'master' in self.equations:
                f.write("### Full Expression:\n")
                f.write("```\n")
                f.write("S(I, D) = exp(-sqrt(\n")
                
                weights = self.equations['master']['weights']
                features = self.equations['master']['features']
                
                for i, (feature, weight) in enumerate(zip(features, [weights[f] for f in features])):
                    if i > 0:
                        f.write(" +\n")
                    f.write(f"    {weight:.10f} * (I_{feature} - D_{feature})^2")
                
                f.write("\n))\n```\n\n")
                
                f.write("### Weight Table:\n")
                f.write("| Feature | Weight |\n")
                f.write("|---------|--------|\n")
                for feature in features:
                    f.write(f"| {feature} | {weights[feature]:.10f} |\n")
                f.write("\n")
                
                f.write("### Interpretation:\n")
                f.write("- S(I, D): Similarity score between images I and D (range: 0 to 1)\n")
                f.write("- I_feature: Value of feature in input image I\n")
                f.write("- D_feature: Value of feature in database image D\n")
                f.write("- Weights are normalized variance-based importance factors\n")
                f.write("- The exponential of negative distance ensures similarity decreases with distance\n")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        try:
            self.load_all_json_files()
            self.prepare_dataframe()
            self.calculate_basic_statistics()
            self.calculate_advanced_statistics()
            self.calculate_correlations()
            self.fit_regression_models()
            self.create_master_equation()
            self.save_results()
            
            total_time = (datetime.now() - self.start_time).total_seconds()
            self.log_progress(f"\nAnalysis complete in {total_time:.1f} seconds!")
            print("\nGenerated files:")
            print("- basic_statistics.json")
            print("- advanced_statistics.json")
            print("- correlations.json")
            print("- equations.json")
            print("- statistical_analysis_report.md")
            print("- advanced_statistical_analysis.md")
            print("- full_mathematical_expressions.md")
            
        except Exception as e:
            self.log_progress(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    # Set the directory path
    directory_path = "/media/jarvis/New Volume/GitHub/polar-bear/Network/statistics/consensus"
    
    # Create analyzer and run
    analyzer = OptimizedJSONAnalyzer(directory_path, batch_size=500)
    analyzer.run_analysis()