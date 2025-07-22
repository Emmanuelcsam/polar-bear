import numpy as np
from sympy import symbols, expand, latex, simplify
import json
from datetime import datetime

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
                    autocorr = np.corrcoef(flat_data[:-lag], flat_data[lag:])[0, 1]
                    if not np.isnan(autocorr):
                        expression_lines.append(f"    + ({autocorr:.20e} * w_{{{file_idx}}}_autocorr_lag_{lag})")
                        term_counter += 1
            
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

# Main execution function
def run_complete_analysis(folder_path):
    """Run all three levels of analysis"""
    print("Starting comprehensive NPY analysis...")
    print("="*50)
    
    # Step 1: Basic comprehensive analysis
    print("\nStep 1: Running basic comprehensive analysis...")
    basic_analyzer = ComprehensiveNPYAnalyzer(folder_path)
    basic_results = basic_analyzer.analyze_all()
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
    # IMPORTANT: Replace this with your actual folder path
    folder_path = r"C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\defect-library-npyr"
    
    # Run the complete analysis
    basic, advanced, expressions = run_complete_analysis(folder_path)
    
    print("\nAnalysis complete! Check the output files for results.")
