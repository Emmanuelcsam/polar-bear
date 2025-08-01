#!/usr/bin/env python3
"""
Advanced Feature Extraction Module
Comprehensive feature extraction for image analysis and ML applications.
Extracted and optimized from detection.py OmniFiberAnalyzer.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging


class AdvancedFeatureExtractor:
    """
    Comprehensive feature extraction class for image analysis.
    Includes statistical, texture, morphological, and spectral features.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _sanitize_feature_value(self, value):
        """Ensure feature values are JSON-serializable floats."""
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def _compute_skewness(self, data):
        """Compute skewness of data distribution."""
        try:
            data_flat = data.flatten()
            n = len(data_flat)
            if n < 3:
                return 0.0
            
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            
            if std == 0:
                return 0.0
            
            skew = np.sum(((data_flat - mean) / std) ** 3) / n
            return skew
        except:
            return 0.0
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis of data distribution."""
        try:
            data_flat = data.flatten()
            n = len(data_flat)
            if n < 4:
                return 0.0
            
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            
            if std == 0:
                return 0.0
            
            kurt = np.sum(((data_flat - mean) / std) ** 4) / n - 3
            return kurt
        except:
            return 0.0
    
    def _compute_entropy(self, data, bins=256):
        """Compute information entropy of image data."""
        try:
            hist, _ = np.histogram(data.flatten(), bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy
        except:
            return 0.0
    
    def extract_statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive statistical features from grayscale image."""
        flat = gray.flatten().astype(np.float64)
        
        # Basic statistics
        percentiles = np.percentile(flat, [10, 25, 50, 75, 90])
        
        features = {
            # Central tendency
            'stat_mean': self._sanitize_feature_value(np.mean(gray)),
            'stat_median': self._sanitize_feature_value(np.median(gray)),
            'stat_mode': self._sanitize_feature_value(np.bincount(gray.flatten()).argmax()),
            
            # Dispersion
            'stat_std': self._sanitize_feature_value(np.std(gray)),
            'stat_variance': self._sanitize_feature_value(np.var(gray)),
            'stat_range': self._sanitize_feature_value(np.max(gray) - np.min(gray)),
            'stat_iqr': self._sanitize_feature_value(percentiles[3] - percentiles[1]),
            'stat_mad': self._sanitize_feature_value(np.median(np.abs(gray - np.median(gray)))),
            
            # Shape of distribution
            'stat_skew': self._sanitize_feature_value(self._compute_skewness(flat)),
            'stat_kurtosis': self._sanitize_feature_value(self._compute_kurtosis(flat)),
            
            # Extremes
            'stat_min': self._sanitize_feature_value(np.min(gray)),
            'stat_max': self._sanitize_feature_value(np.max(gray)),
            
            # Percentiles
            'stat_p10': self._sanitize_feature_value(percentiles[0]),
            'stat_p25': self._sanitize_feature_value(percentiles[1]),
            'stat_p50': self._sanitize_feature_value(percentiles[2]),
            'stat_p75': self._sanitize_feature_value(percentiles[3]),
            'stat_p90': self._sanitize_feature_value(percentiles[4]),
            
            # Information theory
            'stat_entropy': self._sanitize_feature_value(self._compute_entropy(gray)),
            'stat_energy': self._sanitize_feature_value(np.sum(gray.astype(np.float64)**2)),
            
            # Robust statistics
            'stat_trimmed_mean_10': self._sanitize_feature_value(np.mean(np.sort(flat)[int(len(flat)*0.1):int(len(flat)*0.9)])),
            'stat_trimmed_mean_20': self._sanitize_feature_value(np.mean(np.sort(flat)[int(len(flat)*0.2):int(len(flat)*0.8)])),
        }
        
        return features
    
    def extract_matrix_norms(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract various matrix norms and linear algebra features."""
        try:
            gray_float = gray.astype(np.float64)
            
            features = {
                'norm_frobenius': self._sanitize_feature_value(np.linalg.norm(gray_float, 'fro')),
                'norm_l1': self._sanitize_feature_value(np.sum(np.abs(gray_float))),
                'norm_l2': self._sanitize_feature_value(np.sqrt(np.sum(gray_float**2))),
                'norm_linf': self._sanitize_feature_value(np.max(np.abs(gray_float))),
                'norm_trace': self._sanitize_feature_value(np.trace(gray_float)),
            }
            
            # SVD features (on downsampled image for performance)
            if gray.shape[0] > 100 or gray.shape[1] > 100:
                small_gray = cv2.resize(gray_float, (64, 64))
            else:
                small_gray = gray_float
            
            try:
                svd_values = np.linalg.svd(small_gray, compute_uv=False)
                features.update({
                    'norm_nuclear': self._sanitize_feature_value(np.sum(svd_values)),
                    'svd_rank_approx': self._sanitize_feature_value(np.sum(svd_values > svd_values[0] * 0.01)),
                    'svd_entropy': self._sanitize_feature_value(self._compute_entropy(svd_values)),
                    'svd_energy_ratio': self._sanitize_feature_value(svd_values[0] / (np.sum(svd_values) + 1e-10)),
                })
            except:
                features.update({
                    'norm_nuclear': 0.0,
                    'svd_rank_approx': 0.0,
                    'svd_entropy': 0.0,
                    'svd_energy_ratio': 0.0,
                })
            
            return features
        except:
            return {
                'norm_frobenius': 0.0, 'norm_l1': 0.0, 'norm_l2': 0.0, 
                'norm_linf': 0.0, 'norm_trace': 0.0, 'norm_nuclear': 0.0,
                'svd_rank_approx': 0.0, 'svd_entropy': 0.0, 'svd_energy_ratio': 0.0
            }
    
    def extract_lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern features using custom implementation."""
        features = {}
        
        try:
            # Compute LBP at multiple radii
            for radius in [1, 2, 3, 5]:
                # Initialize LBP result array
                lbp = np.zeros_like(gray, dtype=np.float32)
                
                # Iterate through neighborhood offsets
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        # Skip center pixel
                        if dx == 0 and dy == 0:
                            continue
                        
                        # Create shifted version of image
                        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                        
                        # Compare shifted with original (binary pattern)
                        lbp += (shifted >= gray).astype(np.float32)
                
                # Compute statistics of LBP
                features[f'lbp_r{radius}_mean'] = self._sanitize_feature_value(np.mean(lbp))
                features[f'lbp_r{radius}_std'] = self._sanitize_feature_value(np.std(lbp))
                features[f'lbp_r{radius}_entropy'] = self._sanitize_feature_value(self._compute_entropy(lbp))
                features[f'lbp_r{radius}_energy'] = self._sanitize_feature_value(np.sum(lbp**2) / lbp.size)
                features[f'lbp_r{radius}_contrast'] = self._sanitize_feature_value(np.var(lbp))
                features[f'lbp_r{radius}_uniformity'] = self._sanitize_feature_value(np.sum((lbp / (lbp.sum() + 1e-10))**2))
        
        except Exception as e:
            self.logger.warning(f"LBP feature extraction failed: {e}")
            # Return default values
            for radius in [1, 2, 3, 5]:
                features.update({
                    f'lbp_r{radius}_mean': 0.0,
                    f'lbp_r{radius}_std': 0.0,
                    f'lbp_r{radius}_entropy': 0.0,
                    f'lbp_r{radius}_energy': 0.0,
                    f'lbp_r{radius}_contrast': 0.0,
                    f'lbp_r{radius}_uniformity': 0.0,
                })
        
        return features
    
    def extract_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix features using custom implementation."""
        features = {}
        
        try:
            # Quantize image to reduce computation
            img_q = (gray // 32).astype(np.uint8)
            levels = 8
            
            # Define distances and angles for GLCM
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]  # degrees
            
            # Compute GLCM for each distance and angle
            for dist in distances:
                for angle in angles:
                    # Initialize GLCM matrix
                    glcm = np.zeros((levels, levels), dtype=np.float32)
                    
                    # Determine pixel offset based on angle
                    if angle == 0:
                        dy, dx = 0, dist       # Horizontal
                    elif angle == 45:
                        dy, dx = -dist, dist   # Diagonal up-right
                    elif angle == 90:
                        dy, dx = -dist, 0      # Vertical
                    else:  # 135
                        dy, dx = -dist, -dist  # Diagonal up-left
                    
                    # Build GLCM by counting co-occurrences
                    rows, cols = img_q.shape
                    for i in range(rows):
                        for j in range(cols):
                            # Check if neighbor is within bounds
                            if 0 <= i + dy < rows and 0 <= j + dx < cols:
                                # Increment co-occurrence count
                                glcm[img_q[i, j], img_q[i + dy, j + dx]] += 1
                    
                    # Normalize GLCM to probabilities
                    glcm = glcm / (glcm.sum() + 1e-10)
                    
                    # Compute Haralick texture features
                    feature_prefix = f'glcm_d{dist}_a{angle}'
                    
                    # Energy (Angular Second Moment)
                    energy = np.sum(glcm**2)
                    features[f'{feature_prefix}_energy'] = self._sanitize_feature_value(energy)
                    
                    # Contrast
                    contrast = 0
                    for i in range(levels):
                        for j in range(levels):
                            contrast += glcm[i, j] * (i - j)**2
                    features[f'{feature_prefix}_contrast'] = self._sanitize_feature_value(contrast)
                    
                    # Correlation
                    # First compute means
                    mu_i = np.sum([i * np.sum(glcm[i, :]) for i in range(levels)])
                    mu_j = np.sum([j * np.sum(glcm[:, j]) for j in range(levels)])
                    
                    # Compute correlation
                    correlation = 0
                    sigma_i = np.sqrt(np.sum([(i - mu_i)**2 * np.sum(glcm[i, :]) for i in range(levels)]))
                    sigma_j = np.sqrt(np.sum([(j - mu_j)**2 * np.sum(glcm[:, j]) for j in range(levels)]))
                    
                    if sigma_i > 0 and sigma_j > 0:
                        for i in range(levels):
                            for j in range(levels):
                                correlation += glcm[i, j] * (i - mu_i) * (j - mu_j) / (sigma_i * sigma_j)
                    
                    features[f'{feature_prefix}_correlation'] = self._sanitize_feature_value(correlation)
                    
                    # Homogeneity (Inverse Difference Moment)
                    homogeneity = np.sum([glcm[i, j] / (1 + abs(i - j)) for i in range(levels) for j in range(levels)])
                    features[f'{feature_prefix}_homogeneity'] = self._sanitize_feature_value(homogeneity)
                    
                    # Entropy
                    entropy = -np.sum([glcm[i, j] * np.log2(glcm[i, j] + 1e-10) for i in range(levels) for j in range(levels)])
                    features[f'{feature_prefix}_entropy'] = self._sanitize_feature_value(entropy)
        
        except Exception as e:
            self.logger.warning(f"GLCM feature extraction failed: {e}")
            # Return default values for expected features
            for dist in [1, 2, 3]:
                for angle in [0, 45, 90, 135]:
                    prefix = f'glcm_d{dist}_a{angle}'
                    features.update({
                        f'{prefix}_energy': 0.0,
                        f'{prefix}_contrast': 0.0,
                        f'{prefix}_correlation': 0.0,
                        f'{prefix}_homogeneity': 0.0,
                        f'{prefix}_entropy': 0.0,
                    })
        
        return features
    
    def extract_fourier_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        features = {}
        
        try:
            # Compute 2D FFT
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            # Power spectrum
            power_spectrum = magnitude**2
            
            # Basic frequency domain statistics
            features['fft_mean_magnitude'] = self._sanitize_feature_value(np.mean(magnitude))
            features['fft_std_magnitude'] = self._sanitize_feature_value(np.std(magnitude))
            features['fft_energy'] = self._sanitize_feature_value(np.sum(power_spectrum))
            features['fft_entropy'] = self._sanitize_feature_value(self._compute_entropy(magnitude))
            
            # Phase statistics
            features['fft_phase_mean'] = self._sanitize_feature_value(np.mean(phase))
            features['fft_phase_std'] = self._sanitize_feature_value(np.std(phase))
            
            # Radial frequency analysis
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Compute radial power spectrum
            max_radius = int(min(cx, cy))
            radial_power = []
            for r in range(0, max_radius, max(1, max_radius // 20)):
                mask = (radius >= r) & (radius < r + 1)
                if np.any(mask):
                    radial_power.append(np.mean(power_spectrum[mask]))
                else:
                    radial_power.append(0)
            
            radial_power = np.array(radial_power)
            features['fft_radial_mean'] = self._sanitize_feature_value(np.mean(radial_power))
            features['fft_radial_std'] = self._sanitize_feature_value(np.std(radial_power))
            features['fft_radial_peak'] = self._sanitize_feature_value(np.max(radial_power))
            
            # Low/high frequency energy ratio
            center_size = min(h, w) // 4
            center_mask = (radius <= center_size)
            low_freq_energy = np.sum(power_spectrum[center_mask])
            high_freq_energy = np.sum(power_spectrum[~center_mask])
            total_energy = low_freq_energy + high_freq_energy
            
            if total_energy > 0:
                features['fft_low_freq_ratio'] = self._sanitize_feature_value(low_freq_energy / total_energy)
                features['fft_high_freq_ratio'] = self._sanitize_feature_value(high_freq_energy / total_energy)
            else:
                features['fft_low_freq_ratio'] = 0.0
                features['fft_high_freq_ratio'] = 0.0
                
        except Exception as e:
            self.logger.warning(f"Fourier feature extraction failed: {e}")
            features = {
                'fft_mean_magnitude': 0.0, 'fft_std_magnitude': 0.0, 'fft_energy': 0.0, 'fft_entropy': 0.0,
                'fft_phase_mean': 0.0, 'fft_phase_std': 0.0, 'fft_radial_mean': 0.0, 'fft_radial_std': 0.0,
                'fft_radial_peak': 0.0, 'fft_low_freq_ratio': 0.0, 'fft_high_freq_ratio': 0.0
            }
        
        return features
    
    def extract_morphological_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract morphological features using various structuring elements."""
        features = {}
        
        try:
            # Convert to binary image for morphological operations
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Define structuring elements
            kernels = {
                'rect_3x3': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                'rect_5x5': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                'ellipse_3x3': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                'ellipse_5x5': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            }
            
            for kernel_name, kernel in kernels.items():
                # Morphological operations
                erosion = cv2.erode(binary, kernel, iterations=1)
                dilation = cv2.dilate(binary, kernel, iterations=1)
                opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
                
                # Compute features for each operation
                operations = {
                    'erosion': erosion,
                    'dilation': dilation,
                    'opening': opening,
                    'closing': closing,
                    'gradient': gradient
                }
                
                for op_name, result in operations.items():
                    feature_name = f'morph_{kernel_name}_{op_name}'
                    features[f'{feature_name}_mean'] = self._sanitize_feature_value(np.mean(result))
                    features[f'{feature_name}_std'] = self._sanitize_feature_value(np.std(result))
                    features[f'{feature_name}_nonzero'] = self._sanitize_feature_value(np.count_nonzero(result))
                    
        except Exception as e:
            self.logger.warning(f"Morphological feature extraction failed: {e}")
            # Return default values
            kernels = ['rect_3x3', 'rect_5x5', 'ellipse_3x3', 'ellipse_5x5']
            operations = ['erosion', 'dilation', 'opening', 'closing', 'gradient']
            for kernel_name in kernels:
                for op_name in operations:
                    feature_name = f'morph_{kernel_name}_{op_name}'
                    features.update({
                        f'{feature_name}_mean': 0.0,
                        f'{feature_name}_std': 0.0,
                        f'{feature_name}_nonzero': 0.0,
                    })
        
        return features
    
    def extract_gradient_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract gradient-based features."""
        features = {}
        
        try:
            # Compute gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude and direction
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            # Magnitude statistics
            features['grad_magnitude_mean'] = self._sanitize_feature_value(np.mean(magnitude))
            features['grad_magnitude_std'] = self._sanitize_feature_value(np.std(magnitude))
            features['grad_magnitude_max'] = self._sanitize_feature_value(np.max(magnitude))
            features['grad_magnitude_energy'] = self._sanitize_feature_value(np.sum(magnitude**2))
            
            # Direction statistics
            features['grad_direction_std'] = self._sanitize_feature_value(np.std(direction))
            features['grad_direction_entropy'] = self._sanitize_feature_value(self._compute_entropy(direction))
            
            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['laplacian_mean'] = self._sanitize_feature_value(np.mean(np.abs(laplacian)))
            features['laplacian_std'] = self._sanitize_feature_value(np.std(laplacian))
            features['laplacian_energy'] = self._sanitize_feature_value(np.sum(laplacian**2))
            
        except Exception as e:
            self.logger.warning(f"Gradient feature extraction failed: {e}")
            features = {
                'grad_magnitude_mean': 0.0, 'grad_magnitude_std': 0.0, 'grad_magnitude_max': 0.0,
                'grad_magnitude_energy': 0.0, 'grad_direction_std': 0.0, 'grad_direction_entropy': 0.0,
                'laplacian_mean': 0.0, 'laplacian_std': 0.0, 'laplacian_energy': 0.0
            }
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract all available features from an image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        all_features = {}
        
        # Extract all feature types
        feature_extractors = [
            ('statistical', self.extract_statistical_features),
            ('matrix_norms', self.extract_matrix_norms),
            ('lbp', self.extract_lbp_features),
            ('glcm', self.extract_glcm_features),
            ('fourier', self.extract_fourier_features),
            ('morphological', self.extract_morphological_features),
            ('gradient', self.extract_gradient_features),
        ]
        
        for feature_type, extractor in feature_extractors:
            try:
                features = extractor(gray)
                all_features.update(features)
                self.logger.info(f"Extracted {len(features)} {feature_type} features")
            except Exception as e:
                self.logger.error(f"Failed to extract {feature_type} features: {e}")
        
        self.logger.info(f"Total features extracted: {len(all_features)}")
        return all_features


def main():
    """Test the AdvancedFeatureExtractor functionality."""
    print("Testing AdvancedFeatureExtractor...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create feature extractor
    extractor = AdvancedFeatureExtractor()
    
    # Ask for image path
    image_path = input("Enter path to test image (or press Enter to create synthetic): ").strip()
    
    if not image_path:
        print("Creating synthetic test image...")
        # Create a synthetic test image with patterns
        img = np.zeros((200, 200), dtype=np.uint8)
        
        # Add circles
        cv2.circle(img, (50, 50), 30, 150, -1)
        cv2.circle(img, (150, 150), 25, 200, -1)
        
        # Add rectangles
        cv2.rectangle(img, (80, 80), (120, 120), 100, -1)
        
        # Add noise
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        test_path = "synthetic_feature_test.png"
        cv2.imwrite(test_path, img)
        image_path = test_path
        print(f"Created synthetic test image: {test_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # Extract features
    features = extractor.extract_all_features(img)
    
    print(f"\nExtracted {len(features)} features:")
    
    # Group features by type for better readability
    feature_groups = {}
    for feature_name, value in features.items():
        group = feature_name.split('_')[0]
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append((feature_name, value))
    
    for group, group_features in feature_groups.items():
        print(f"\n{group.upper()} Features ({len(group_features)}):")
        for name, value in sorted(group_features)[:5]:  # Show first 5 of each group
            print(f"  {name}: {value:.4f}")
        if len(group_features) > 5:
            print(f"  ... and {len(group_features) - 5} more")
    
    # Save features to JSON file
    from . import numpy_json_encoder
    output_file = "extracted_features.json"
    numpy_json_encoder.save_numpy_data(features, output_file)
    print(f"\nFeatures saved to: {output_file}")


if __name__ == "__main__":
    import os
    main()
