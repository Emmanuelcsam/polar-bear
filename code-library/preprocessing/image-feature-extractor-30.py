#!/usr/bin/env python3
"""
Image Feature Extractor - Ultra Comprehensive Feature Extraction
Extracted from detection.py - Standalone modular script
"""

import cv2
import numpy as np
import json
import sys
import os
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class UltraFeatureExtractor:
    """Ultra comprehensive feature extractor for image analysis."""
    
    def __init__(self):
        self.logger = logger
    
    def load_image(self, image_path):
        """Load image from file path."""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
        return img
    
    def _sanitize_feature_value(self, value):
        """Ensure feature value is finite and valid."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    
    def _compute_skewness(self, data):
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data):
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_entropy(self, data, bins=256):
        """Compute Shannon entropy."""
        hist, _ = np.histogram(data, bins=bins, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _extract_statistical_features(self, gray):
        """Extract comprehensive statistical features."""
        flat = gray.flatten()
        percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
        
        return {
            'stat_mean': float(np.mean(gray)),
            'stat_std': float(np.std(gray)),
            'stat_variance': float(np.var(gray)),
            'stat_skew': float(self._compute_skewness(flat)),
            'stat_kurtosis': float(self._compute_kurtosis(flat)),
            'stat_min': float(np.min(gray)),
            'stat_max': float(np.max(gray)),
            'stat_range': float(np.max(gray) - np.min(gray)),
            'stat_median': float(np.median(gray)),
            'stat_mad': float(np.median(np.abs(gray - np.median(gray)))),
            'stat_iqr': float(percentiles[3] - percentiles[1]),
            'stat_entropy': float(self._compute_entropy(gray)),
            'stat_energy': float(np.sum(gray**2)),
            'stat_p10': float(percentiles[0]),
            'stat_p25': float(percentiles[1]),
            'stat_p50': float(percentiles[2]),
            'stat_p75': float(percentiles[3]),
            'stat_p90': float(percentiles[4]),
        }
    
    def _extract_matrix_norms(self, gray):
        """Extract various matrix norms."""
        return {
            'norm_frobenius': float(np.linalg.norm(gray, 'fro')),
            'norm_l1': float(np.sum(np.abs(gray))),
            'norm_l2': float(np.sqrt(np.sum(gray**2))),
            'norm_linf': float(np.max(np.abs(gray))),
            'norm_nuclear': float(np.sum(np.linalg.svd(gray, compute_uv=False))),
            'norm_trace': float(np.trace(gray)),
        }
    
    def _extract_gradient_features(self, gray):
        """Extract gradient-based features."""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_orient = np.arctan2(grad_y, grad_x)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        return {
            'gradient_magnitude_mean': float(np.mean(grad_mag)),
            'gradient_magnitude_std': float(np.std(grad_mag)),
            'gradient_magnitude_max': float(np.max(grad_mag)),
            'gradient_magnitude_sum': float(np.sum(grad_mag)),
            'gradient_orientation_mean': float(np.mean(grad_orient)),
            'gradient_orientation_std': float(np.std(grad_orient)),
            'laplacian_mean': float(np.mean(np.abs(laplacian))),
            'laplacian_std': float(np.std(laplacian)),
            'laplacian_sum': float(np.sum(np.abs(laplacian))),
            'edge_density': float(edge_density),
            'edge_count': float(np.sum(edges > 0)),
        }
    
    def _extract_fourier_features(self, gray):
        """Extract 2D Fourier Transform features."""
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        power = magnitude**2
        phase = np.angle(fshift)
        
        center = np.array(power.shape) // 2
        y, x = np.ogrid[:power.shape[0], :power.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        radial_prof = []
        for radius in range(1, min(center)):
            mask = (r >= radius - 1) & (r < radius)
            if mask.any():
                radial_prof.append(np.mean(power[mask]))
        
        radial_prof = np.array(radial_prof)
        
        if len(radial_prof) > 0:
            spectral_centroid = float(np.sum(np.arange(len(radial_prof)) * radial_prof) / (np.sum(radial_prof) + 1e-10))
            spectral_spread = float(np.sqrt(np.sum((np.arange(len(radial_prof)) - spectral_centroid)**2 * radial_prof) / (np.sum(radial_prof) + 1e-10)))
        else:
            spectral_centroid = 0.0
            spectral_spread = 0.0
        
        return {
            'fft_mean_magnitude': float(np.mean(magnitude)),
            'fft_std_magnitude': float(np.std(magnitude)),
            'fft_max_magnitude': float(np.max(magnitude)),
            'fft_total_power': float(np.sum(power)),
            'fft_dc_component': float(magnitude[center[0], center[1]]),
            'fft_mean_phase': float(np.mean(phase)),
            'fft_std_phase': float(np.std(phase)),
            'fft_spectral_centroid': spectral_centroid,
            'fft_spectral_spread': spectral_spread,
        }
    
    def _extract_morphological_features(self, gray):
        """Extract morphological features."""
        features = {}
        
        for size in [3, 5, 7, 11]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            
            wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
            features[f'morph_wth_{size}_max'] = float(np.max(wth))
            features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
            features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
            features[f'morph_bth_{size}_max'] = float(np.max(bth))
            features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        
        features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)
        features['morph_gradient_sum'] = float(np.sum(gradient))
        features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))
        features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))
        
        return features
    
    def extract_ultra_comprehensive_features(self, image):
        """Extract 100+ features using all available methods."""
        features = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        self.logger.info("Extracting features...")
        
        feature_extractors = [
            ("Stats", self._extract_statistical_features),
            ("Norms", self._extract_matrix_norms),
            ("FFT", self._extract_fourier_features),
            ("Morph", self._extract_morphological_features),
            ("Gradient", self._extract_gradient_features),
        ]
        
        for name, extractor in feature_extractors:
            try:
                features.update(extractor(gray))
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for {name}: {e}")
        
        sanitized_features = {}
        for key, value in features.items():
            sanitized_features[key] = self._sanitize_feature_value(value)
        
        feature_names = sorted(sanitized_features.keys())
        return sanitized_features, feature_names
    
    def extract_features_from_file(self, image_path, output_path=None):
        """Extract features from image file and optionally save to JSON."""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        features, feature_names = self.extract_ultra_comprehensive_features(image)
        
        result = {
            'image_path': image_path,
            'feature_count': len(features),
            'feature_names': feature_names,
            'features': features,
            'extracted_at': str(np.datetime64('now')),
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Features saved to: {output_path}")
        
        return result


def main():
    """Command line interface for feature extraction."""
    parser = argparse.ArgumentParser(description='Extract ultra-comprehensive features from images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    extractor = UltraFeatureExtractor()
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        input_path = Path(args.image_path)
        output_path = input_path.parent / f"{input_path.stem}_features.json"
    
    result = extractor.extract_features_from_file(args.image_path, output_path)
    
    if result:
        print(f"Successfully extracted {result['feature_count']} features")
        print(f"Features saved to: {output_path}")
    else:
        print("Feature extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
