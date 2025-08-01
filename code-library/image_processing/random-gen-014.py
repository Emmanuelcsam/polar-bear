#!/usr/bin/env python3
"""
Gabor Filter Bank Module
========================

This module provides comprehensive Gabor filtering for detecting oriented textures,
edges, and patterns in fiber optic defect analysis.

Author: Modular Analysis Team
Version: 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
import logging


class GaborFilterBank:
    """
    Comprehensive Gabor filter bank for texture and orientation analysis.
    """
    
    def __init__(self, 
                 ksize: int = 31,
                 sigma_x: float = 4.0,
                 sigma_y: float = 4.0,
                 theta_range: Tuple[float, float] = (0, np.pi),
                 theta_steps: int = 8,
                 lambda_range: Tuple[float, float] = (8, 32),
                 lambda_steps: int = 4,
                 gamma: float = 0.5,
                 psi: float = 0):
        """
        Initialize Gabor filter bank.
        
        Args:
            ksize: Kernel size (should be odd)
            sigma_x: Standard deviation in x direction
            sigma_y: Standard deviation in y direction  
            theta_range: Range of orientations (start, end) in radians
            theta_steps: Number of orientation steps
            lambda_range: Range of wavelengths (start, end)
            lambda_steps: Number of wavelength steps
            gamma: Aspect ratio (ellipticity)
            psi: Phase offset
        """
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta_range = theta_range
        self.theta_steps = theta_steps
        self.lambda_range = lambda_range
        self.lambda_steps = lambda_steps
        self.gamma = gamma
        self.psi = psi
        self.logger = logging.getLogger(__name__)
        
        # Generate filter bank
        self.filters = self._generate_filter_bank()
        self.logger.info(f"Generated Gabor filter bank with {len(self.filters)} filters")
    
    def _generate_filter_bank(self) -> List[Dict]:
        """Generate the complete Gabor filter bank."""
        filters = []
        
        # Generate orientations
        thetas = np.linspace(self.theta_range[0], self.theta_range[1], 
                           self.theta_steps, endpoint=False)
        
        # Generate wavelengths
        lambdas = np.linspace(self.lambda_range[0], self.lambda_range[1], 
                            self.lambda_steps)
        
        filter_id = 0
        for theta in thetas:
            for lambd in lambdas:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (self.ksize, self.ksize),
                    self.sigma_x,
                    theta,
                    lambd,
                    self.gamma,
                    self.psi,
                    ktype=cv2.CV_32F
                )
                
                # Normalize kernel
                kernel = kernel / (2.0 * np.sum(np.abs(kernel)))
                
                filters.append({
                    'id': filter_id,
                    'kernel': kernel,
                    'theta': theta,
                    'lambda': lambd,
                    'sigma_x': self.sigma_x,
                    'sigma_y': self.sigma_y,
                    'gamma': self.gamma,
                    'psi': self.psi
                })
                filter_id += 1
        
        return filters
    
    def apply_filter_bank(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Apply the complete Gabor filter bank to an image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary with filtered images and analysis results
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale")
        
        # Convert to float32 for processing
        img_float = image.astype(np.float32)
        
        # Store individual filter responses
        responses = []
        orientations = []
        wavelengths = []
        
        # Apply each filter
        for filter_info in self.filters:
            response = cv2.filter2D(img_float, cv2.CV_32F, filter_info['kernel'])
            responses.append(response)
            orientations.append(filter_info['theta'])
            wavelengths.append(filter_info['lambda'])
        
        # Convert to numpy arrays for easier processing
        responses = np.array(responses)
        orientations = np.array(orientations)
        wavelengths = np.array(wavelengths)
        
        # Compute aggregate responses
        results = self._compute_aggregate_responses(responses, orientations, wavelengths)
        
        # Add individual responses
        results['individual_responses'] = responses
        results['filter_info'] = {
            'orientations': orientations,
            'wavelengths': wavelengths,
            'filter_count': len(self.filters)
        }
        
        return results
    
    def _compute_aggregate_responses(self, responses: np.ndarray, 
                                   orientations: np.ndarray, 
                                   wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute aggregate responses from individual filter outputs."""
        results = {}
        
        # Maximum response across all filters
        results['max_response'] = np.max(responses, axis=0)
        
        # Mean response across all filters
        results['mean_response'] = np.mean(responses, axis=0)
        
        # Standard deviation of responses
        results['std_response'] = np.std(responses, axis=0)
        
        # Dominant orientation (orientation of maximum response)
        max_indices = np.argmax(responses, axis=0)
        results['dominant_orientation'] = orientations[max_indices]
        
        # Dominant wavelength
        results['dominant_wavelength'] = wavelengths[max_indices]
        
        # Orientation energy (sum of responses for each orientation)
        unique_orientations = np.unique(orientations)
        orientation_energy = np.zeros((len(unique_orientations), *responses.shape[1:]))
        
        for i, theta in enumerate(unique_orientations):
            theta_mask = np.isclose(orientations, theta, atol=1e-6)
            if np.any(theta_mask):
                orientation_energy[i] = np.sum(responses[theta_mask], axis=0)
        
        results['orientation_energy'] = orientation_energy
        results['unique_orientations'] = unique_orientations
        
        # Wavelength energy (sum of responses for each wavelength)
        unique_wavelengths = np.unique(wavelengths)
        wavelength_energy = np.zeros((len(unique_wavelengths), *responses.shape[1:]))
        
        for i, lambd in enumerate(unique_wavelengths):
            lambda_mask = np.isclose(wavelengths, lambd, atol=1e-6)
            if np.any(lambda_mask):
                wavelength_energy[i] = np.sum(responses[lambda_mask], axis=0)
        
        results['wavelength_energy'] = wavelength_energy
        results['unique_wavelengths'] = unique_wavelengths
        
        # Coherence measure (how consistent the responses are)
        max_orient_energy = np.max(orientation_energy, axis=0)
        sum_orient_energy = np.sum(orientation_energy, axis=0)
        results['orientation_coherence'] = np.divide(max_orient_energy, 
                                                   sum_orient_energy + 1e-10)
        
        # Anisotropy measure (directional variance)
        results['anisotropy'] = self._compute_anisotropy(orientation_energy, unique_orientations)
        
        return results
    
    def _compute_anisotropy(self, orientation_energy: np.ndarray, 
                          orientations: np.ndarray) -> np.ndarray:
        """Compute anisotropy measure from orientation energies."""
        # Convert orientations to complex representation
        complex_orientations = np.exp(2j * orientations)
        
        # Compute weighted mean direction
        weights = orientation_energy
        total_weight = np.sum(weights, axis=0) + 1e-10
        
        # Weighted sum of complex orientations
        weighted_sum = np.zeros(orientation_energy.shape[1:], dtype=complex)
        for i in range(len(orientations)):
            weighted_sum += weights[i] * complex_orientations[i]
        
        # Mean direction strength (anisotropy)
        mean_direction_strength = np.abs(weighted_sum / total_weight)
        
        return mean_direction_strength
    
    def detect_oriented_features(self, image: np.ndarray, 
                                threshold_factor: float = 2.0) -> Dict[str, Any]:
        """
        Detect oriented features (like scratches) using Gabor responses.
        
        Args:
            image: Input grayscale image
            threshold_factor: Factor for automatic thresholding
            
        Returns:
            Dictionary with detected features and orientations
        """
        # Apply filter bank
        gabor_results = self.apply_filter_bank(image)
        
        # Get maximum response
        max_response = gabor_results['max_response']
        orientation_coherence = gabor_results['orientation_coherence']
        dominant_orientation = gabor_results['dominant_orientation']
        
        # Automatic thresholding
        mean_resp = np.mean(max_response)
        std_resp = np.std(max_response)
        threshold = mean_resp + threshold_factor * std_resp
        
        # Create feature mask
        feature_mask = (max_response > threshold).astype(np.uint8) * 255
        
        # Refine using coherence (prefer highly oriented features)
        coherence_threshold = np.percentile(orientation_coherence, 75)
        coherence_mask = (orientation_coherence > coherence_threshold).astype(np.uint8) * 255
        
        # Combine masks
        refined_mask = cv2.bitwise_and(feature_mask, coherence_mask)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        return {
            'feature_mask': refined_mask,
            'raw_feature_mask': feature_mask,
            'coherence_mask': coherence_mask,
            'max_response': max_response,
            'dominant_orientation': dominant_orientation,
            'orientation_coherence': orientation_coherence,
            'threshold_used': threshold
        }
    
    def analyze_texture_properties(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze texture properties using Gabor filter responses.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary with texture properties
        """
        gabor_results = self.apply_filter_bank(image)
        
        # Extract features
        max_response = gabor_results['max_response']
        mean_response = gabor_results['mean_response']
        std_response = gabor_results['std_response']
        orientation_coherence = gabor_results['orientation_coherence']
        anisotropy = gabor_results['anisotropy']
        
        properties = {
            # Response statistics
            'mean_max_response': np.mean(max_response),
            'std_max_response': np.std(max_response),
            'mean_avg_response': np.mean(mean_response),
            'response_contrast': np.mean(std_response),
            
            # Orientation properties
            'mean_coherence': np.mean(orientation_coherence),
            'std_coherence': np.std(orientation_coherence),
            'mean_anisotropy': np.mean(anisotropy),
            'std_anisotropy': np.std(anisotropy),
            
            # Texture regularity
            'texture_energy': np.sum(max_response**2),
            'texture_entropy': self._compute_entropy(max_response),
            'texture_homogeneity': 1.0 / (1.0 + np.var(max_response)),
            
            # Directional properties
            'dominant_orientation_std': np.std(gabor_results['dominant_orientation']),
            'orientation_uniformity': 1.0 - np.std(orientation_coherence)
        }
        
        return properties
    
    def _compute_entropy(self, image: np.ndarray, bins: int = 256) -> float:
        """Compute entropy of image intensities."""
        hist, _ = np.histogram(image.flatten(), bins=bins, range=(image.min(), image.max()))
        hist = hist + 1e-10  # Avoid log(0)
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def visualize_filter_bank(self, save_path: Optional[str] = None):
        """Visualize the Gabor filter bank."""
        n_filters = len(self.filters)
        n_cols = min(8, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, filter_info in enumerate(self.filters):
            row, col = divmod(i, n_cols)
            
            kernel = filter_info['kernel']
            axes[row, col].imshow(kernel, cmap='gray')
            axes[row, col].set_title(f"θ={filter_info['theta']:.2f}\\nλ={filter_info['lambda']:.1f}")
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_filters, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].axis('off')
        
        plt.suptitle('Gabor Filter Bank', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Filter bank visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_gabor_filter_bank():
    """Test the Gabor filter bank with synthetic data."""
    print("Testing Gabor Filter Bank Module")
    print("=" * 40)
    
    # Create test images with different patterns
    def create_oriented_pattern(size=256, orientation=0.0, wavelength=20):
        """Create a sinusoidal pattern with given orientation and wavelength."""
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        
        # Rotate coordinates
        x_rot = x * np.cos(orientation) + y * np.sin(orientation)
        
        # Create sinusoidal pattern
        pattern = np.sin(2 * np.pi * x_rot / wavelength)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, pattern.shape)
        pattern = pattern + noise
        
        # Normalize to 0-255
        pattern = ((pattern + 1) * 127.5).astype(np.uint8)
        
        return pattern
    
    def create_scratch_pattern(size=256):
        """Create a pattern with multiple scratches."""
        image = np.ones((size, size), dtype=np.uint8) * 128
        
        # Add several scratches with different orientations
        scratches = [
            (45, 5, 100),   # (angle_deg, width, length)
            (120, 3, 80),
            (10, 4, 120),
        ]
        
        for angle_deg, width, length in scratches:
            # Create scratch
            scratch = np.zeros((size, size), dtype=np.uint8)
            center = (size//2, size//2)
            
            angle_rad = np.radians(angle_deg)
            start_x = int(center[0] - length//2 * np.cos(angle_rad))
            start_y = int(center[1] - length//2 * np.sin(angle_rad))
            end_x = int(center[0] + length//2 * np.cos(angle_rad))
            end_y = int(center[1] + length//2 * np.sin(angle_rad))
            
            cv2.line(scratch, (start_x, start_y), (end_x, end_y), (255,), width)
            
            # Blend with main image
            image = cv2.subtract(image, (scratch * 0.3).astype(np.uint8))
        
        return image
    
    # Test cases
    test_cases = [
        ("horizontal_lines", create_oriented_pattern(256, 0, 20)),
        ("diagonal_lines", create_oriented_pattern(256, np.pi/4, 15)),
        ("vertical_lines", create_oriented_pattern(256, np.pi/2, 25)),
        ("scratches", create_scratch_pattern(256))
    ]
    
    # Initialize Gabor filter bank
    gabor_bank = GaborFilterBank(
        ksize=31,
        theta_steps=8,
        lambda_steps=4,
        lambda_range=(8, 32)
    )
    
    # Visualize filter bank
    gabor_bank.visualize_filter_bank("gabor_filter_bank.png")
    
    # Test each case
    for case_name, test_image in test_cases:
        print(f"\\nTesting {case_name}...")
        
        # Apply Gabor filter bank
        results = gabor_bank.apply_filter_bank(test_image)
        
        # Detect oriented features
        feature_results = gabor_bank.detect_oriented_features(test_image)
        
        # Analyze texture properties
        texture_props = gabor_bank.analyze_texture_properties(test_image)
        
        print(f"  Detected features: {np.sum(feature_results['feature_mask'] > 0)} pixels")
        print(f"  Mean coherence: {texture_props['mean_coherence']:.3f}")
        print(f"  Mean anisotropy: {texture_props['mean_anisotropy']:.3f}")
        print(f"  Texture energy: {texture_props['texture_energy']:.1f}")
        
        # Visualize results
        visualize_gabor_results(test_image, results, feature_results, 
                              f"gabor_test_{case_name}.png")


def visualize_gabor_results(image: np.ndarray, gabor_results: Dict, 
                          feature_results: Dict, save_path: Optional[str] = None):
    """Visualize Gabor filter results."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Maximum response
    axes[0, 1].imshow(gabor_results['max_response'], cmap='hot')
    axes[0, 1].set_title('Max Response')
    axes[0, 1].axis('off')
    
    # Dominant orientation
    orientation_img = gabor_results['dominant_orientation']
    im = axes[0, 2].imshow(orientation_img, cmap='hsv', vmin=0, vmax=np.pi)
    axes[0, 2].set_title('Dominant Orientation')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Orientation coherence
    axes[0, 3].imshow(gabor_results['orientation_coherence'], cmap='viridis')
    axes[0, 3].set_title('Orientation Coherence')
    axes[0, 3].axis('off')
    
    # Feature mask
    axes[1, 0].imshow(feature_results['feature_mask'], cmap='gray')
    axes[1, 0].set_title('Detected Features')
    axes[1, 0].axis('off')
    
    # Anisotropy
    axes[1, 1].imshow(gabor_results['anisotropy'], cmap='plasma')
    axes[1, 1].set_title('Anisotropy')
    axes[1, 1].axis('off')
    
    # Standard deviation response
    axes[1, 2].imshow(gabor_results['std_response'], cmap='copper')
    axes[1, 2].set_title('Response Std Dev')
    axes[1, 2].axis('off')
    
    # Overlay features on original
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    feature_contours, _ = cv2.findContours(feature_results['feature_mask'], 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, feature_contours, -1, (255, 0, 0), 2)
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title('Features Overlay')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def demo_gabor_filter_bank():
    """Demonstrate the Gabor filter bank capabilities."""
    print("Gabor Filter Bank Demo")
    print("=" * 30)
    
    # Test the module
    test_gabor_filter_bank()
    
    print("\\nDemo completed!")
    print("Generated files:")
    print("  - gabor_filter_bank.png")
    print("  - gabor_test_*.png")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    demo_gabor_filter_bank()
