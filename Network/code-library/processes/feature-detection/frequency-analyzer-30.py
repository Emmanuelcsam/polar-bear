#!/usr/bin/env python3
"""
Frequency Domain Analysis Module for Fiber Optic Defect Detection
=================================================================

This module implements frequency domain methods for detecting defects in fiber
optic end-face images. Extracted from advanced_defect_analysis.py and defect_analysis.py.

Functions:
- FFT-based high-frequency defect detection
- Wavelet analysis
- Gabor filter banks
- Phase congruency analysis
- Spectral analysis

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
try:
    from skimage import filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some filters will use manual implementations.")
    SKIMAGE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def fft_highpass_detection(image, radius=30, mask=None):
    """
    High-frequency defect detection using FFT and high-pass filtering.
    
    Args:
        image (np.ndarray): Input grayscale image
        radius (int): Radius for high-pass filter
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: FFT results and detected high-frequency defects
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Compute FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Create high-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create circular high-pass mask
    y, x = np.ogrid[:rows, :cols]
    center_mask = (x - ccol)**2 + (y - crow)**2 <= radius**2
    
    hp_mask = np.ones((rows, cols), dtype=np.uint8)
    hp_mask[center_mask] = 0
    
    # Apply high-pass filter
    f_shift_hp = f_shift * hp_mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(f_shift_hp)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-10)
    img_back = (img_back * 255).astype(np.uint8)
    
    # Threshold high-frequency components
    if np.sum(mask) > 0:
        threshold = np.percentile(img_back[mask], 95)
    else:
        threshold = np.percentile(img_back, 95)
    
    defects = (img_back > threshold) & mask
    
    return {
        'magnitude_spectrum': magnitude_spectrum,
        'highpass_mask': hp_mask,
        'filtered_magnitude': np.abs(f_shift_hp),
        'reconstructed': img_back,
        'defects': defects,
        'threshold': threshold,
        'filter_radius': radius
    }


def wavelet_edge_detection(image, scales=[1, 2, 4], mask=None):
    """
    Wavelet-based edge detection using Sobel-like wavelets.
    
    Args:
        image (np.ndarray): Input grayscale image
        scales (list): List of scales for multi-scale analysis
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Wavelet responses and edge maps
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    
    scale_responses = {}
    combined_edges = np.zeros_like(img_float)
    
    for scale in scales:
        # Gaussian smoothing at current scale
        smoothed = ndimage.gaussian_filter(img_float, scale)
        
        # Compute gradients (wavelet-like operators)
        grad_x = ndimage.sobel(smoothed, axis=1)
        grad_y = ndimage.sobel(smoothed, axis=0)
        
        # Magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Scale normalization
        magnitude *= scale**2
        
        scale_responses[f'scale_{scale}'] = {
            'magnitude': magnitude,
            'direction': direction,
            'grad_x': grad_x,
            'grad_y': grad_y
        }
        
        # Add to combined response
        combined_edges = np.maximum(combined_edges, magnitude)
    
    # Threshold combined edges
    if np.sum(mask) > 0:
        threshold = np.percentile(combined_edges[mask], 95)
    else:
        threshold = np.percentile(combined_edges, 95)
    
    edge_defects = (combined_edges > threshold) & mask
    
    return {
        'scale_responses': scale_responses,
        'combined_edges': combined_edges,
        'edge_defects': edge_defects,
        'threshold': threshold,
        'scales': scales
    }


def gabor_filter_bank(image, frequencies=[0.1, 0.3, 0.5], orientations=[0, 45, 90, 135], mask=None):
    """
    Gabor filter bank for oriented texture and defect detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        frequencies (list): Spatial frequencies
        orientations (list): Orientations in degrees
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Gabor responses and combined defect map
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    responses = {}
    combined_response = np.zeros_like(img_float)
    
    for freq in frequencies:
        for orientation in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (31, 31),  # Kernel size
                3.0,       # Standard deviation
                np.radians(orientation),  # Orientation
                2 * np.pi * freq,        # Wavelength
                0.5,       # Aspect ratio
                0,         # Phase offset
                ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered = cv2.filter2D(img_float, cv2.CV_32F, kernel)
            filtered = np.abs(filtered)
            
            key = f'freq_{freq:.1f}_orient_{orientation}'
            responses[key] = filtered
            
            # Add to combined response
            combined_response = np.maximum(combined_response, filtered)
    
    # Threshold for defect detection
    if np.sum(mask) > 0:
        threshold = np.percentile(combined_response[mask], 95)
    else:
        threshold = np.percentile(combined_response, 95)
    
    gabor_defects = (combined_response > threshold) & mask
    
    return {
        'responses': responses,
        'combined_response': combined_response,
        'gabor_defects': gabor_defects,
        'threshold': threshold,
        'frequencies': frequencies,
        'orientations': orientations
    }


def phase_congruency_analysis(image, nscale=4, norient=6, mask=None):
    """
    Phase congruency for feature detection using log-Gabor filters.
    
    Args:
        image (np.ndarray): Input grayscale image
        nscale (int): Number of scales
        norient (int): Number of orientations
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Phase congruency map and feature points
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    rows, cols = image.shape
    img_float = image.astype(np.float64)
    
    # Fourier transform
    IM = np.fft.fft2(img_float)
    
    # Initialize phase congruency
    PC = np.zeros((rows, cols))
    
    # Frequency coordinates
    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1  # Avoid division by zero
    
    # Log-Gabor filter parameters
    wavelength = 6
    
    for s in range(nscale):
        lambda_s = wavelength * (2**s)
        fo = 1.0 / lambda_s
        
        # Log-Gabor radial component
        logGabor = np.exp(-(np.log(radius/fo))**2 / (2 * np.log(0.65)**2))
        logGabor[radius < fo/3] = 0
        
        for o in range(norient):
            angle = o * np.pi / norient
            
            # Angular component
            theta = np.arctan2(v, u)
            ds = np.sin(theta - angle)
            spread = np.pi / norient / 1.5
            
            angular = np.exp(-(ds**2) / (2 * spread**2))
            
            # Combined filter
            filter_bank = logGabor * angular
            
            # Apply filter
            response = np.fft.ifft2(IM * filter_bank)
            
            # Phase congruency calculation
            magnitude = np.abs(response)
            phase = np.angle(response)
            
            # Add to phase congruency (simplified version)
            PC += magnitude * np.cos(phase - np.mean(phase))
    
    # Normalize
    PC = PC / (nscale * norient)
    PC = (PC - PC.min()) / (PC.max() - PC.min() + 1e-10)
    
    # Threshold for feature detection
    if np.sum(mask) > 0:
        threshold = np.percentile(PC[mask], 95)
    else:
        threshold = np.percentile(PC, 95)
    
    features = (PC > threshold) & mask
    
    return {
        'phase_congruency': PC,
        'features': features,
        'threshold': threshold,
        'nscale': nscale,
        'norient': norient
    }


def spectral_analysis(image, mask=None):
    """
    Spectral analysis to characterize frequency content.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Spectral features and statistics
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[~mask] = np.mean(image[mask]) if np.sum(mask) > 0 else 128
    
    # Compute FFT
    f_transform = np.fft.fft2(masked_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    power_spectrum = magnitude_spectrum**2
    
    # Radial frequency analysis
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    
    # Create radial coordinate system
    y, x = np.ogrid[:rows, :cols]
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Radial averaging
    max_radius = int(min(rows, cols) // 2)
    radial_profile = np.zeros(max_radius)
    
    for r in range(max_radius):
        mask_ring = (radius >= r) & (radius < r + 1)
        if np.sum(mask_ring) > 0:
            radial_profile[r] = np.mean(power_spectrum[mask_ring])
    
    # Spectral features
    total_power = np.sum(power_spectrum)
    low_freq_power = np.sum(radial_profile[:max_radius//4])
    mid_freq_power = np.sum(radial_profile[max_radius//4:max_radius//2])
    high_freq_power = np.sum(radial_profile[max_radius//2:])
    
    # Normalize
    low_freq_ratio = low_freq_power / total_power if total_power > 0 else 0
    mid_freq_ratio = mid_freq_power / total_power if total_power > 0 else 0
    high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0
    
    # Spectral centroid (frequency center of mass)
    frequencies = np.arange(max_radius)
    spectral_centroid = np.sum(frequencies * radial_profile) / np.sum(radial_profile) if np.sum(radial_profile) > 0 else 0
    
    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(((frequencies - spectral_centroid)**2) * radial_profile) / np.sum(radial_profile)
    ) if np.sum(radial_profile) > 0 else 0
    
    return {
        'magnitude_spectrum': magnitude_spectrum,
        'power_spectrum': power_spectrum,
        'radial_profile': radial_profile,
        'total_power': total_power,
        'low_freq_ratio': low_freq_ratio,
        'mid_freq_ratio': mid_freq_ratio,
        'high_freq_ratio': high_freq_ratio,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth
    }


def comprehensive_frequency_analysis(image, mask=None):
    """
    Comprehensive frequency domain analysis combining multiple methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Combined frequency analysis results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    print("Running comprehensive frequency analysis...")
    
    # 1. FFT high-pass detection
    print("  - FFT high-pass filtering...")
    fft_result = fft_highpass_detection(image, mask=mask)
    results['fft_highpass'] = fft_result
    
    # 2. Wavelet edge detection
    print("  - Wavelet edge detection...")
    wavelet_result = wavelet_edge_detection(image, mask=mask)
    results['wavelet_edges'] = wavelet_result
    
    # 3. Gabor filter bank
    print("  - Gabor filter bank...")
    gabor_result = gabor_filter_bank(image, mask=mask)
    results['gabor'] = gabor_result
    
    # 4. Phase congruency
    print("  - Phase congruency analysis...")
    phase_result = phase_congruency_analysis(image, mask=mask)
    results['phase_congruency'] = phase_result
    
    # 5. Spectral analysis
    print("  - Spectral analysis...")
    spectral_result = spectral_analysis(image, mask=mask)
    results['spectral'] = spectral_result
    
    # Combine all frequency-based defect detections
    combined_defects = np.zeros_like(mask, dtype=bool)
    combined_defects |= fft_result['defects']
    combined_defects |= wavelet_result['edge_defects']
    combined_defects |= gabor_result['gabor_defects']
    combined_defects |= phase_result['features']
    
    results['combined_defects'] = combined_defects
    results['defect_count'] = np.sum(combined_defects)
    results['defect_percentage'] = (np.sum(combined_defects) / np.sum(mask) * 100) if np.sum(mask) > 0 else 0
    
    return results


def visualize_frequency_results(image, results, save_path=None):
    """
    Visualize frequency domain analysis results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from frequency analysis
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Frequency Domain Analysis Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # FFT magnitude spectrum
    if 'fft_highpass' in results:
        magnitude = results['fft_highpass']['magnitude_spectrum']
        axes[0, 1].imshow(np.log(magnitude + 1), cmap='gray')
        axes[0, 1].set_title('FFT Magnitude Spectrum')
        axes[0, 1].axis('off')
    
    # FFT high-pass defects
    if 'fft_highpass' in results:
        axes[0, 2].imshow(results['fft_highpass']['defects'], cmap='hot')
        count = np.sum(results['fft_highpass']['defects'])
        axes[0, 2].set_title(f'FFT High-pass Defects ({count})')
        axes[0, 2].axis('off')
    
    # Wavelet edges
    if 'wavelet_edges' in results:
        axes[1, 0].imshow(results['wavelet_edges']['combined_edges'], cmap='viridis')
        axes[1, 0].set_title('Wavelet Edge Map')
        axes[1, 0].axis('off')
    
    # Gabor response
    if 'gabor' in results:
        axes[1, 1].imshow(results['gabor']['combined_response'], cmap='viridis')
        axes[1, 1].set_title('Gabor Combined Response')
        axes[1, 1].axis('off')
    
    # Phase congruency
    if 'phase_congruency' in results:
        axes[1, 2].imshow(results['phase_congruency']['phase_congruency'], cmap='viridis')
        axes[1, 2].set_title('Phase Congruency')
        axes[1, 2].axis('off')
    
    # Radial power spectrum
    if 'spectral' in results:
        radial_profile = results['spectral']['radial_profile']
        axes[2, 0].plot(radial_profile)
        axes[2, 0].set_title('Radial Power Spectrum')
        axes[2, 0].set_xlabel('Frequency')
        axes[2, 0].set_ylabel('Power')
    
    # Combined defects
    if 'combined_defects' in results:
        axes[2, 1].imshow(results['combined_defects'], cmap='hot')
        count = results['defect_count']
        percentage = results['defect_percentage']
        axes[2, 1].set_title(f'Combined Defects\n{count} pixels ({percentage:.2f}%)')
        axes[2, 1].axis('off')
    
    # Spectral features
    if 'spectral' in results:
        spectral = results['spectral']
        feature_text = f"Low freq: {spectral['low_freq_ratio']:.3f}\n"
        feature_text += f"Mid freq: {spectral['mid_freq_ratio']:.3f}\n"
        feature_text += f"High freq: {spectral['high_freq_ratio']:.3f}\n"
        feature_text += f"Centroid: {spectral['spectral_centroid']:.1f}\n"
        feature_text += f"Bandwidth: {spectral['spectral_bandwidth']:.1f}"
        
        axes[2, 2].text(0.1, 0.5, feature_text, fontsize=12, verticalalignment='center')
        axes[2, 2].set_title('Spectral Features')
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of frequency domain analysis functions.
    """
    # Create a test image with various frequency components
    test_image = np.random.normal(128, 10, (200, 200)).astype(np.uint8)
    
    # Add high-frequency defects (sharp edges)
    test_image[50:60, 50:60] = 255  # Bright square
    test_image[100, 50:150] = 0     # Dark line
    
    # Add periodic pattern
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, 200), np.linspace(0, 4*np.pi, 200))
    periodic = 20 * np.sin(5*x) * np.cos(5*y)
    test_image = np.clip(test_image.astype(float) + periodic, 0, 255).astype(np.uint8)
    
    print("Testing Frequency Domain Analysis Module")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_frequency_analysis(test_image)
    
    # Print summary
    print(f"\nFrequency Analysis Summary:")
    print(f"FFT high-pass defects: {np.sum(results['fft_highpass']['defects'])}")
    print(f"Wavelet edge defects: {np.sum(results['wavelet_edges']['edge_defects'])}")
    print(f"Gabor defects: {np.sum(results['gabor']['gabor_defects'])}")
    print(f"Phase congruency features: {np.sum(results['phase_congruency']['features'])}")
    print(f"Total combined defects: {results['defect_count']} ({results['defect_percentage']:.2f}%)")
    
    spectral = results['spectral']
    print(f"\nSpectral Features:")
    print(f"Low frequency ratio: {spectral['low_freq_ratio']:.3f}")
    print(f"Mid frequency ratio: {spectral['mid_freq_ratio']:.3f}")
    print(f"High frequency ratio: {spectral['high_freq_ratio']:.3f}")
    print(f"Spectral centroid: {spectral['spectral_centroid']:.1f}")
    print(f"Spectral bandwidth: {spectral['spectral_bandwidth']:.1f}")
    
    # Visualize results
    visualize_frequency_results(test_image, results, 'frequency_analysis_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
