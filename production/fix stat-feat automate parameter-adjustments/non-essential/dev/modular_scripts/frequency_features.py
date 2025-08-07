#!/usr/bin/env python3
"""
Frequency domain feature extraction module using FFT.
Analyzes images in frequency space for pattern detection.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List


def compute_fft_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Extract 2D Fourier Transform features from grayscale image.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Dictionary of FFT-based features
    """
    # Compute 2D FFT
    f = np.fft.fft2(gray)
    
    # Shift zero frequency to center
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fshift)
    
    # Compute power spectrum
    power = magnitude**2
    
    # Compute phase spectrum
    phase = np.angle(fshift)
    
    # Calculate center coordinates
    center = np.array(power.shape) // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    
    # Compute distance from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Compute radial profile
    radial_prof = compute_radial_profile(power, center)
    
    # Compute spectral features
    if len(radial_prof) > 0:
        # Weighted average of frequencies
        freqs = np.arange(len(radial_prof))
        spectral_centroid = float(np.sum(freqs * radial_prof) / (np.sum(radial_prof) + 1e-10))
        
        # Weighted standard deviation
        spectral_spread = float(np.sqrt(
            np.sum((freqs - spectral_centroid)**2 * radial_prof) / (np.sum(radial_prof) + 1e-10)
        ))
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
        'fft_high_freq_ratio': compute_high_freq_ratio(power, center)
    }


def compute_radial_profile(power: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Compute radial power profile from FFT power spectrum.
    
    Args:
        power: Power spectrum (2D array)
        center: Center coordinates
        
    Returns:
        Radial profile array
    """
    # Create coordinate grids
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    
    # Compute distance from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Compute radial profile
    radial_prof = []
    max_radius = min(center)
    
    for radius in range(1, max_radius):
        # Create ring mask
        mask = (r >= radius - 1) & (r < radius)
        
        # Average power in ring
        if mask.any():
            radial_prof.append(np.mean(power[mask]))
    
    return np.array(radial_prof)


def compute_high_freq_ratio(power: np.ndarray, center: np.ndarray) -> float:
    """
    Compute ratio of high frequency to total power.
    
    Args:
        power: Power spectrum
        center: Center coordinates
        
    Returns:
        High frequency ratio (0-1)
    """
    # Create coordinate grids
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    
    # Compute distance from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Define high frequency region (outer half)
    max_radius = min(center)
    high_freq_mask = r > max_radius / 2
    
    # Compute ratio
    high_freq_power = np.sum(power[high_freq_mask])
    total_power = np.sum(power)
    
    return float(high_freq_power / (total_power + 1e-10))


def apply_frequency_filter(gray: np.ndarray, 
                          filter_type: str = 'lowpass',
                          cutoff: float = 0.3) -> np.ndarray:
    """
    Apply frequency domain filter to image.
    
    Args:
        gray: Grayscale image (uint8)
        filter_type: 'lowpass', 'highpass', or 'bandpass'
        cutoff: Cutoff frequency (0-1, fraction of max frequency)
        
    Returns:
        Filtered image
    """
    # Compute FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Get dimensions
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create filter mask
    mask = create_frequency_mask(gray.shape, filter_type, cutoff)
    
    # Apply filter
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_filtered)
    img_filtered = np.abs(img_filtered)
    
    # Normalize to uint8
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    return img_filtered


def create_frequency_mask(shape: Tuple[int, int], 
                         filter_type: str,
                         cutoff: float) -> np.ndarray:
    """
    Create frequency domain filter mask.
    
    Args:
        shape: Image shape (height, width)
        filter_type: 'lowpass', 'highpass', or 'bandpass'
        cutoff: Cutoff frequency (0-1)
        
    Returns:
        Filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]
    
    # Distance from center
    d = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Normalize distance
    max_d = np.sqrt(crow**2 + ccol**2)
    d_norm = d / max_d
    
    # Create mask based on filter type
    if filter_type == 'lowpass':
        mask = (d_norm <= cutoff).astype(float)
    elif filter_type == 'highpass':
        mask = (d_norm > cutoff).astype(float)
    elif filter_type == 'bandpass':
        # Bandpass between cutoff/2 and cutoff
        mask = ((d_norm > cutoff/2) & (d_norm <= cutoff)).astype(float)
    else:
        mask = np.ones(shape)
    
    return mask


def detect_periodic_patterns(gray: np.ndarray, 
                            threshold: float = 0.8) -> List[Tuple[int, int]]:
    """
    Detect periodic patterns using FFT peak detection.
    
    Args:
        gray: Grayscale image (uint8)
        threshold: Threshold for peak detection (0-1)
        
    Returns:
        List of (frequency_x, frequency_y) peaks
    """
    # Compute FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fshift)
    
    # Normalize magnitude
    mag_norm = magnitude / np.max(magnitude)
    
    # Find peaks above threshold
    peaks_mask = mag_norm > threshold
    
    # Get center
    center = np.array(magnitude.shape) // 2
    
    # Exclude DC component
    peaks_mask[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = False
    
    # Find peak coordinates
    peaks_y, peaks_x = np.where(peaks_mask)
    
    # Convert to frequency coordinates relative to center
    freq_peaks = []
    for y, x in zip(peaks_y, peaks_x):
        freq_x = x - center[1]
        freq_y = y - center[0]
        freq_peaks.append((freq_x, freq_y))
    
    return freq_peaks


def visualize_frequency_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Create visualization of frequency spectrum.
    
    Args:
        gray: Grayscale image (uint8)
        
    Returns:
        Visualization image (uint8)
    """
    # Compute FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fshift)
    
    # Log transform for visualization
    magnitude_log = np.log(magnitude + 1)
    
    # Normalize to 0-255
    magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude_norm.astype(np.uint8)


def main():
    """Standalone test function."""
    print("Frequency Features Module - Standalone Test")
    print("-" * 40)
    
    # Create test image with patterns
    test_image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add sinusoidal pattern (creates peaks in frequency domain)
    x = np.arange(256)
    y = np.arange(256)
    X, Y = np.meshgrid(x, y)
    
    # Horizontal stripes
    pattern1 = np.sin(2 * np.pi * Y / 20) * 50 + 128
    
    # Diagonal stripes
    pattern2 = np.sin(2 * np.pi * (X + Y) / 30) * 30
    
    test_image = np.clip(pattern1 + pattern2, 0, 255).astype(np.uint8)
    
    # Add some noise
    noise = np.random.randint(-10, 10, test_image.shape)
    test_image = np.clip(test_image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    print("Extracting frequency features...")
    features = compute_fft_features(test_image)
    
    print("\nFrequency Domain Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")
    
    # Detect periodic patterns
    print("\nDetecting periodic patterns...")
    peaks = detect_periodic_patterns(test_image, threshold=0.3)
    print(f"Found {len(peaks)} frequency peaks:")
    for i, (fx, fy) in enumerate(peaks[:5], 1):
        print(f"  Peak {i}: frequency=({fx}, {fy})")
    
    # Apply filters
    print("\nApplying frequency filters...")
    
    # Low-pass filter
    lowpass = apply_frequency_filter(test_image, 'lowpass', 0.2)
    cv2.imwrite("freq_lowpass_test.png", lowpass)
    print("  Low-pass filtered image saved")
    
    # High-pass filter
    highpass = apply_frequency_filter(test_image, 'highpass', 0.1)
    cv2.imwrite("freq_highpass_test.png", highpass)
    print("  High-pass filtered image saved")
    
    # Visualize spectrum
    spectrum_viz = visualize_frequency_spectrum(test_image)
    cv2.imwrite("freq_spectrum_test.png", spectrum_viz)
    print("  Frequency spectrum saved")
    
    # Save original for comparison
    cv2.imwrite("freq_original_test.png", test_image)
    print("  Original image saved")


if __name__ == "__main__":
    main()
