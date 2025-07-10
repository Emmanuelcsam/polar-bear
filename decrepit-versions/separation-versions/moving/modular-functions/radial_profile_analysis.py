#!/usr/bin/env python3
"""
Radial Profile Analysis Functions
Extracted from multiple fiber optic analysis scripts

This module contains functions for radial profile analysis:
- Radial intensity profile computation
- Gradient analysis along radial profiles
- Peak detection for boundary identification
- Multi-scale radial analysis
- Radial profile smoothing and filtering
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d


def compute_radial_intensity_profile(image: np.ndarray, center: Tuple[int, int], 
                                   max_radius: Optional[int] = None, 
                                   num_angles: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial intensity profile from center outwards.
    
    From: pixel_separation.py, segmentation.py, sergio.py
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates
        max_radius: Maximum radius to analyze (None = auto)
        num_angles: Number of angular samples for averaging
        
    Returns:
        (radii, intensities) arrays
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    center_x, center_y = center
    
    # Calculate max radius if not provided
    if max_radius is None:
        max_radius = int(min(
            center_x, center_y, 
            w - center_x, h - center_y
        ))
    
    # Create radii array
    radii = np.arange(0, max_radius)
    radial_profile = np.zeros(max_radius)
    
    # Sample at multiple angles and average
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    for angle in angles:
        # Calculate points along this radius
        x_coords = center_x + radii * np.cos(angle)
        y_coords = center_y + radii * np.sin(angle)
        
        # Ensure coordinates are within image bounds
        valid_mask = ((x_coords >= 0) & (x_coords < w) & 
                     (y_coords >= 0) & (y_coords < h))
        
        if np.any(valid_mask):
            valid_radii = radii[valid_mask]
            valid_x = x_coords[valid_mask].astype(int)
            valid_y = y_coords[valid_mask].astype(int)
            
            # Add intensities to profile
            radial_profile[valid_radii] += gray[valid_y, valid_x]
    
    # Average by number of valid samples
    radial_profile = radial_profile / num_angles
    
    return radii, radial_profile


def compute_radial_gradient_profile(image: np.ndarray, center: Tuple[int, int], 
                                  max_radius: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial gradient magnitude profile.
    
    From: segmentation.py, fiber_optic_segmentation.py
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates
        max_radius: Maximum radius to analyze
        
    Returns:
        (radii, gradient_magnitudes) arrays
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Compute radial profile of gradient magnitude
    return compute_radial_intensity_profile(gradient_magnitude, center, max_radius)


def smooth_radial_profile(profile: np.ndarray, method: str = 'gaussian', 
                         window_size: int = 5) -> np.ndarray:
    """
    Smooth radial profile using various methods.
    
    From: segmentation.py, pixel_separation.py
    
    Args:
        profile: Input radial intensity profile
        method: Smoothing method ('gaussian', 'savgol', 'median')
        window_size: Size of smoothing window
        
    Returns:
        Smoothed profile
    """
    if len(profile) < window_size:
        return profile.copy()
    
    if method == 'gaussian':
        sigma = window_size / 3.0
        return gaussian_filter1d(profile, sigma)
    
    elif method == 'savgol':
        # Ensure window size is odd and not larger than profile
        window = min(window_size, len(profile))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return profile.copy()
        
        poly_order = min(3, window - 1)
        return savgol_filter(profile, window, poly_order)
    
    elif method == 'median':
        # Apply median filter
        filtered = np.zeros_like(profile)
        half_window = window_size // 2
        
        for i in range(len(profile)):
            start = max(0, i - half_window)
            end = min(len(profile), i + half_window + 1)
            filtered[i] = np.median(profile[start:end])
        
        return filtered
    
    else:
        return profile.copy()


def detect_profile_peaks(profile: np.ndarray, min_prominence: float = 5.0, 
                        min_distance: int = 10) -> Tuple[np.ndarray, Dict]:
    """
    Detect peaks in radial profile for boundary identification.
    
    From: fiber_optic_segmentation.py, segmentation.py
    
    Args:
        profile: Input radial profile
        min_prominence: Minimum peak prominence
        min_distance: Minimum distance between peaks
        
    Returns:
        (peak_indices, peak_properties)
    """
    # Find peaks
    peaks, properties = find_peaks(profile, 
                                 prominence=min_prominence,
                                 distance=min_distance,
                                 width=(None, None),
                                 rel_height=1.0)
    
    return peaks, properties


def detect_gradient_peaks(radii: np.ndarray, gradient_profile: np.ndarray, 
                         smoothing: bool = True) -> List[int]:
    """
    Detect peaks in gradient profile for boundary detection.
    
    From: separation_old2.py, sergio.py
    
    Args:
        radii: Radii array
        gradient_profile: Gradient magnitude profile
        smoothing: Whether to apply smoothing before peak detection
        
    Returns:
        List of peak radii (boundary locations)
    """
    profile = gradient_profile.copy()
    
    if smoothing and len(profile) > 5:
        profile = smooth_radial_profile(profile, method='gaussian', window_size=5)
    
    # Find significant peaks
    peaks, _ = find_peaks(profile, 
                         prominence=np.std(profile),
                         distance=len(profile) // 20)  # Minimum 5% of profile length apart
    
    # Convert peak indices to radii
    peak_radii = [int(radii[peak]) for peak in peaks if peak < len(radii)]
    
    return peak_radii


def analyze_multi_scale_profiles(image: np.ndarray, center: Tuple[int, int], 
                               scales: List[float] = [1.0, 1.5, 2.0]) -> Dict:
    """
    Analyze radial profiles at multiple scales for robust boundary detection.
    
    From: segmentation.py
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates
        scales: List of blur scales to analyze
        
    Returns:
        Dictionary containing profiles and detected boundaries at each scale
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    results = {}
    
    for scale in scales:
        # Apply blur for this scale
        kernel_size = max(3, int(scale * 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), scale)
        
        # Compute profiles
        radii, intensity_profile = compute_radial_intensity_profile(blurred, center)
        _, gradient_profile = compute_radial_gradient_profile(blurred, center)
        
        # Smooth profiles
        smooth_intensity = smooth_radial_profile(intensity_profile)
        smooth_gradient = smooth_radial_profile(gradient_profile)
        
        # Detect boundaries
        gradient_peaks = detect_gradient_peaks(radii, smooth_gradient)
        intensity_peaks, _ = detect_profile_peaks(smooth_intensity)
        
        results[f'scale_{scale}'] = {
            'radii': radii,
            'intensity_profile': intensity_profile,
            'gradient_profile': gradient_profile,
            'smooth_intensity': smooth_intensity,
            'smooth_gradient': smooth_gradient,
            'gradient_peaks': gradient_peaks,
            'intensity_peaks': intensity_peaks.tolist()
        }
    
    return results


def find_fiber_boundaries_from_profile(radii: np.ndarray, intensity_profile: np.ndarray, 
                                     gradient_profile: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """
    Find core and cladding boundaries from radial profiles.
    
    From: sergio.py, separation_old2.py
    
    Args:
        radii: Radii array
        intensity_profile: Radial intensity profile
        gradient_profile: Optional gradient profile for refinement
        
    Returns:
        (core_radius, cladding_radius)
    """
    # Smooth the profiles
    smooth_intensity = smooth_radial_profile(intensity_profile, method='gaussian', window_size=5)
    
    if gradient_profile is not None:
        smooth_gradient = smooth_radial_profile(gradient_profile, method='gaussian', window_size=5)
        gradient_peaks = detect_gradient_peaks(radii, smooth_gradient)
    else:
        gradient_peaks = []
    
    # Find intensity-based boundaries
    # Core boundary: first significant drop in intensity
    intensity_diff = np.diff(smooth_intensity)
    core_candidates = np.where(intensity_diff < -np.std(intensity_diff))[0]
    
    # Cladding boundary: look for second major transition
    cladding_candidates = []
    if len(core_candidates) > 0:
        core_estimate = core_candidates[0]
        # Look for boundaries beyond the core
        later_drops = np.where((intensity_diff < -np.std(intensity_diff)) & 
                              (np.arange(len(intensity_diff)) > core_estimate + 10))[0]
        if len(later_drops) > 0:
            cladding_candidates = later_drops
    
    # Combine with gradient-based boundaries
    all_boundaries = []
    if len(core_candidates) > 0:
        all_boundaries.append(radii[core_candidates[0]])
    if len(cladding_candidates) > 0:
        all_boundaries.append(radii[cladding_candidates[0]])
    
    # Add gradient peaks
    all_boundaries.extend(gradient_peaks)
    
    # Sort and select best boundaries
    all_boundaries = sorted(set(all_boundaries))
    
    # Default values
    max_radius = len(radii) - 1
    core_radius = min(max_radius // 4, 50)  # Default core radius
    cladding_radius = min(max_radius // 2, 100)  # Default cladding radius
    
    if len(all_boundaries) >= 2:
        core_radius = min(all_boundaries[0], max_radius)
        cladding_radius = min(all_boundaries[1], max_radius)
    elif len(all_boundaries) == 1:
        core_radius = min(all_boundaries[0], max_radius)
        cladding_radius = min(int(core_radius * 1.5), max_radius)
    
    return core_radius, cladding_radius


def compute_directional_profiles(image: np.ndarray, center: Tuple[int, int], 
                               angles: List[float]) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute radial profiles at specific angles.
    
    Args:
        image: Input grayscale image
        center: (x, y) center coordinates
        angles: List of angles in radians
        
    Returns:
        Dictionary mapping angles to (radii, intensities)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    center_x, center_y = center
    
    max_radius = int(min(center_x, center_y, w - center_x, h - center_y))
    radii = np.arange(0, max_radius)
    
    profiles = {}
    
    for angle in angles:
        # Calculate points along this radius
        x_coords = center_x + radii * np.cos(angle)
        y_coords = center_y + radii * np.sin(angle)
        
        # Ensure coordinates are within image bounds
        valid_mask = ((x_coords >= 0) & (x_coords < w) & 
                     (y_coords >= 0) & (y_coords < h))
        
        intensities = np.zeros_like(radii, dtype=float)
        
        if np.any(valid_mask):
            valid_radii = radii[valid_mask]
            valid_x = x_coords[valid_mask].astype(int)
            valid_y = y_coords[valid_mask].astype(int)
            
            intensities[valid_radii] = gray[valid_y, valid_x]
        
        profiles[angle] = (radii, intensities)
    
    return profiles


def main():
    """Test the radial profile analysis functions"""
    # Create a test image with concentric circular structures
    test_image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add background
    test_image[:] = 100
    
    # Add circular structures with different intensities
    center = (200, 200)
    cv2.circle(test_image, center, 60, (200,), -1)   # Bright core
    cv2.circle(test_image, center, 100, (150,), 20)  # Cladding ring
    cv2.circle(test_image, center, 150, (80,), -1)   # Outer region
    
    print("Testing Radial Profile Analysis Functions...")
    
    # Test radial intensity profile
    radii, intensity_profile = compute_radial_intensity_profile(test_image, center)
    print(f"✓ Radial intensity profile: {len(intensity_profile)} points")
    
    # Test gradient profile
    grad_radii, gradient_profile = compute_radial_gradient_profile(test_image, center)
    print(f"✓ Gradient profile: {len(gradient_profile)} points")
    
    # Test profile smoothing
    smooth_intensity = smooth_radial_profile(intensity_profile, 'gaussian')
    smooth_gradient = smooth_radial_profile(gradient_profile, 'savgol')
    print(f"✓ Profile smoothing: gaussian and savgol")
    
    # Test peak detection
    peaks, properties = detect_profile_peaks(smooth_gradient)
    print(f"✓ Detected {len(peaks)} peaks in gradient profile")
    
    # Test gradient peak detection
    gradient_peaks = detect_gradient_peaks(grad_radii, gradient_profile)
    print(f"✓ Gradient-based peaks: {gradient_peaks}")
    
    # Test boundary detection
    core_radius, cladding_radius = find_fiber_boundaries_from_profile(
        radii, intensity_profile, gradient_profile)
    print(f"✓ Detected boundaries - Core: {core_radius}, Cladding: {cladding_radius}")
    
    # Test multi-scale analysis
    multi_scale_results = analyze_multi_scale_profiles(test_image, center)
    print(f"✓ Multi-scale analysis: {len(multi_scale_results)} scales")
    
    # Test directional profiles
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    directional_profiles = compute_directional_profiles(test_image, center, angles)
    print(f"✓ Directional profiles: {len(directional_profiles)} angles")
    
    print("\nAll radial profile analysis functions tested successfully!")


if __name__ == "__main__":
    main()
