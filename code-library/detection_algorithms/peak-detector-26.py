#!/usr/bin/env python3
"""
Peak Detection and Signal Processing Functions
Extracted from multiple fiber optic analysis scripts

This module contains functions for signal processing and peak detection:
- Histogram analysis and peak finding
- Adaptive peak detection with various algorithms
- Signal smoothing and filtering
- Boundary detection from signal peaks
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


def calculate_histogram_peaks(image: np.ndarray, bins: int = 256, 
                             prominence: float = 100.0) -> Tuple[List[int], Dict]:
    """
    Calculate histogram and find peaks for intensity-based segmentation.
    
    From: fiber_optic_segmentation.py, sam.py
    
    Args:
        image: Input grayscale image
        bins: Number of histogram bins
        prominence: Minimum peak prominence
        
    Returns:
        (peak_positions, peak_properties)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate histogram
    histogram = cv2.calcHist([gray], [0], None, [bins], [0, bins])
    histogram = histogram.flatten()
    
    # Simple peak detection (finding local maxima)
    peaks = []
    peak_properties = {'left_ips': [], 'right_ips': [], 'prominences': []}
    
    # Find local maxima with minimum prominence
    for i in range(1, len(histogram) - 1):
        if (histogram[i] > histogram[i-1] and 
            histogram[i] > histogram[i+1] and 
            histogram[i] > prominence):
            
            peaks.append(i)
            
            # Find left and right boundaries (simplified)
            left_bound = i
            while left_bound > 0 and histogram[left_bound] > histogram[i] / 2:
                left_bound -= 1
            
            right_bound = i
            while right_bound < len(histogram) - 1 and histogram[right_bound] > histogram[i] / 2:
                right_bound += 1
            
            peak_properties['left_ips'].append(left_bound)
            peak_properties['right_ips'].append(right_bound)
            peak_properties['prominences'].append(histogram[i])
    
    return peaks, peak_properties


def adaptive_threshold_peaks(data: np.ndarray, adaptive_factor: float = 1.5) -> List[int]:
    """
    Find peaks using adaptive threshold based on data statistics.
    
    Args:
        data: 1D array of data
        adaptive_factor: Factor for adaptive threshold calculation
        
    Returns:
        List of peak indices
    """
    if len(data) < 3:
        return []
    
    # Calculate adaptive threshold
    mean_val = np.mean(data)
    std_val = np.std(data)
    threshold = mean_val + adaptive_factor * std_val
    
    peaks = []
    
    # Find local maxima above threshold
    for i in range(1, len(data) - 1):
        if (data[i] > data[i-1] and 
            data[i] > data[i+1] and 
            data[i] > threshold):
            peaks.append(i)
    
    return peaks


def gradient_peak_detection(data: np.ndarray, min_prominence: float = 0.1) -> List[int]:
    """
    Detect peaks in gradient data for boundary identification.
    
    From: separation_old2.py, gradient_approach.py
    
    Args:
        data: 1D gradient magnitude data
        min_prominence: Minimum relative prominence (0-1)
        
    Returns:
        List of peak indices
    """
    if len(data) < 5:
        return []
    
    # Smooth the data slightly
    smoothed = np.convolve(data, np.ones(3)/3, mode='same')
    
    # Calculate prominence threshold
    data_range = np.max(smoothed) - np.min(smoothed)
    prominence_threshold = min_prominence * data_range
    
    peaks = []
    
    # Find peaks with minimum prominence
    for i in range(2, len(smoothed) - 2):
        if (smoothed[i] > smoothed[i-1] and 
            smoothed[i] > smoothed[i+1] and
            smoothed[i] > smoothed[i-2] and 
            smoothed[i] > smoothed[i+2]):
            
            # Check prominence
            left_min = np.min(smoothed[max(0, i-10):i])
            right_min = np.min(smoothed[i:min(len(smoothed), i+10)])
            prominence = smoothed[i] - max(left_min, right_min)
            
            if prominence > prominence_threshold:
                peaks.append(i)
    
    return peaks


def multi_scale_peak_detection(data: np.ndarray, scales: List[int] = [1, 3, 5]) -> Dict[str, List[int]]:
    """
    Detect peaks at multiple scales for robust identification.
    
    Args:
        data: 1D input data
        scales: List of smoothing scales
        
    Returns:
        Dictionary with peaks detected at each scale
    """
    results = {}
    
    for scale in scales:
        if scale == 1:
            smoothed = data.copy()
        else:
            # Simple moving average smoothing
            kernel = np.ones(scale) / scale
            smoothed = np.convolve(data, kernel, mode='same')
        
        peaks = adaptive_threshold_peaks(smoothed)
        results[f'scale_{scale}'] = peaks
    
    return results


def consensus_peak_detection(multi_scale_results: Dict[str, List[int]], 
                           tolerance: int = 2) -> List[int]:
    """
    Find consensus peaks across multiple scales.
    
    Args:
        multi_scale_results: Results from multi_scale_peak_detection
        tolerance: Maximum distance for peak matching
        
    Returns:
        List of consensus peak positions
    """
    all_peaks = []
    
    # Collect all peaks
    for scale_peaks in multi_scale_results.values():
        all_peaks.extend(scale_peaks)
    
    if not all_peaks:
        return []
    
    # Sort peaks
    all_peaks = sorted(all_peaks)
    
    # Group nearby peaks
    consensus_peaks = []
    current_group = [all_peaks[0]]
    
    for peak in all_peaks[1:]:
        if peak - current_group[-1] <= tolerance:
            current_group.append(peak)
        else:
            # Finalize current group
            if len(current_group) >= 2:  # At least 2 scales agree
                consensus_peaks.append(int(np.median(current_group)))
            current_group = [peak]
    
    # Don't forget the last group
    if len(current_group) >= 2:
        consensus_peaks.append(int(np.median(current_group)))
    
    return consensus_peaks


def moving_average_smoothing(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing to data.
    
    Args:
        data: Input 1D data
        window_size: Size of moving average window
        
    Returns:
        Smoothed data
    """
    if window_size <= 1 or len(data) < window_size:
        return data.copy()
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential smoothing to data.
    
    Args:
        data: Input 1D data
        alpha: Smoothing factor (0-1)
        
    Returns:
        Smoothed data
    """
    if len(data) == 0:
        return data.copy()
    
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def detect_step_changes(data: np.ndarray, min_step_size: float = 0.1) -> List[int]:
    """
    Detect step changes in data for boundary identification.
    
    Args:
        data: Input 1D data
        min_step_size: Minimum relative step size (0-1)
        
    Returns:
        List of step change positions
    """
    if len(data) < 5:
        return []
    
    # Calculate first difference
    diff = np.diff(data)
    
    # Calculate threshold for significant steps
    data_range = np.max(data) - np.min(data)
    step_threshold = min_step_size * data_range
    
    # Find significant steps
    step_positions = []
    
    for i in range(len(diff)):
        if abs(diff[i]) > step_threshold:
            step_positions.append(i)
    
    return step_positions


def find_intensity_boundaries(intensity_profile: np.ndarray, 
                            gradient_profile: Optional[np.ndarray] = None) -> Tuple[List[int], List[int]]:
    """
    Find intensity boundaries combining intensity and gradient analysis.
    
    From: sergio.py, separation_old2.py
    
    Args:
        intensity_profile: Radial intensity profile
        gradient_profile: Optional gradient profile
        
    Returns:
        (intensity_boundaries, gradient_boundaries)
    """
    # Smooth profiles
    smooth_intensity = moving_average_smoothing(intensity_profile, window_size=5)
    
    # Find intensity-based boundaries
    intensity_boundaries = detect_step_changes(smooth_intensity, min_step_size=0.1)
    
    # Find gradient-based boundaries if provided
    gradient_boundaries = []
    if gradient_profile is not None:
        smooth_gradient = moving_average_smoothing(gradient_profile, window_size=3)
        gradient_boundaries = gradient_peak_detection(smooth_gradient)
    
    return intensity_boundaries, gradient_boundaries


def analyze_signal_quality(data: np.ndarray) -> Dict[str, float]:
    """
    Analyze signal quality metrics for peak detection reliability.
    
    Args:
        data: Input 1D signal data
        
    Returns:
        Dictionary of quality metrics
    """
    if len(data) == 0:
        return {}
    
    # Basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    # Signal-to-noise ratio estimate
    snr = mean_val / std_val if std_val > 0 else 0
    
    # Dynamic range
    dynamic_range = np.max(data) - np.min(data)
    
    # Smoothness (inverse of second derivative magnitude)
    if len(data) > 2:
        second_diff = np.diff(data, n=2)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(second_diff)))
    else:
        smoothness = 1.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'snr': snr,
        'dynamic_range': dynamic_range,
        'smoothness': smoothness
    }


def filter_peaks_by_distance(peaks: List[int], min_distance: int = 5) -> List[int]:
    """
    Filter peaks to maintain minimum distance between them.
    
    Args:
        peaks: List of peak positions
        min_distance: Minimum distance between peaks
        
    Returns:
        Filtered list of peaks
    """
    if len(peaks) <= 1:
        return peaks.copy()
    
    # Sort peaks
    sorted_peaks = sorted(peaks)
    filtered_peaks = [sorted_peaks[0]]
    
    for peak in sorted_peaks[1:]:
        if peak - filtered_peaks[-1] >= min_distance:
            filtered_peaks.append(peak)
    
    return filtered_peaks


def main():
    """Test the peak detection and signal processing functions"""
    # Create test data with peaks
    x = np.linspace(0, 10, 200)
    test_signal = (2 * np.exp(-(x-2)**2/0.5) + 
                  1.5 * np.exp(-(x-5)**2/0.3) + 
                  1 * np.exp(-(x-8)**2/0.7) + 
                  0.1 * np.random.randn(200))
    
    print("Testing Peak Detection and Signal Processing Functions...")
    
    # Test moving average smoothing
    smoothed = moving_average_smoothing(test_signal, window_size=5)
    print(f"✓ Moving average smoothing: {len(smoothed)} points")
    
    # Test exponential smoothing
    exp_smoothed = exponential_smoothing(test_signal, alpha=0.3)
    print(f"✓ Exponential smoothing: {len(exp_smoothed)} points")
    
    # Test adaptive threshold peaks
    adaptive_peaks = adaptive_threshold_peaks(test_signal)
    print(f"✓ Adaptive threshold peaks: {len(adaptive_peaks)} peaks found")
    
    # Test gradient peak detection
    gradient_peaks = gradient_peak_detection(test_signal)
    print(f"✓ Gradient peak detection: {len(gradient_peaks)} peaks found")
    
    # Test multi-scale peak detection
    multi_scale_results = multi_scale_peak_detection(test_signal)
    print(f"✓ Multi-scale peak detection: {len(multi_scale_results)} scales")
    
    # Test consensus peak detection
    consensus_peaks = consensus_peak_detection(multi_scale_results)
    print(f"✓ Consensus peaks: {len(consensus_peaks)} peaks")
    
    # Test step change detection
    step_changes = detect_step_changes(test_signal)
    print(f"✓ Step changes detected: {len(step_changes)} positions")
    
    # Test signal quality analysis
    quality_metrics = analyze_signal_quality(test_signal)
    print(f"✓ Signal quality analysis: {list(quality_metrics.keys())}")
    
    # Test peak filtering by distance
    all_peaks = adaptive_peaks + gradient_peaks
    filtered_peaks = filter_peaks_by_distance(all_peaks, min_distance=10)
    print(f"✓ Peak filtering: {len(all_peaks)} -> {len(filtered_peaks)} peaks")
    
    # Create test image for histogram analysis
    test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    hist_peaks, hist_props = calculate_histogram_peaks(test_image)
    print(f"✓ Histogram peaks: {len(hist_peaks)} peaks found")
    
    # Test boundary detection
    gradient_data = np.abs(np.diff(test_signal))
    intensity_bounds, grad_bounds = find_intensity_boundaries(test_signal, gradient_data)
    print(f"✓ Boundary detection: {len(intensity_bounds)} intensity, {len(grad_bounds)} gradient")
    
    print("\nAll peak detection and signal processing functions tested successfully!")


if __name__ == "__main__":
    main()
