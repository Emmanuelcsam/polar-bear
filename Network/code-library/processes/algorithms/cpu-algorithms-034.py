"""
DO2MR (Difference of Min-Max Ranking) Detection Module

This module implements the DO2MR defect detection method specifically designed 
for fiber optic end face analysis. The method uses local min-max filtering
and statistical thresholding to identify defects.

Based on research paper implementations found in defect_detection2.py
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt


def do2mr_detection(image, mask=None, window_size=5, gamma=3.0, median_filter_size=3):
    """
    DO2MR (Difference of Min-Max Ranking) defect detection.
    
    This method uses local min-max filtering to identify defects by computing
    the residual between max and min filtered images, then applying statistical
    thresholding based on the residual distribution.
    
    Args:
        image: Input grayscale image (numpy array)
        mask: Optional binary mask to limit analysis region
        window_size: Size of the min-max filter window (default: 5)
        gamma: Statistical threshold multiplier (default: 3.0)
        median_filter_size: Size of median filter for residual smoothing (default: 3)
    
    Returns:
        Dictionary containing:
            - 'defects': Binary defect mask
            - 'residual': Min-max residual image
            - 'residual_smooth': Smoothed residual
            - 'threshold_value': Computed threshold value
            - 'statistics': Dictionary with statistical information
    """
    # Input validation
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Pad image for boundary handling
    pad_size = window_size // 2
    padded = np.pad(image, pad_size, mode='reflect')
    
    # Initialize output arrays
    min_filtered = np.zeros_like(image)
    max_filtered = np.zeros_like(image)
    
    # Apply min-max filtering
    print("Applying min-max filtering...")
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            window_region = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            min_filtered[i-pad_size, j-pad_size] = np.min(window_region)
            max_filtered[i-pad_size, j-pad_size] = np.max(window_region)
    
    # Compute residual (difference of min-max)
    residual = max_filtered - min_filtered
    
    # Apply median smoothing to residual
    residual_smooth = cv2.medianBlur(residual.astype(np.uint8), median_filter_size)
    
    # Statistical thresholding on masked region
    if np.sum(mask) == 0:
        print("Warning: Empty mask provided")
        return {
            'defects': np.zeros_like(image, dtype=bool),
            'residual': residual,
            'residual_smooth': residual_smooth,
            'threshold_value': 0,
            'statistics': {}
        }
    
    masked_residual = residual_smooth[mask]
    
    # Compute statistics
    mean_residual = np.mean(masked_residual)
    std_residual = np.std(masked_residual)
    median_residual = np.median(masked_residual)
    
    # Apply statistical threshold
    threshold_value = mean_residual + gamma * std_residual
    
    # Create defect mask
    defects = (residual_smooth > threshold_value) & mask
    
    # Morphological cleanup to remove noise
    defects = morphology.opening(defects, morphology.disk(1))
    defects = morphology.remove_small_objects(defects, min_size=3)
    
    # Compile statistics
    statistics = {
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'median_residual': median_residual,
        'threshold_value': threshold_value,
        'gamma': gamma,
        'window_size': window_size,
        'defect_count': np.sum(defects),
        'defect_percentage': (np.sum(defects) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0
    }
    
    print(f"DO2MR Detection Results:")
    print(f"  Threshold: {threshold_value:.2f}")
    print(f"  Defects found: {statistics['defect_count']}")
    print(f"  Defect percentage: {statistics['defect_percentage']:.2f}%")
    
    return {
        'defects': defects,
        'residual': residual,
        'residual_smooth': residual_smooth,
        'threshold_value': threshold_value,
        'statistics': statistics
    }


def enhanced_do2mr_detection(image, mask=None, multi_scale=True, scales=[3, 5, 7], 
                           adaptive_gamma=True, post_process=True):
    """
    Enhanced DO2MR detection with multi-scale analysis and adaptive thresholding.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        multi_scale: Use multiple window sizes (default: True)
        scales: List of window sizes to use (default: [3, 5, 7])
        adaptive_gamma: Use adaptive gamma based on image properties (default: True)
        post_process: Apply advanced post-processing (default: True)
    
    Returns:
        Dictionary with enhanced detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    if not multi_scale:
        # Single scale detection
        if adaptive_gamma:
            # Compute adaptive gamma based on image noise level
            noise_level = np.std(image[mask]) / np.mean(image[mask])
            gamma = 2.0 + min(2.0, noise_level * 10)  # Adaptive between 2.0 and 4.0
        else:
            gamma = 3.0
        
        return do2mr_detection(image, mask, window_size=5, gamma=gamma)
    
    # Multi-scale detection
    print("Performing multi-scale DO2MR detection...")
    
    # Store results for each scale
    scale_results = []
    combined_defects = np.zeros_like(image, dtype=bool)
    
    for scale in scales:
        if adaptive_gamma:
            # Scale-dependent gamma
            base_gamma = 2.5
            noise_level = np.std(image[mask]) / np.mean(image[mask])
            scale_factor = 1.0 + (scale - 5) * 0.1  # Scale adjustment
            gamma = base_gamma * scale_factor + noise_level
        else:
            gamma = 3.0
        
        print(f"  Processing scale {scale} with gamma {gamma:.2f}")
        
        result = do2mr_detection(image, mask, window_size=scale, gamma=gamma)
        scale_results.append({
            'scale': scale,
            'gamma': gamma,
            'result': result
        })
        
        # Combine defects using logical OR
        combined_defects |= result['defects']
    
    if post_process:
        # Advanced post-processing
        print("Applying post-processing...")
        
        # Remove very small objects
        combined_defects = morphology.remove_small_objects(combined_defects, min_size=5)
        
        # Fill small holes
        combined_defects = morphology.remove_small_holes(combined_defects, area_threshold=10)
        
        # Separate touching objects using watershed
        if np.sum(combined_defects) > 0:
            distance = ndimage.distance_transform_edt(combined_defects)
            from skimage import feature
            local_max = feature.peak_local_max(distance, min_distance=3, indices=False)
            markers = ndimage.label(local_max)[0]
            if np.max(markers) > 0:
                combined_defects = morphology.watershed(-distance, markers, mask=combined_defects)
                combined_defects = combined_defects > 0
    
    # Compile comprehensive statistics
    comprehensive_stats = {
        'multi_scale': True,
        'scales_used': scales,
        'total_defects': np.sum(combined_defects),
        'defect_percentage': (np.sum(combined_defects) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0,
        'scale_results': scale_results
    }
    
    print(f"Multi-scale DO2MR Results:")
    print(f"  Total defects: {comprehensive_stats['total_defects']}")
    print(f"  Defect percentage: {comprehensive_stats['defect_percentage']:.2f}%")
    
    return {
        'defects': combined_defects,
        'scale_results': scale_results,
        'statistics': comprehensive_stats
    }


def visualize_do2mr_results(image, results, save_path=None):
    """
    Visualize DO2MR detection results.
    
    Args:
        image: Original input image
        results: Results dictionary from do2mr_detection or enhanced_do2mr_detection
        save_path: Optional path to save the visualization
    """
    # Determine layout based on available results
    if 'scale_results' in results:
        # Multi-scale results
        num_scales = len(results['scale_results'])
        fig, axes = plt.subplots(2, num_scales + 1, figsize=(4 * (num_scales + 1), 8))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Final combined result
        axes[1, 0].imshow(results['defects'], cmap='hot')
        axes[1, 0].set_title('Combined Defects')
        axes[1, 0].axis('off')
        
        # Individual scale results
        for i, scale_result in enumerate(results['scale_results']):
            scale = scale_result['scale']
            defects = scale_result['result']['defects']
            residual = scale_result['result']['residual']
            
            axes[0, i + 1].imshow(residual, cmap='viridis')
            axes[0, i + 1].set_title(f'Residual (Scale {scale})')
            axes[0, i + 1].axis('off')
            
            axes[1, i + 1].imshow(defects, cmap='hot')
            axes[1, i + 1].set_title(f'Defects (Scale {scale})')
            axes[1, i + 1].axis('off')
    
    else:
        # Single scale results
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['residual'], cmap='viridis')
        axes[0, 1].set_title('Min-Max Residual')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(results['residual_smooth'], cmap='viridis')
        axes[1, 0].set_title('Smoothed Residual')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['defects'], cmap='hot')
        axes[1, 1].set_title('Detected Defects')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


# Demo and test code
if __name__ == "__main__":
    print("DO2MR Detection Module - Demo")
    print("=" * 40)
    
    # Create synthetic test image with defects
    print("Creating synthetic test image...")
    test_image = np.random.randint(100, 150, (200, 200), dtype=np.uint8)
    
    # Add some defects
    # Bright spot defects
    test_image[50:60, 50:60] = 200
    test_image[120:125, 150:160] = 220
    
    # Dark spot defects  
    test_image[80:90, 120:130] = 50
    test_image[160:170, 30:40] = 30
    
    # Linear scratch-like defect
    test_image[100:102, 20:80] = 180
    
    # Add some noise
    noise = np.random.normal(0, 5, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create circular mask
    center = (100, 100)
    radius = 80
    y, x = np.ogrid[:200, :200]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Mask coverage: {np.sum(mask)} pixels")
    
    # Test single-scale DO2MR
    print("\n1. Testing single-scale DO2MR detection...")
    single_results = do2mr_detection(test_image, mask, window_size=5, gamma=3.0)
    
    # Test multi-scale DO2MR
    print("\n2. Testing multi-scale DO2MR detection...")
    multi_results = enhanced_do2mr_detection(
        test_image, mask, 
        multi_scale=True, 
        scales=[3, 5, 7],
        adaptive_gamma=True
    )
    
    # Visualize results
    print("\n3. Visualizing results...")
    
    # Single scale visualization
    fig1 = visualize_do2mr_results(test_image, single_results)
    plt.suptitle('Single-Scale DO2MR Detection', fontsize=16)
    
    # Multi-scale visualization  
    fig2 = visualize_do2mr_results(test_image, multi_results)
    plt.suptitle('Multi-Scale DO2MR Detection', fontsize=16)
    
    # Performance comparison
    print("\n4. Performance comparison:")
    print(f"Single-scale defects: {single_results['statistics']['defect_count']}")
    print(f"Multi-scale defects: {multi_results['statistics']['total_defects']}")
    print(f"Improvement: {multi_results['statistics']['total_defects'] - single_results['statistics']['defect_count']} additional defects")
    
    print("\nDemo completed successfully!")
