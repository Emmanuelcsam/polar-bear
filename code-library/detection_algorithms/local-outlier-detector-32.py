"""
Local Outlier Factor (LOF) Detection Module

This module implements the Local Outlier Factor algorithm for detecting
anomalous pixels in fiber optic end face images. LOF is particularly
effective at finding outliers in dense regions.

Extracted from defect_detection2.py comprehensive detection system.
"""

import numpy as np
import cv2
from scipy.spatial import distance
from sklearn.neighbors import LocalOutlierFactor
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


def local_outlier_factor_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                                  k_neighbors: int = 20, contamination: float = 0.1,
                                  spatial_weight: float = 0.5, 
                                  use_spatial_features: bool = True) -> Dict:
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.
    
    The LOF algorithm measures the local density deviation of a data point
    with respect to its neighbors. Points with substantially lower density
    than their neighbors are considered outliers.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask to limit analysis region
        k_neighbors: Number of neighbors to consider for LOF calculation
        contamination: Expected proportion of outliers in the data
        spatial_weight: Weight for spatial coordinates vs intensity (0-1)
        use_spatial_features: Include spatial coordinates in feature space
    
    Returns:
        Dictionary containing LOF detection results
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # Get valid pixel coordinates and intensities
    valid_coords = np.column_stack(np.where(mask))
    valid_intensities = image[mask]
    
    if len(valid_coords) < k_neighbors + 1:
        print(f"Warning: Insufficient pixels ({len(valid_coords)}) for LOF with k={k_neighbors}")
        return {
            'outliers': np.zeros_like(mask, dtype=bool),
            'lof_scores': np.zeros_like(image, dtype=float),
            'statistics': {'error': 'Insufficient data'}
        }
    
    print(f"Running LOF detection on {len(valid_coords)} pixels with k={k_neighbors}")
    
    # Prepare feature matrix
    if use_spatial_features:
        # Normalize spatial coordinates to similar scale as intensities
        coords_normalized = valid_coords.astype(float)
        coords_normalized[:, 0] = coords_normalized[:, 0] / image.shape[0] * 255
        coords_normalized[:, 1] = coords_normalized[:, 1] / image.shape[1] * 255
        
        # Combine spatial and intensity features
        intensity_features = valid_intensities.reshape(-1, 1)
        spatial_features = coords_normalized * spatial_weight
        intensity_features = intensity_features * (1 - spatial_weight)
        
        features = np.hstack([intensity_features, spatial_features])
        print(f"Using spatial+intensity features: shape {features.shape}")
    else:
        # Use only intensity values
        features = valid_intensities.reshape(-1, 1)
        print(f"Using intensity-only features: shape {features.shape}")
    
    # Apply LOF algorithm
    try:
        lof = LocalOutlierFactor(
            n_neighbors=k_neighbors,
            contamination=contamination,
            algorithm='auto',
            metric='euclidean'
        )
        
        outlier_labels = lof.fit_predict(features)
        lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores
        
    except Exception as e:
        print(f"LOF algorithm failed: {e}")
        return {
            'outliers': np.zeros_like(mask, dtype=bool),
            'lof_scores': np.zeros_like(image, dtype=float),
            'statistics': {'error': str(e)}
        }
    
    # Create result masks
    outlier_mask = np.zeros_like(mask, dtype=bool)
    score_map = np.zeros_like(image, dtype=float)
    
    # Map results back to image space
    outlier_indices = np.where(outlier_labels == -1)[0]
    
    for i, (y, x) in enumerate(valid_coords):
        score_map[y, x] = lof_scores[i]
        if i in outlier_indices:
            outlier_mask[y, x] = True
    
    # Post-processing: remove isolated pixels
    outlier_mask_cleaned = morphology.remove_small_objects(outlier_mask, min_size=2)
    outlier_mask_cleaned = morphology.opening(outlier_mask_cleaned, morphology.disk(1))
    
    # Calculate statistics
    num_outliers = np.sum(outlier_mask_cleaned)
    total_pixels = np.sum(mask)
    outlier_percentage = (num_outliers / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Score statistics
    valid_scores = lof_scores[lof_scores > 0]  # Exclude undefined scores
    if len(valid_scores) > 0:
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        max_score = np.max(valid_scores)
        min_score = np.min(valid_scores)
        score_threshold = np.percentile(valid_scores, 100 * (1 - contamination))
    else:
        mean_score = std_score = max_score = min_score = score_threshold = 0.0
    
    statistics = {
        'k_neighbors': k_neighbors,
        'contamination': contamination,
        'spatial_weight': spatial_weight,
        'use_spatial_features': use_spatial_features,
        'total_pixels': total_pixels,
        'outliers_found': num_outliers,
        'outlier_percentage': outlier_percentage,
        'mean_lof_score': mean_score,
        'std_lof_score': std_score,
        'max_lof_score': max_score,
        'min_lof_score': min_score,
        'score_threshold': score_threshold,
        'feature_dimensions': features.shape[1]
    }
    
    print(f"LOF Detection Results:")
    print(f"  Outliers found: {num_outliers} ({outlier_percentage:.2f}%)")
    print(f"  Mean LOF score: {mean_score:.3f}")
    print(f"  Score threshold: {score_threshold:.3f}")
    
    return {
        'outliers': outlier_mask_cleaned,
        'outliers_raw': outlier_mask,
        'lof_scores': score_map,
        'features': features,
        'valid_coords': valid_coords,
        'statistics': statistics
    }


def adaptive_lof_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                          k_range: Tuple[int, int] = (5, 30),
                          contamination_range: Tuple[float, float] = (0.05, 0.2),
                          auto_tune: bool = True) -> Dict:
    """
    Adaptive LOF detection with automatic parameter tuning.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        k_range: Range of k values to test (min_k, max_k)
        contamination_range: Range of contamination values to test
        auto_tune: Automatically select best parameters
    
    Returns:
        Enhanced LOF detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    print("Performing adaptive LOF detection...")
    
    if not auto_tune:
        # Use middle values from ranges
        k_neighbors = (k_range[0] + k_range[1]) // 2
        contamination = (contamination_range[0] + contamination_range[1]) / 2
        return local_outlier_factor_detection(image, mask, k_neighbors, contamination)
    
    # Automatic parameter tuning
    print("Auto-tuning LOF parameters...")
    
    # Analyze image properties for parameter selection
    valid_pixels = image[mask]
    if len(valid_pixels) == 0:
        return {
            'outliers': np.zeros_like(mask, dtype=bool),
            'lof_scores': np.zeros_like(image, dtype=float),
            'statistics': {'error': 'Empty mask'}
        }
    
    # Estimate noise level
    noise_level = np.std(valid_pixels) / np.mean(valid_pixels)
    
    # Estimate density variation
    local_std = cv2.GaussianBlur(image.astype(float), (5, 5), 1.0)
    density_variation = np.std(local_std[mask]) / np.mean(local_std[mask])
    
    # Adaptive parameter selection
    # Higher noise -> larger k (more robust)
    # Higher density variation -> higher contamination
    
    base_k = min(k_range[1], max(k_range[0], int(10 + noise_level * 20)))
    base_contamination = min(contamination_range[1], 
                           max(contamination_range[0], 
                               0.05 + density_variation * 0.15))
    
    print(f"Selected parameters: k={base_k}, contamination={base_contamination:.3f}")
    print(f"  Based on: noise_level={noise_level:.3f}, density_variation={density_variation:.3f}")
    
    # Run LOF with selected parameters
    result = local_outlier_factor_detection(
        image, mask, 
        k_neighbors=base_k,
        contamination=base_contamination,
        use_spatial_features=True,
        spatial_weight=0.3  # Moderate spatial weighting
    )
    
    # Add adaptation info to statistics
    result['statistics'].update({
        'adaptive': True,
        'noise_level': noise_level,
        'density_variation': density_variation,
        'parameter_selection': {
            'k_range': k_range,
            'contamination_range': contamination_range,
            'selected_k': base_k,
            'selected_contamination': base_contamination
        }
    })
    
    return result


def multi_scale_lof_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                             scales: List[int] = [3, 5, 7], 
                             k_neighbors: int = 20,
                             ensemble_method: str = 'voting') -> Dict:
    """
    Multi-scale LOF detection using different image scales.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        scales: List of Gaussian blur sigma values for different scales
        k_neighbors: Number of neighbors for LOF
        ensemble_method: Method to combine scales ('voting', 'union', 'intersection')
    
    Returns:
        Multi-scale LOF detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    print(f"Performing multi-scale LOF detection with scales: {scales}")
    
    scale_results = []
    combined_outliers = np.zeros_like(mask, dtype=int)
    
    for scale in scales:
        print(f"\nProcessing scale {scale}...")
        
        # Apply Gaussian smoothing at this scale
        if scale > 0:
            smoothed = cv2.GaussianBlur(image, (0, 0), scale)
        else:
            smoothed = image.copy()
        
        # Estimate contamination based on scale
        # Larger scales should detect fewer, more significant outliers
        base_contamination = 0.1
        scale_factor = scale / max(scales)
        contamination = base_contamination * (1.0 - 0.5 * scale_factor)
        
        # Run LOF at this scale
        result = local_outlier_factor_detection(
            smoothed, mask,
            k_neighbors=k_neighbors,
            contamination=contamination,
            use_spatial_features=True
        )
        
        scale_results.append({
            'scale': scale,
            'contamination': contamination,
            'result': result
        })
        
        # Accumulate votes
        combined_outliers += result['outliers'].astype(int)
    
    # Combine results based on ensemble method
    if ensemble_method == 'voting':
        # Require majority vote
        threshold = len(scales) // 2 + 1
        final_outliers = combined_outliers >= threshold
    elif ensemble_method == 'union':
        # Any scale detects outlier
        final_outliers = combined_outliers > 0
    elif ensemble_method == 'intersection':
        # All scales must detect outlier
        final_outliers = combined_outliers == len(scales)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # Post-processing
    final_outliers = morphology.remove_small_objects(final_outliers, min_size=3)
    
    # Statistics
    num_outliers = np.sum(final_outliers)
    total_pixels = np.sum(mask)
    
    comprehensive_stats = {
        'multi_scale': True,
        'scales': scales,
        'ensemble_method': ensemble_method,
        'k_neighbors': k_neighbors,
        'total_outliers': num_outliers,
        'outlier_percentage': (num_outliers / total_pixels) * 100 if total_pixels > 0 else 0,
        'scale_results': scale_results,
        'vote_distribution': np.bincount(combined_outliers.flatten())
    }
    
    print(f"\nMulti-scale LOF Results:")
    print(f"  Ensemble method: {ensemble_method}")
    print(f"  Final outliers: {num_outliers} ({comprehensive_stats['outlier_percentage']:.2f}%)")
    
    return {
        'outliers': final_outliers,
        'vote_map': combined_outliers,
        'scale_results': scale_results,
        'statistics': comprehensive_stats
    }


def visualize_lof_results(image: np.ndarray, results: Dict, save_path: Optional[str] = None):
    """
    Visualize LOF detection results.
    
    Args:
        image: Original input image
        results: Results from LOF detection functions
        save_path: Optional path to save visualization
    """
    if 'scale_results' in results:
        # Multi-scale results
        num_scales = len(results['scale_results'])
        fig, axes = plt.subplots(3, num_scales + 1, figsize=(4 * (num_scales + 1), 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Final combined result
        axes[1, 0].imshow(results['outliers'], cmap='hot')
        axes[1, 0].set_title(f"Combined Outliers\n({results['statistics']['ensemble_method']})")
        axes[1, 0].axis('off')
        
        # Vote map
        if 'vote_map' in results:
            axes[2, 0].imshow(results['vote_map'], cmap='viridis')
            axes[2, 0].set_title('Vote Map')
            axes[2, 0].axis('off')
        
        # Individual scale results
        for i, scale_result in enumerate(results['scale_results']):
            scale = scale_result['scale']
            result = scale_result['result']
            
            # Show LOF scores
            axes[0, i + 1].imshow(result['lof_scores'], cmap='viridis')
            axes[0, i + 1].set_title(f'LOF Scores (σ={scale})')
            axes[0, i + 1].axis('off')
            
            # Show outliers
            axes[1, i + 1].imshow(result['outliers'], cmap='hot')
            axes[1, i + 1].set_title(f'Outliers (σ={scale})')
            axes[1, i + 1].axis('off')
            
            # Show score histogram
            valid_scores = result['lof_scores'][result['lof_scores'] > 0]
            if len(valid_scores) > 0:
                axes[2, i + 1].hist(valid_scores, bins=30, alpha=0.7)
                axes[2, i + 1].axvline(result['statistics']['score_threshold'], 
                                      color='red', linestyle='--', label='Threshold')
                axes[2, i + 1].set_title(f'Score Dist. (σ={scale})')
                axes[2, i + 1].legend()
    
    else:
        # Single scale results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['lof_scores'], cmap='viridis')
        axes[0, 1].set_title('LOF Scores')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['outliers'], cmap='hot')
        axes[0, 2].set_title('Detected Outliers')
        axes[0, 2].axis('off')
        
        # Raw vs cleaned outliers
        if 'outliers_raw' in results:
            axes[1, 0].imshow(results['outliers_raw'], cmap='hot')
            axes[1, 0].set_title('Raw Outliers')
            axes[1, 0].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        overlay[results['outliers']] = [255, 0, 0]
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Outliers Overlay')
        axes[1, 1].axis('off')
        
        # Score histogram
        valid_scores = results['lof_scores'][results['lof_scores'] > 0]
        if len(valid_scores) > 0:
            axes[1, 2].hist(valid_scores, bins=30, alpha=0.7, edgecolor='black')
            if 'score_threshold' in results['statistics']:
                axes[1, 2].axvline(results['statistics']['score_threshold'], 
                                  color='red', linestyle='--', label='Threshold')
                axes[1, 2].legend()
            axes[1, 2].set_title('LOF Score Distribution')
            axes[1, 2].set_xlabel('LOF Score')
            axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


# Demo and test code
if __name__ == "__main__":
    print("Local Outlier Factor (LOF) Detection Module - Demo")
    print("=" * 55)
    
    # Create synthetic test image with outliers
    print("Creating synthetic test image with outliers...")
    np.random.seed(42)
    
    # Base image with normal distribution
    test_image = np.random.normal(130, 10, (150, 150)).astype(np.uint8)
    test_image = np.clip(test_image, 0, 255)
    
    # Add some structured background
    x, y = np.meshgrid(np.linspace(0, 1, 150), np.linspace(0, 1, 150))
    background_pattern = 20 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    test_image = np.clip(test_image.astype(float) + background_pattern, 0, 255).astype(np.uint8)
    
    # Add outliers
    # Bright spot outliers
    test_image[30:35, 40:45] = 220
    test_image[80:83, 100:103] = 210
    test_image[120:125, 50:55] = 200
    
    # Dark spot outliers
    test_image[60:65, 70:75] = 50
    test_image[100:103, 30:33] = 40
    test_image[40:43, 120:123] = 60
    
    # Linear anomaly
    test_image[70:72, 80:120] = 180
    
    # Create circular mask
    center = (75, 75)
    radius = 60
    y_grid, x_grid = np.ogrid[:150, :150]
    mask = (x_grid - center[0])**2 + (y_grid - center[1])**2 <= radius**2
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Mask coverage: {np.sum(mask)} pixels")
    print("Added various outlier types")
    
    # Test basic LOF detection
    print("\n1. Testing basic LOF detection...")
    basic_results = local_outlier_factor_detection(
        test_image, mask,
        k_neighbors=15,
        contamination=0.1,
        use_spatial_features=True,
        spatial_weight=0.4
    )
    
    # Test adaptive LOF detection
    print("\n2. Testing adaptive LOF detection...")
    adaptive_results = adaptive_lof_detection(
        test_image, mask,
        k_range=(5, 25),
        contamination_range=(0.05, 0.2),
        auto_tune=True
    )
    
    # Test multi-scale LOF detection
    print("\n3. Testing multi-scale LOF detection...")
    multiscale_results = multi_scale_lof_detection(
        test_image, mask,
        scales=[0, 1, 2],
        k_neighbors=15,
        ensemble_method='voting'
    )
    
    # Visualize results
    print("\n4. Visualizing results...")
    
    # Basic LOF visualization
    fig1 = visualize_lof_results(test_image, basic_results)
    plt.suptitle('Basic LOF Detection', fontsize=16)
    
    # Adaptive LOF visualization
    fig2 = visualize_lof_results(test_image, adaptive_results)
    plt.suptitle('Adaptive LOF Detection', fontsize=16)
    
    # Multi-scale LOF visualization
    fig3 = visualize_lof_results(test_image, multiscale_results)
    plt.suptitle('Multi-Scale LOF Detection', fontsize=16)
    
    # Performance comparison
    print("\n5. Performance comparison:")
    basic_count = basic_results['statistics']['outliers_found']
    adaptive_count = adaptive_results['statistics']['outliers_found']
    multiscale_count = multiscale_results['statistics']['total_outliers']
    
    print(f"Basic LOF: {basic_count} outliers ({basic_results['statistics']['outlier_percentage']:.2f}%)")
    print(f"Adaptive LOF: {adaptive_count} outliers ({adaptive_results['statistics']['outlier_percentage']:.2f}%)")
    print(f"Multi-scale LOF: {multiscale_count} outliers ({multiscale_results['statistics']['outlier_percentage']:.2f}%)")
    
    # Parameter analysis
    print("\n6. Parameter analysis:")
    print(f"Basic LOF: k={basic_results['statistics']['k_neighbors']}, contamination={basic_results['statistics']['contamination']}")
    if 'parameter_selection' in adaptive_results['statistics']:
        params = adaptive_results['statistics']['parameter_selection']
        print(f"Adaptive LOF: k={params['selected_k']}, contamination={params['selected_contamination']:.3f}")
    print(f"Multi-scale LOF: ensemble={multiscale_results['statistics']['ensemble_method']}, scales={multiscale_results['statistics']['scales']}")
    
    print("\nDemo completed successfully!")
