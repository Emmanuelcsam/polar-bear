#!/usr/bin/env python3
"""
Statistical Outlier Detection Module for Fiber Optic Defect Analysis
====================================================================

This module implements multiple statistical methods for detecting anomalous pixels
in fiber optic end-face images. Extracted from advanced_defect_analysis.py.

Functions:
- Z-score outlier detection
- Modified Z-score using Median Absolute Deviation (MAD)
- Interquartile Range (IQR) method
- Isolation Forest machine learning approach

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')


def zscore_outlier_detection(image, mask=None, threshold=3.0):
    """
    Detect outliers using Z-score method.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        threshold (float): Z-score threshold for outlier detection
    
    Returns:
        dict: Contains outlier mask, count, and percentage
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0  # Non-black pixels
    
    pixels = image[mask]
    if len(pixels) == 0:
        return {'outliers': np.zeros_like(mask), 'count': 0, 'percentage': 0.0}
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(pixels))
    z_outliers = z_scores > threshold
    
    # Create full image outlier mask
    outlier_mask = np.zeros_like(mask, dtype=bool)
    outlier_mask[mask] = z_outliers
    
    return {
        'outliers': outlier_mask,
        'count': np.sum(z_outliers),
        'percentage': (np.sum(z_outliers) / len(pixels)) * 100,
        'z_scores': z_scores
    }


def mad_outlier_detection(image, mask=None, threshold=3.5):
    """
    Detect outliers using Modified Z-score with Median Absolute Deviation.
    More robust to extreme outliers than standard Z-score.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        threshold (float): Modified Z-score threshold
    
    Returns:
        dict: Contains outlier mask, count, and percentage
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    pixels = image[mask]
    if len(pixels) == 0:
        return {'outliers': np.zeros_like(mask), 'count': 0, 'percentage': 0.0}
    
    # Calculate MAD-based modified Z-score
    median = np.median(pixels)
    mad = np.median(np.abs(pixels - median))
    
    if mad == 0:
        mad = 1e-10  # Avoid division by zero
    
    modified_z_scores = 0.6745 * (pixels - median) / mad
    mad_outliers = np.abs(modified_z_scores) > threshold
    
    # Create full image outlier mask
    outlier_mask = np.zeros_like(mask, dtype=bool)
    outlier_mask[mask] = mad_outliers
    
    return {
        'outliers': outlier_mask,
        'count': np.sum(mad_outliers),
        'percentage': (np.sum(mad_outliers) / len(pixels)) * 100,
        'mad_scores': modified_z_scores,
        'median': median,
        'mad': mad
    }


def iqr_outlier_detection(image, mask=None, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        multiplier (float): IQR multiplier for outlier bounds
    
    Returns:
        dict: Contains outlier mask, count, and percentage
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    pixels = image[mask]
    if len(pixels) == 0:
        return {'outliers': np.zeros_like(mask), 'count': 0, 'percentage': 0.0}
    
    # Calculate quartiles and IQR
    Q1 = np.percentile(pixels, 25)
    Q3 = np.percentile(pixels, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Find outliers
    iqr_outliers = (pixels < lower_bound) | (pixels > upper_bound)
    
    # Create full image outlier mask
    outlier_mask = np.zeros_like(mask, dtype=bool)
    outlier_mask[mask] = iqr_outliers
    
    return {
        'outliers': outlier_mask,
        'count': np.sum(iqr_outliers),
        'percentage': (np.sum(iqr_outliers) / len(pixels)) * 100,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'bounds': (lower_bound, upper_bound)
    }


def isolation_forest_outlier_detection(image, mask=None, contamination=0.05, random_state=42):
    """
    Detect outliers using Isolation Forest machine learning algorithm.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        contamination (float): Expected proportion of outliers
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Contains outlier mask, count, and percentage
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    pixels = image[mask]
    if len(pixels) == 0:
        return {'outliers': np.zeros_like(mask), 'count': 0, 'percentage': 0.0}
    
    # Reshape for sklearn
    pixel_reshape = pixels.reshape(-1, 1)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    iso_predictions = iso_forest.fit_predict(pixel_reshape)
    iso_outliers = iso_predictions == -1
    
    # Create full image outlier mask
    outlier_mask = np.zeros_like(mask, dtype=bool)
    outlier_mask[mask] = iso_outliers
    
    return {
        'outliers': outlier_mask,
        'count': np.sum(iso_outliers),
        'percentage': (np.sum(iso_outliers) / len(pixels)) * 100,
        'predictions': iso_predictions,
        'decision_scores': iso_forest.decision_function(pixel_reshape)
    }


def comprehensive_statistical_detection(image, mask=None, methods='all'):
    """
    Run multiple statistical outlier detection methods and combine results.
    
    Args:
        image (np.ndarray): Input image
        mask (np.ndarray, optional): Binary mask for region of interest
        methods (str or list): Methods to use ('all' or list of method names)
    
    Returns:
        dict: Results from all methods plus combined analysis
    """
    if methods == 'all':
        methods = ['zscore', 'mad', 'iqr', 'isolation_forest']
    
    results = {}
    combined_mask = np.zeros_like(image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=bool)
    
    if 'zscore' in methods:
        results['zscore'] = zscore_outlier_detection(image, mask)
        combined_mask |= results['zscore']['outliers']
    
    if 'mad' in methods:
        results['mad'] = mad_outlier_detection(image, mask)
        combined_mask |= results['mad']['outliers']
    
    if 'iqr' in methods:
        results['iqr'] = iqr_outlier_detection(image, mask)
        combined_mask |= results['iqr']['outliers']
    
    if 'isolation_forest' in methods:
        results['isolation_forest'] = isolation_forest_outlier_detection(image, mask)
        combined_mask |= results['isolation_forest']['outliers']
    
    # Combined statistics
    if mask is None:
        mask = (image > 0) if len(image.shape) == 2 else (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0)
    
    total_pixels = np.sum(mask)
    combined_count = np.sum(combined_mask)
    
    results['combined'] = {
        'outliers': combined_mask,
        'count': combined_count,
        'percentage': (combined_count / total_pixels * 100) if total_pixels > 0 else 0.0
    }
    
    return results


def visualize_statistical_results(image, results, save_path=None):
    """
    Visualize statistical outlier detection results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from statistical detection functions
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        gray = image
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Statistical Outlier Detection Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Individual methods
    method_names = ['zscore', 'mad', 'iqr', 'isolation_forest']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for i, method in enumerate(method_names):
        if method in results:
            row, col = positions[i]
            axes[row, col].imshow(results[method]['outliers'], cmap='hot')
            count = results[method]['count']
            percentage = results[method]['percentage']
            axes[row, col].set_title(f'{method.upper()}\n{count} outliers ({percentage:.2f}%)')
            axes[row, col].axis('off')
    
    # Combined results
    if 'combined' in results:
        axes[1, 2].imshow(results['combined']['outliers'], cmap='hot')
        count = results['combined']['count']
        percentage = results['combined']['percentage']
        axes[1, 2].set_title(f'Combined\n{count} outliers ({percentage:.2f}%)')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of statistical outlier detection functions.
    """
    # Create a test image with artificial defects
    test_image = np.random.normal(128, 20, (200, 200)).astype(np.uint8)
    
    # Add some artificial defects (outliers)
    test_image[50:60, 50:60] = 255  # Bright defect
    test_image[100:110, 100:110] = 0  # Dark defect
    test_image[150, 50:150] = 200  # Bright line
    
    print("Testing Statistical Outlier Detection Module")
    print("=" * 50)
    
    # Test individual methods
    print("\n1. Z-score method:")
    zscore_result = zscore_outlier_detection(test_image)
    print(f"   Found {zscore_result['count']} outliers ({zscore_result['percentage']:.2f}%)")
    
    print("\n2. MAD method:")
    mad_result = mad_outlier_detection(test_image)
    print(f"   Found {mad_result['count']} outliers ({mad_result['percentage']:.2f}%)")
    
    print("\n3. IQR method:")
    iqr_result = iqr_outlier_detection(test_image)
    print(f"   Found {iqr_result['count']} outliers ({iqr_result['percentage']:.2f}%)")
    
    print("\n4. Isolation Forest:")
    iso_result = isolation_forest_outlier_detection(test_image)
    print(f"   Found {iso_result['count']} outliers ({iso_result['percentage']:.2f}%)")
    
    # Test comprehensive detection
    print("\n5. Comprehensive detection:")
    comprehensive_result = comprehensive_statistical_detection(test_image)
    print(f"   Combined: {comprehensive_result['combined']['count']} outliers ({comprehensive_result['combined']['percentage']:.2f}%)")
    
    # Visualize results
    visualize_statistical_results(test_image, comprehensive_result, 'statistical_outlier_test.png')
    
    return comprehensive_result


if __name__ == "__main__":
    results = main()
