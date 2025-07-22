"""
Quality Metrics Calculation Module

This module provides comprehensive quality assessment functions for fiber optic
end face images, including surface roughness, uniformity, contrast measures,
and structural similarity metrics.

Extracted from defect_detection2.py comprehensive detection system.
"""

import numpy as np
import cv2
from scipy import stats, ndimage
from skimage import measure, filters
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def calculate_surface_roughness(image: np.ndarray, mask: Optional[np.ndarray] = None,
                               method: str = 'rms') -> Dict:
    """
    Calculate surface roughness metrics.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask for region of interest
        method: Roughness calculation method ('rms', 'ra', 'rz', 'all')
    
    Returns:
        Dictionary containing roughness metrics
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # Get pixels within mask
    roi_pixels = image[mask]
    
    if len(roi_pixels) == 0:
        return {'error': 'Empty mask or region'}
    
    # Calculate mean level (centerline)
    mean_level = np.mean(roi_pixels)
    
    # Deviations from mean
    deviations = roi_pixels.astype(float) - mean_level
    
    results = {}
    
    if method in ['rms', 'all']:
        # RMS (Root Mean Square) roughness
        rms_roughness = np.sqrt(np.mean(deviations**2))
        results['rms_roughness'] = rms_roughness
    
    if method in ['ra', 'all']:
        # Ra (Average roughness) - arithmetic mean of absolute deviations
        ra_roughness = np.mean(np.abs(deviations))
        results['ra_roughness'] = ra_roughness
    
    if method in ['rz', 'all']:
        # Rz (Ten-point height) - average of 5 highest peaks and 5 lowest valleys
        sorted_devs = np.sort(deviations)
        n = len(sorted_devs)
        if n >= 10:
            # Take 5 highest and 5 lowest values
            highest_5 = sorted_devs[-5:]
            lowest_5 = sorted_devs[:5]
            rz_roughness = np.mean(highest_5) - np.mean(lowest_5)
        else:
            # If less than 10 values, use max - min
            rz_roughness = np.max(sorted_devs) - np.min(sorted_devs)
        results['rz_roughness'] = rz_roughness
    
    if method == 'all':
        # Additional metrics
        results.update({
            'mean_level': mean_level,
            'std_deviation': np.std(roi_pixels),
            'peak_to_valley': np.max(roi_pixels) - np.min(roi_pixels),
            'skewness': stats.skew(roi_pixels),
            'kurtosis': stats.kurtosis(roi_pixels)
        })
    
    return results


def calculate_uniformity_metrics(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate uniformity and homogeneity metrics.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask for region of interest
    
    Returns:
        Dictionary containing uniformity metrics
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    roi_pixels = image[mask]
    
    if len(roi_pixels) == 0:
        return {'error': 'Empty mask or region'}
    
    mean_intensity = np.mean(roi_pixels)
    std_intensity = np.std(roi_pixels)
    
    # Basic uniformity (coefficient of variation)
    if mean_intensity > 0:
        uniformity_coefficient = 1.0 - (std_intensity / mean_intensity)
    else:
        uniformity_coefficient = 0.0
    
    # Normalized uniformity (0-1 scale)
    normalized_uniformity = max(0, uniformity_coefficient)
    
    # Local uniformity using local standard deviation
    local_std_map = filters.rank.standard(image, np.ones((5, 5)))
    local_uniformity = 1.0 - np.mean(local_std_map[mask]) / 255.0
    
    # Histogram uniformity (entropy-based)
    hist, _ = np.histogram(roi_pixels, bins=256, range=(0, 256))
    hist_normalized = hist / (np.sum(hist) + 1e-10)
    hist_normalized = hist_normalized[hist_normalized > 0]
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    max_entropy = np.log2(256)  # Maximum possible entropy
    entropy_uniformity = 1.0 - (entropy / max_entropy)
    
    return {
        'uniformity_coefficient': uniformity_coefficient,
        'normalized_uniformity': normalized_uniformity,
        'local_uniformity': local_uniformity,
        'entropy_uniformity': entropy_uniformity,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'coefficient_of_variation': std_intensity / (mean_intensity + 1e-10),
        'entropy': entropy,
        'dynamic_range': np.max(roi_pixels) - np.min(roi_pixels)
    }


def calculate_contrast_metrics(image: np.ndarray, defect_mask: Optional[np.ndarray] = None,
                              roi_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate contrast metrics between defects and background.
    
    Args:
        image: Input grayscale image
        defect_mask: Binary mask of detected defects
        roi_mask: Binary mask for region of interest
    
    Returns:
        Dictionary containing contrast metrics
    """
    if roi_mask is None:
        roi_mask = np.ones_like(image, dtype=bool)
    else:
        roi_mask = roi_mask.astype(bool)
    
    # Background pixels (ROI excluding defects)
    if defect_mask is not None:
        background_mask = roi_mask & ~defect_mask.astype(bool)
        defect_pixels = image[defect_mask.astype(bool) & roi_mask]
    else:
        background_mask = roi_mask
        defect_pixels = np.array([])
    
    background_pixels = image[background_mask]
    
    if len(background_pixels) == 0:
        return {'error': 'No background pixels available'}
    
    background_mean = np.mean(background_pixels)
    background_std = np.std(background_pixels)
    
    results = {
        'background_mean': background_mean,
        'background_std': background_std,
    }
    
    if len(defect_pixels) > 0:
        defect_mean = np.mean(defect_pixels)
        defect_std = np.std(defect_pixels)
        
        # Michelson contrast
        max_val = max(background_mean, defect_mean)
        min_val = min(background_mean, defect_mean)
        if max_val + min_val > 0:
            michelson_contrast = (max_val - min_val) / (max_val + min_val)
        else:
            michelson_contrast = 0.0
        
        # Weber contrast
        if background_mean > 0:
            weber_contrast = abs(defect_mean - background_mean) / background_mean
        else:
            weber_contrast = 0.0
        
        # RMS contrast
        all_roi_pixels = image[roi_mask]
        if len(all_roi_pixels) > 0:
            roi_mean = np.mean(all_roi_pixels)
            rms_contrast = np.sqrt(np.mean((all_roi_pixels - roi_mean)**2)) / (roi_mean + 1e-10)
        else:
            rms_contrast = 0.0
        
        # Signal-to-noise ratio
        if background_std > 0:
            snr = abs(defect_mean - background_mean) / background_std
        else:
            snr = float('inf') if defect_mean != background_mean else 0.0
        
        results.update({
            'defect_mean': defect_mean,
            'defect_std': defect_std,
            'michelson_contrast': michelson_contrast,
            'weber_contrast': weber_contrast,
            'rms_contrast': rms_contrast,
            'snr': snr,
            'mean_difference': abs(defect_mean - background_mean),
            'contrast_ratio': defect_mean / (background_mean + 1e-10)
        })
    else:
        # Only RMS contrast for background
        if len(background_pixels) > 0:
            rms_contrast = np.sqrt(np.mean((background_pixels - background_mean)**2)) / (background_mean + 1e-10)
        else:
            rms_contrast = 0.0
        
        results.update({
            'rms_contrast': rms_contrast,
            'michelson_contrast': 0.0,
            'weber_contrast': 0.0,
            'snr': 0.0,
            'defect_mean': 0.0,
            'defect_std': 0.0
        })
    
    return results


def calculate_structural_similarity(image: np.ndarray, reference: Optional[np.ndarray] = None,
                                   mask: Optional[np.ndarray] = None,
                                   window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> Dict:
    """
    Calculate structural similarity metrics.
    
    Args:
        image: Input image
        reference: Reference image (if None, uses idealized smooth surface)
        mask: Optional binary mask
        window_size: Size of sliding window for SSIM calculation
        k1, k2: SSIM constants
    
    Returns:
        Dictionary containing structural similarity metrics
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    if reference is None:
        # Create idealized smooth reference by heavy smoothing
        reference = cv2.GaussianBlur(image, (15, 15), 5.0)
    
    # Ensure same shape
    if image.shape != reference.shape:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
    
    # SSIM constants
    c1 = (k1 * 255)**2
    c2 = (k2 * 255)**2
    
    # Get masked pixels
    img_pixels = image[mask].astype(float)
    ref_pixels = reference[mask].astype(float)
    
    if len(img_pixels) == 0:
        return {'error': 'Empty mask'}
    
    # Calculate SSIM components
    mu1 = np.mean(img_pixels)
    mu2 = np.mean(ref_pixels)
    
    sigma1_sq = np.var(img_pixels)
    sigma2_sq = np.var(ref_pixels)
    sigma12 = np.mean((img_pixels - mu1) * (ref_pixels - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    if denominator > 0:
        ssim = numerator / denominator
    else:
        ssim = 0.0
    
    # Additional similarity metrics
    # Correlation coefficient
    if len(img_pixels) > 1:
        correlation = np.corrcoef(img_pixels, ref_pixels)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Mean squared error
    mse = np.mean((img_pixels - ref_pixels)**2)
    
    # Peak signal-to-noise ratio
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Normalized cross-correlation
    if np.std(img_pixels) > 0 and np.std(ref_pixels) > 0:
        ncc = np.mean(((img_pixels - mu1) / np.std(img_pixels)) * 
                     ((ref_pixels - mu2) / np.std(ref_pixels)))
    else:
        ncc = 0.0
    
    return {
        'ssim': ssim,
        'correlation': correlation,
        'mse': mse,
        'psnr': psnr,
        'ncc': ncc,
        'mu1': mu1,
        'mu2': mu2,
        'sigma1': np.sqrt(sigma1_sq),
        'sigma2': np.sqrt(sigma2_sq),
        'sigma12': sigma12
    }


def calculate_comprehensive_quality_metrics(image: np.ndarray, 
                                          defect_mask: Optional[np.ndarray] = None,
                                          roi_mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate comprehensive quality metrics for fiber optic end face analysis.
    
    Args:
        image: Input grayscale image
        defect_mask: Binary mask of detected defects
        roi_mask: Binary mask for region of interest
    
    Returns:
        Comprehensive dictionary containing all quality metrics
    """
    if roi_mask is None:
        roi_mask = np.ones_like(image, dtype=bool)
    
    print("Calculating comprehensive quality metrics...")
    
    # 1. Surface roughness metrics
    print("  Computing surface roughness...")
    roughness_metrics = calculate_surface_roughness(image, roi_mask, method='all')
    
    # 2. Uniformity metrics
    print("  Computing uniformity metrics...")
    uniformity_metrics = calculate_uniformity_metrics(image, roi_mask)
    
    # 3. Contrast metrics
    print("  Computing contrast metrics...")
    contrast_metrics = calculate_contrast_metrics(image, defect_mask, roi_mask)
    
    # 4. Structural similarity
    print("  Computing structural similarity...")
    similarity_metrics = calculate_structural_similarity(image, mask=roi_mask)
    
    # 5. Defect density metrics
    if defect_mask is not None:
        defect_area = np.sum(defect_mask.astype(bool) & roi_mask)
        roi_area = np.sum(roi_mask)
        defect_density = defect_area / roi_area if roi_area > 0 else 0.0
        
        # Count individual defects
        labeled_defects, num_defects = ndimage.label(defect_mask.astype(bool) & roi_mask)
        
        defect_metrics = {
            'defect_density': defect_density,
            'defect_percentage': defect_density * 100,
            'total_defect_area': defect_area,
            'defect_count': num_defects,
            'roi_area': roi_area
        }
    else:
        defect_metrics = {
            'defect_density': 0.0,
            'defect_percentage': 0.0,
            'total_defect_area': 0,
            'defect_count': 0,
            'roi_area': np.sum(roi_mask)
        }
    
    # 6. Overall quality score (0-100)
    print("  Computing overall quality score...")
    
    # Normalize individual metrics to 0-1 scale (higher is better)
    uniformity_score = uniformity_metrics.get('normalized_uniformity', 0.0)
    
    # Invert roughness (lower roughness = higher quality)
    max_roughness = 50.0  # Assume max reasonable roughness
    roughness_score = max(0, 1.0 - roughness_metrics.get('rms_roughness', 0) / max_roughness)
    
    # Structural similarity score
    similarity_score = max(0, similarity_metrics.get('ssim', 0.0))
    
    # Defect score (lower defect density = higher quality)
    defect_score = max(0, 1.0 - defect_metrics['defect_density'] * 10)  # Scale by 10
    
    # Weighted overall quality (0-100)
    weights = {'uniformity': 0.25, 'roughness': 0.25, 'similarity': 0.25, 'defects': 0.25}
    overall_quality = (weights['uniformity'] * uniformity_score +
                      weights['roughness'] * roughness_score +
                      weights['similarity'] * similarity_score +
                      weights['defects'] * defect_score) * 100
    
    quality_breakdown = {
        'overall_quality': overall_quality,
        'uniformity_score': uniformity_score * 100,
        'roughness_score': roughness_score * 100,
        'similarity_score': similarity_score * 100,
        'defect_score': defect_score * 100,
        'weights': weights
    }
    
    # Compile all results
    comprehensive_metrics = {
        'roughness': roughness_metrics,
        'uniformity': uniformity_metrics,
        'contrast': contrast_metrics,
        'similarity': similarity_metrics,
        'defects': defect_metrics,
        'quality': quality_breakdown
    }
    
    print(f"  Overall quality score: {overall_quality:.1f}/100")
    
    return comprehensive_metrics


def generate_quality_report(image: np.ndarray, metrics: Dict, 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a comprehensive quality report with visualizations.
    
    Args:
        image: Original image
        metrics: Quality metrics dictionary
        save_path: Optional path to save the report
    
    Returns:
        Matplotlib figure containing the report
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 2. Quality score gauge
    ax2 = fig.add_subplot(gs[0, 1])
    quality_score = metrics['quality']['overall_quality']
    
    # Create a simple quality gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Color based on quality
    if quality_score >= 80:
        color = 'green'
    elif quality_score >= 60:
        color = 'orange'
    else:
        color = 'red'
    
    ax2.plot(x, y, 'k-', linewidth=2)
    ax2.fill_between(x, 0, y, alpha=0.3, color=color)
    ax2.text(0, 0.5, f'{quality_score:.1f}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(0, 1.2)
    ax2.set_title('Overall Quality Score')
    ax2.axis('off')
    
    # 3. Quality breakdown bar chart
    ax3 = fig.add_subplot(gs[0, 2:])
    quality_data = metrics['quality']
    scores = [quality_data['uniformity_score'], quality_data['roughness_score'],
              quality_data['similarity_score'], quality_data['defect_score']]
    labels = ['Uniformity', 'Roughness', 'Similarity', 'Defects']
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax3.bar(labels, scores, color=colors, alpha=0.7)
    ax3.set_ylabel('Score (0-100)')
    ax3.set_title('Quality Component Breakdown')
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}', ha='center', va='bottom')
    
    # 4. Roughness metrics table
    ax4 = fig.add_subplot(gs[1, :2])
    roughness_data = metrics['roughness']
    roughness_text = []
    roughness_text.append(['Metric', 'Value'])
    roughness_text.append(['RMS Roughness', f"{roughness_data.get('rms_roughness', 0):.2f}"])
    roughness_text.append(['Ra Roughness', f"{roughness_data.get('ra_roughness', 0):.2f}"])
    roughness_text.append(['Rz Roughness', f"{roughness_data.get('rz_roughness', 0):.2f}"])
    roughness_text.append(['Peak-to-Valley', f"{roughness_data.get('peak_to_valley', 0):.1f}"])
    
    table = ax4.table(cellText=roughness_text[1:], colLabels=roughness_text[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax4.set_title('Surface Roughness Metrics')
    ax4.axis('off')
    
    # 5. Uniformity metrics table
    ax5 = fig.add_subplot(gs[1, 2:])
    uniformity_data = metrics['uniformity']
    uniformity_text = []
    uniformity_text.append(['Metric', 'Value'])
    uniformity_text.append(['Uniformity Coeff.', f"{uniformity_data.get('uniformity_coefficient', 0):.3f}"])
    uniformity_text.append(['Local Uniformity', f"{uniformity_data.get('local_uniformity', 0):.3f}"])
    uniformity_text.append(['Entropy Uniformity', f"{uniformity_data.get('entropy_uniformity', 0):.3f}"])
    uniformity_text.append(['Coeff. of Variation', f"{uniformity_data.get('coefficient_of_variation', 0):.3f}"])
    
    table = ax5.table(cellText=uniformity_text[1:], colLabels=uniformity_text[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax5.set_title('Uniformity Metrics')
    ax5.axis('off')
    
    # 6. Contrast metrics table
    ax6 = fig.add_subplot(gs[2, :2])
    contrast_data = metrics['contrast']
    contrast_text = []
    contrast_text.append(['Metric', 'Value'])
    if 'michelson_contrast' in contrast_data:
        contrast_text.append(['Michelson Contrast', f"{contrast_data['michelson_contrast']:.3f}"])
        contrast_text.append(['Weber Contrast', f"{contrast_data['weber_contrast']:.3f}"])
        contrast_text.append(['RMS Contrast', f"{contrast_data['rms_contrast']:.3f}"])
        contrast_text.append(['SNR', f"{contrast_data['snr']:.2f}"])
    
    if len(contrast_text) > 1:
        table = ax6.table(cellText=contrast_text[1:], colLabels=contrast_text[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    ax6.set_title('Contrast Metrics')
    ax6.axis('off')
    
    # 7. Defect statistics table
    ax7 = fig.add_subplot(gs[2, 2:])
    defect_data = metrics['defects']
    defect_text = []
    defect_text.append(['Metric', 'Value'])
    defect_text.append(['Defect Count', f"{defect_data['defect_count']}"])
    defect_text.append(['Defect Percentage', f"{defect_data['defect_percentage']:.2f}%"])
    defect_text.append(['Total Defect Area', f"{defect_data['total_defect_area']} px"])
    defect_text.append(['Defect Density', f"{defect_data['defect_density']:.4f}"])
    
    table = ax7.table(cellText=defect_text[1:], colLabels=defect_text[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax7.set_title('Defect Statistics')
    ax7.axis('off')
    
    # 8. Summary text
    ax8 = fig.add_subplot(gs[3, :])
    summary_text = f"""
QUALITY ASSESSMENT SUMMARY

Overall Quality Score: {quality_score:.1f}/100 ({'Excellent' if quality_score >= 90 else 'Good' if quality_score >= 70 else 'Fair' if quality_score >= 50 else 'Poor'})

Key Findings:
• Surface uniformity: {metrics['quality']['uniformity_score']:.1f}/100
• Surface roughness: {metrics['quality']['roughness_score']:.1f}/100 (RMS: {metrics['roughness'].get('rms_roughness', 0):.2f})
• Structural integrity: {metrics['quality']['similarity_score']:.1f}/100
• Defect assessment: {metrics['quality']['defect_score']:.1f}/100 ({defect_data['defect_count']} defects found)

Recommendations:
{"• Excellent quality - no action needed" if quality_score >= 90 else
 "• Good quality - monitor for changes" if quality_score >= 70 else
 "• Fair quality - consider maintenance" if quality_score >= 50 else
 "• Poor quality - immediate attention required"}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Quality Assessment Summary')
    ax8.axis('off')
    
    plt.suptitle('Fiber Optic End Face Quality Report', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quality report saved to {save_path}")
    
    return fig


# Demo and test code
if __name__ == "__main__":
    print("Quality Metrics Calculation Module - Demo")
    print("=" * 45)
    
    # Create synthetic test image
    print("Creating synthetic test image...")
    np.random.seed(42)
    
    # Base image with some texture
    test_image = np.random.randint(130, 150, (200, 200), dtype=np.uint8)
    
    # Add some surface variations (simulating fiber texture)
    for i in range(5):
        x = np.random.randint(20, 180)
        y = np.random.randint(20, 180)
        radius = np.random.randint(10, 20)
        intensity = np.random.randint(120, 160)
        cv2.circle(test_image, (x, y), radius, intensity, -1)
    
    # Add defects
    cv2.circle(test_image, (50, 50), 8, 80, -1)  # Dark defect
    cv2.circle(test_image, (150, 100), 6, 200, -1)  # Bright defect
    cv2.line(test_image, (30, 120), (90, 140), 70, 2)  # Scratch
    
    # Add noise
    noise = np.random.normal(0, 2, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create masks
    center = (100, 100)
    radius = 90
    y, x = np.ogrid[:200, :200]
    roi_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # Create simple defect mask
    defect_mask = np.zeros_like(test_image, dtype=bool)
    cv2.circle(defect_mask, (50, 50), 8, True, -1)
    cv2.circle(defect_mask, (150, 100), 6, True, -1)
    cv2.rectangle(defect_mask, (30, 118), (90, 142), True, -1)
    
    print(f"Test image shape: {test_image.shape}")
    print(f"ROI coverage: {np.sum(roi_mask)} pixels")
    print(f"Defect coverage: {np.sum(defect_mask)} pixels")
    
    # Test individual metric calculations
    print("\n1. Testing surface roughness calculation...")
    roughness = calculate_surface_roughness(test_image, roi_mask, method='all')
    print(f"   RMS roughness: {roughness.get('rms_roughness', 0):.2f}")
    print(f"   Ra roughness: {roughness.get('ra_roughness', 0):.2f}")
    
    print("\n2. Testing uniformity metrics...")
    uniformity = calculate_uniformity_metrics(test_image, roi_mask)
    print(f"   Uniformity coefficient: {uniformity.get('uniformity_coefficient', 0):.3f}")
    print(f"   Entropy uniformity: {uniformity.get('entropy_uniformity', 0):.3f}")
    
    print("\n3. Testing contrast metrics...")
    contrast = calculate_contrast_metrics(test_image, defect_mask, roi_mask)
    print(f"   Michelson contrast: {contrast.get('michelson_contrast', 0):.3f}")
    print(f"   SNR: {contrast.get('snr', 0):.2f}")
    
    print("\n4. Testing structural similarity...")
    similarity = calculate_structural_similarity(test_image, mask=roi_mask)
    print(f"   SSIM: {similarity.get('ssim', 0):.3f}")
    print(f"   Correlation: {similarity.get('correlation', 0):.3f}")
    
    print("\n5. Computing comprehensive quality metrics...")
    comprehensive = calculate_comprehensive_quality_metrics(test_image, defect_mask, roi_mask)
    
    print(f"\nOverall Quality Assessment:")
    print(f"   Overall score: {comprehensive['quality']['overall_quality']:.1f}/100")
    print(f"   Uniformity: {comprehensive['quality']['uniformity_score']:.1f}/100")
    print(f"   Roughness: {comprehensive['quality']['roughness_score']:.1f}/100")
    print(f"   Similarity: {comprehensive['quality']['similarity_score']:.1f}/100")
    print(f"   Defects: {comprehensive['quality']['defect_score']:.1f}/100")
    
    # Generate quality report
    print("\n6. Generating quality report...")
    fig = generate_quality_report(test_image, comprehensive)
    plt.show()
    
    print("\nDemo completed successfully!")
