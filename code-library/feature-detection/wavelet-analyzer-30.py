"""
Wavelet Transform Analysis Module

This module implements wavelet-based defect detection for fiber optic end face
images. Wavelet transforms are excellent for detecting transient features and
multi-scale analysis of surface defects.

Extracted from defect_detection2.py comprehensive detection system.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import warnings

# Try to import pywt, provide fallback if not available
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    print("Warning: PyWavelets (pywt) not available. Using fallback implementations.")
    PYWT_AVAILABLE = False

warnings.filterwarnings('ignore')


def simple_wavelet_transform(image: np.ndarray, levels: int = 3) -> Dict:
    """
    Simple wavelet transform using basic filtering (fallback implementation).
    
    Args:
        image: Input grayscale image
        levels: Number of decomposition levels
    
    Returns:
        Dictionary containing approximation and detail coefficients
    """
    # Simple approximation using Gaussian smoothing
    # and details using difference
    
    coeffs = {}
    current = image.astype(float)
    
    for level in range(levels):
        # Low-pass (approximation)
        sigma = 2**(level + 1)
        approx = cv2.GaussianBlur(current, (0, 0), sigma)
        
        # High-pass (detail) - difference between levels
        if level == 0:
            detail = current - approx
        else:
            prev_approx = cv2.GaussianBlur(current, (0, 0), sigma / 2)
            detail = prev_approx - approx
        
        coeffs[f'level_{level}'] = {
            'approximation': approx,
            'detail': detail
        }
        
        current = approx
    
    return coeffs


def wavelet_defect_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                           wavelet: str = 'db4', levels: int = 3,
                           threshold_method: str = 'soft',
                           threshold_mode: str = 'auto') -> Dict:
    """
    Detect defects using wavelet transform analysis.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask to limit analysis region
        wavelet: Wavelet family to use ('db4', 'haar', 'bior2.2', etc.)
        levels: Number of decomposition levels
        threshold_method: Thresholding method ('soft', 'hard', 'adaptive')
        threshold_mode: How to determine threshold ('auto', 'percentile', 'sigma')
    
    Returns:
        Dictionary containing wavelet detection results
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    print(f"Wavelet defect detection using {wavelet} with {levels} levels")
    
    if PYWT_AVAILABLE:
        # Use PyWavelets for proper wavelet transform
        coeffs = pywt.wavedec2(image, wavelet, level=levels)
        print(f"Wavelet decomposition completed: {len(coeffs)} levels")
        
        # Analyze detail coefficients for defects
        defect_maps = []
        
        for level in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level]  # Horizontal, Vertical, Diagonal details
            
            # Combine detail coefficients
            combined_detail = np.sqrt(cH**2 + cV**2 + cD**2)
            
            # Determine threshold
            if threshold_mode == 'auto':
                # Automatic threshold using MAD (Median Absolute Deviation)
                median_val = np.median(combined_detail)
                mad = np.median(np.abs(combined_detail - median_val))
                threshold = median_val + 3 * mad
            elif threshold_mode == 'percentile':
                threshold = np.percentile(combined_detail, 95)
            elif threshold_mode == 'sigma':
                threshold = np.mean(combined_detail) + 3 * np.std(combined_detail)
            else:
                threshold = np.percentile(combined_detail, 90)  # Default
            
            # Apply thresholding
            if threshold_method == 'soft':
                # Soft thresholding
                thresholded = np.sign(combined_detail) * np.maximum(np.abs(combined_detail) - threshold, 0)
            elif threshold_method == 'hard':
                # Hard thresholding
                thresholded = combined_detail * (np.abs(combined_detail) > threshold)
            else:  # adaptive
                # Adaptive thresholding
                local_threshold = cv2.adaptiveThreshold(
                    (combined_detail * 255 / np.max(combined_detail)).astype(np.uint8),
                    255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                thresholded = (local_threshold > 0).astype(float)
            
            # Reconstruct defect map from thresholded coefficients
            # Create zero coefficients except for current level
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            for i in range(1, len(coeffs)):
                if i == level:
                    # Use thresholded details
                    scale_factor = np.max(np.abs(thresholded)) / (np.max(np.abs(cH)) + 1e-10)
                    coeffs_thresh.append((thresholded * scale_factor, 
                                        thresholded * scale_factor, 
                                        thresholded * scale_factor))
                else:
                    # Zero out other levels
                    coeffs_thresh.append((np.zeros_like(coeffs[i][0]), 
                                        np.zeros_like(coeffs[i][1]), 
                                        np.zeros_like(coeffs[i][2])))
            
            # Reconstruct
            try:
                detail_recon = pywt.waverec2(coeffs_thresh, wavelet)
                
                # Resize if necessary
                if detail_recon.shape != image.shape:
                    detail_recon = cv2.resize(detail_recon, (image.shape[1], image.shape[0]))
                
                # Create binary defect map
                detail_normalized = np.abs(detail_recon)
                detail_normalized = (detail_normalized / np.max(detail_normalized) * 255).astype(np.uint8)
                
                _, binary_defects = cv2.threshold(detail_normalized, 0, 255, 
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                defect_maps.append({
                    'level': level,
                    'detail_magnitude': combined_detail,
                    'threshold': threshold,
                    'defects': binary_defects > 0,
                    'reconstruction': detail_recon
                })
                
            except Exception as e:
                print(f"Warning: Reconstruction failed for level {level}: {e}")
                # Fallback: use thresholded details directly
                detail_resized = cv2.resize(thresholded, (image.shape[1], image.shape[0]))
                _, binary_defects = cv2.threshold(
                    (detail_resized * 255 / np.max(detail_resized)).astype(np.uint8),
                    0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                defect_maps.append({
                    'level': level,
                    'detail_magnitude': combined_detail,
                    'threshold': threshold,
                    'defects': binary_defects > 0,
                    'reconstruction': detail_resized
                })
    
    else:
        # Fallback implementation without PyWavelets
        print("Using fallback wavelet implementation")
        simple_coeffs = simple_wavelet_transform(image, levels)
        
        defect_maps = []
        for level in range(levels):
            detail = simple_coeffs[f'level_{level}']['detail']
            
            # Threshold detail coefficients
            threshold = np.percentile(np.abs(detail), 95)
            
            binary_defects = np.abs(detail) > threshold
            
            defect_maps.append({
                'level': level,
                'detail_magnitude': np.abs(detail),
                'threshold': threshold,
                'defects': binary_defects,
                'reconstruction': detail
            })
    
    # Combine defects from all levels
    combined_defects = np.zeros_like(image, dtype=bool)
    for defect_map in defect_maps:
        combined_defects |= defect_map['defects']
    
    # Apply mask
    combined_defects = combined_defects & mask
    
    # Post-processing
    combined_defects = morphology.remove_small_objects(combined_defects, min_size=3)
    combined_defects = morphology.opening(combined_defects, morphology.disk(1))
    
    # Calculate statistics
    total_pixels = np.sum(mask)
    defect_count = np.sum(combined_defects)
    defect_percentage = (defect_count / total_pixels) * 100 if total_pixels > 0 else 0
    
    statistics = {
        'wavelet': wavelet,
        'levels': levels,
        'threshold_method': threshold_method,
        'threshold_mode': threshold_mode,
        'total_pixels': total_pixels,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'level_statistics': []
    }
    
    # Add per-level statistics
    for defect_map in defect_maps:
        level_defects = np.sum(defect_map['defects'] & mask)
        statistics['level_statistics'].append({
            'level': defect_map['level'],
            'threshold': defect_map['threshold'],
            'defect_count': level_defects,
            'defect_percentage': (level_defects / total_pixels) * 100 if total_pixels > 0 else 0
        })
    
    print(f"Wavelet Detection Results:")
    print(f"  Total defects: {defect_count} ({defect_percentage:.2f}%)")
    for stat in statistics['level_statistics']:
        print(f"  Level {stat['level']}: {stat['defect_count']} defects ({stat['defect_percentage']:.2f}%)")
    
    return {
        'defects': combined_defects,
        'level_results': defect_maps,
        'statistics': statistics,
        'pywt_available': PYWT_AVAILABLE
    }


def multi_wavelet_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                           wavelets: List[str] = ['db4', 'haar', 'bior2.2'],
                           levels: int = 3,
                           ensemble_method: str = 'voting') -> Dict:
    """
    Multi-wavelet defect detection using different wavelet families.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        wavelets: List of wavelet families to use
        levels: Number of decomposition levels
        ensemble_method: Method to combine results ('voting', 'union', 'intersection')
    
    Returns:
        Multi-wavelet detection results
    """
    if not PYWT_AVAILABLE and len(wavelets) > 1:
        print("Warning: Multiple wavelets require PyWavelets. Using fallback with single method.")
        wavelets = ['fallback']
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    print(f"Multi-wavelet detection using: {wavelets}")
    
    wavelet_results = []
    combined_votes = np.zeros_like(image, dtype=int)
    
    for wavelet in wavelets:
        print(f"\nProcessing with {wavelet} wavelet...")
        
        if wavelet == 'fallback' or not PYWT_AVAILABLE:
            # Use fallback implementation
            result = wavelet_defect_detection(
                image, mask, 
                wavelet='db4',  # Dummy value for fallback
                levels=levels,
                threshold_method='adaptive',
                threshold_mode='percentile'
            )
        else:
            result = wavelet_defect_detection(
                image, mask,
                wavelet=wavelet,
                levels=levels,
                threshold_method='soft',
                threshold_mode='auto'
            )
        
        wavelet_results.append({
            'wavelet': wavelet,
            'result': result
        })
        
        # Add votes
        combined_votes += result['defects'].astype(int)
    
    # Combine results based on ensemble method
    if ensemble_method == 'voting':
        # Require majority vote
        threshold = len(wavelets) // 2 + 1
        final_defects = combined_votes >= threshold
    elif ensemble_method == 'union':
        # Any wavelet detects defect
        final_defects = combined_votes > 0
    elif ensemble_method == 'intersection':
        # All wavelets must detect defect
        final_defects = combined_votes == len(wavelets)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # Post-processing
    final_defects = morphology.remove_small_objects(final_defects, min_size=5)
    final_defects = morphology.opening(final_defects, morphology.disk(1))
    
    # Statistics
    total_pixels = np.sum(mask)
    final_count = np.sum(final_defects)
    
    comprehensive_stats = {
        'multi_wavelet': True,
        'wavelets_used': wavelets,
        'levels': levels,
        'ensemble_method': ensemble_method,
        'total_defects': final_count,
        'defect_percentage': (final_count / total_pixels) * 100 if total_pixels > 0 else 0,
        'wavelet_results': wavelet_results,
        'vote_distribution': np.bincount(combined_votes.flatten())
    }
    
    print(f"\nMulti-wavelet Results:")
    print(f"  Ensemble method: {ensemble_method}")
    print(f"  Final defects: {final_count} ({comprehensive_stats['defect_percentage']:.2f}%)")
    
    return {
        'defects': final_defects,
        'vote_map': combined_votes,
        'wavelet_results': wavelet_results,
        'statistics': comprehensive_stats
    }


def adaptive_wavelet_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                              auto_select_wavelet: bool = True,
                              auto_select_levels: bool = True) -> Dict:
    """
    Adaptive wavelet detection with automatic parameter selection.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        auto_select_wavelet: Automatically select best wavelet family
        auto_select_levels: Automatically select optimal decomposition levels
    
    Returns:
        Adaptive wavelet detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    print("Adaptive wavelet detection with automatic parameter selection...")
    
    # Analyze image properties
    valid_pixels = image[mask]
    if len(valid_pixels) == 0:
        return {
            'defects': np.zeros_like(mask, dtype=bool),
            'statistics': {'error': 'Empty mask'}
        }
    
    # Image analysis for parameter selection
    image_std = np.std(valid_pixels)
    image_mean = np.mean(valid_pixels)
    noise_level = image_std / image_mean
    
    # Analyze texture/detail content
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges & mask) / np.sum(mask)
    
    print(f"Image analysis: noise_level={noise_level:.3f}, edge_density={edge_density:.3f}")
    
    # Select wavelet family based on image properties
    if auto_select_wavelet:
        if PYWT_AVAILABLE:
            if edge_density > 0.1:  # High detail content
                selected_wavelet = 'db8'  # Good for detailed features
            elif noise_level > 0.1:  # Noisy image
                selected_wavelet = 'bior2.2'  # Good noise characteristics
            else:  # General purpose
                selected_wavelet = 'db4'  # Balanced choice
        else:
            selected_wavelet = 'fallback'
    else:
        selected_wavelet = 'db4'
    
    # Select decomposition levels based on image size and content
    if auto_select_levels:
        min_size = min(image.shape)
        max_levels = int(np.log2(min_size)) - 2  # Leave some resolution
        
        if edge_density > 0.15:  # High detail
            selected_levels = min(max_levels, 4)
        elif noise_level > 0.15:  # High noise
            selected_levels = min(max_levels, 2)  # Fewer levels to avoid noise
        else:
            selected_levels = min(max_levels, 3)  # Standard
    else:
        selected_levels = 3
    
    print(f"Selected parameters: wavelet={selected_wavelet}, levels={selected_levels}")
    
    # Run wavelet detection with selected parameters
    result = wavelet_defect_detection(
        image, mask,
        wavelet=selected_wavelet,
        levels=selected_levels,
        threshold_method='soft',
        threshold_mode='auto'
    )
    
    # Add adaptation info
    result['statistics'].update({
        'adaptive': True,
        'image_analysis': {
            'noise_level': noise_level,
            'edge_density': edge_density,
            'image_std': image_std,
            'image_mean': image_mean
        },
        'parameter_selection': {
            'auto_select_wavelet': auto_select_wavelet,
            'auto_select_levels': auto_select_levels,
            'selected_wavelet': selected_wavelet,
            'selected_levels': selected_levels
        }
    })
    
    return result


def visualize_wavelet_results(image: np.ndarray, results: Dict, save_path: Optional[str] = None):
    """
    Visualize wavelet detection results.
    
    Args:
        image: Original input image
        results: Results from wavelet detection functions
        save_path: Optional path to save visualization
    """
    if 'wavelet_results' in results:
        # Multi-wavelet results
        num_wavelets = len(results['wavelet_results'])
        fig, axes = plt.subplots(3, num_wavelets + 1, figsize=(4 * (num_wavelets + 1), 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Final combined result
        axes[1, 0].imshow(results['defects'], cmap='hot')
        axes[1, 0].set_title(f"Combined Defects\n({results['statistics']['ensemble_method']})")
        axes[1, 0].axis('off')
        
        # Vote map
        if 'vote_map' in results:
            axes[2, 0].imshow(results['vote_map'], cmap='viridis')
            axes[2, 0].set_title('Vote Map')
            axes[2, 0].axis('off')
        
        # Individual wavelet results
        for i, wavelet_result in enumerate(results['wavelet_results']):
            wavelet_name = wavelet_result['wavelet']
            result = wavelet_result['result']
            
            # Show defects
            axes[0, i + 1].imshow(result['defects'], cmap='hot')
            axes[0, i + 1].set_title(f'Defects ({wavelet_name})')
            axes[0, i + 1].axis('off')
            
            # Show one level's detail magnitude
            if result['level_results']:
                level_result = result['level_results'][0]  # Show first level
                axes[1, i + 1].imshow(level_result['detail_magnitude'], cmap='viridis')
                axes[1, i + 1].set_title(f'Details L1 ({wavelet_name})')
                axes[1, i + 1].axis('off')
                
                # Show reconstruction
                axes[2, i + 1].imshow(level_result['reconstruction'], cmap='gray')
                axes[2, i + 1].set_title(f'Reconstruction ({wavelet_name})')
                axes[2, i + 1].axis('off')
    
    else:
        # Single wavelet results
        num_levels = len(results['level_results'])
        fig, axes = plt.subplots(3, min(num_levels + 1, 5), figsize=(20, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Combined defects
        axes[1, 0].imshow(results['defects'], cmap='hot')
        axes[1, 0].set_title('Combined Defects')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        overlay[results['defects']] = [255, 0, 0]
        axes[2, 0].imshow(overlay)
        axes[2, 0].set_title('Defects Overlay')
        axes[2, 0].axis('off')
        
        # Individual levels (up to 4)
        for i, level_result in enumerate(results['level_results'][:4]):
            col = i + 1
            level = level_result['level']
            
            # Detail magnitude
            axes[0, col].imshow(level_result['detail_magnitude'], cmap='viridis')
            axes[0, col].set_title(f'Detail Magnitude L{level}')
            axes[0, col].axis('off')
            
            # Level defects
            axes[1, col].imshow(level_result['defects'], cmap='hot')
            axes[1, col].set_title(f'Level {level} Defects')
            axes[1, col].axis('off')
            
            # Reconstruction
            axes[2, col].imshow(level_result['reconstruction'], cmap='gray')
            axes[2, col].set_title(f'Level {level} Reconstruction')
            axes[2, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


# Demo and test code
if __name__ == "__main__":
    print("Wavelet Transform Analysis Module - Demo")
    print("=" * 45)
    
    if PYWT_AVAILABLE:
        print("PyWavelets is available - using full wavelet functionality")
    else:
        print("PyWavelets not available - using fallback implementations")
    
    # Create synthetic test image with various defects
    print("\nCreating synthetic test image with multi-scale defects...")
    np.random.seed(42)
    
    # Base image
    test_image = np.random.randint(120, 140, (200, 200), dtype=np.uint8)
    
    # Add different scale defects
    # Large scale defects (low frequency)
    cv2.circle(test_image, (50, 50), 15, 80, -1)
    cv2.circle(test_image, (150, 150), 12, 200, -1)
    
    # Medium scale defects
    cv2.circle(test_image, (100, 50), 6, 70, -1)
    cv2.circle(test_image, (50, 150), 8, 180, -1)
    
    # Small scale defects (high frequency)
    for i in range(10):
        x = np.random.randint(20, 180)
        y = np.random.randint(20, 180)
        test_image[y:y+2, x:x+2] = np.random.randint(60, 220)
    
    # Linear defects (scratches)
    cv2.line(test_image, (30, 100), (170, 120), 90, 2)
    
    # Add noise
    noise = np.random.normal(0, 3, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create circular mask
    center = (100, 100)
    radius = 80
    y, x = np.ogrid[:200, :200]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Added multi-scale defects and noise")
    print(f"Mask coverage: {np.sum(mask)} pixels")
    
    # Test basic wavelet detection
    print("\n1. Testing basic wavelet detection...")
    if PYWT_AVAILABLE:
        basic_results = wavelet_defect_detection(
            test_image, mask,
            wavelet='db4',
            levels=3,
            threshold_method='soft',
            threshold_mode='auto'
        )
    else:
        basic_results = wavelet_defect_detection(test_image, mask)
    
    # Test adaptive wavelet detection
    print("\n2. Testing adaptive wavelet detection...")
    adaptive_results = adaptive_wavelet_detection(
        test_image, mask,
        auto_select_wavelet=True,
        auto_select_levels=True
    )
    
    # Test multi-wavelet detection (if PyWavelets available)
    if PYWT_AVAILABLE:
        print("\n3. Testing multi-wavelet detection...")
        multi_results = multi_wavelet_detection(
            test_image, mask,
            wavelets=['db4', 'haar', 'bior2.2'],
            levels=3,
            ensemble_method='voting'
        )
    else:
        print("\n3. Skipping multi-wavelet detection (requires PyWavelets)")
        multi_results = None
    
    # Visualize results
    print("\n4. Visualizing results...")
    
    # Basic wavelet visualization
    fig1 = visualize_wavelet_results(test_image, basic_results)
    plt.suptitle('Basic Wavelet Detection', fontsize=16)
    
    # Adaptive wavelet visualization
    fig2 = visualize_wavelet_results(test_image, adaptive_results)
    plt.suptitle('Adaptive Wavelet Detection', fontsize=16)
    
    # Multi-wavelet visualization (if available)
    if multi_results is not None:
        fig3 = visualize_wavelet_results(test_image, multi_results)
        plt.suptitle('Multi-Wavelet Detection', fontsize=16)
    
    # Performance comparison
    print("\n5. Performance comparison:")
    basic_count = basic_results['statistics']['defect_count']
    adaptive_count = adaptive_results['statistics']['defect_count']
    
    print(f"Basic wavelet: {basic_count} defects ({basic_results['statistics']['defect_percentage']:.2f}%)")
    print(f"Adaptive wavelet: {adaptive_count} defects ({adaptive_results['statistics']['defect_percentage']:.2f}%)")
    
    if multi_results is not None:
        multi_count = multi_results['statistics']['total_defects']
        print(f"Multi-wavelet: {multi_count} defects ({multi_results['statistics']['defect_percentage']:.2f}%)")
    
    # Level analysis
    print("\n6. Level-by-level analysis (basic wavelet):")
    for level_stat in basic_results['statistics']['level_statistics']:
        print(f"  Level {level_stat['level']}: {level_stat['defect_count']} defects "
              f"({level_stat['defect_percentage']:.2f}%), threshold={level_stat['threshold']:.2f}")
    
    # Adaptive parameter analysis
    if 'parameter_selection' in adaptive_results['statistics']:
        params = adaptive_results['statistics']['parameter_selection']
        analysis = adaptive_results['statistics']['image_analysis']
        print(f"\n7. Adaptive parameter selection:")
        print(f"  Selected wavelet: {params['selected_wavelet']}")
        print(f"  Selected levels: {params['selected_levels']}")
        print(f"  Image noise level: {analysis['noise_level']:.3f}")
        print(f"  Edge density: {analysis['edge_density']:.3f}")
    
    print("\nDemo completed successfully!")
