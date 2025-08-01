"""
LEI (Linear Enhancement Inspector) Scratch Detection Module

This module implements the LEI method for detecting linear scratches in fiber optic 
end face images. The method uses oriented linear detectors with statistical analysis
to identify scratch-like defects.

Based on research paper implementations found in defect_detection2.py
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List


def lei_scratch_detection(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                         orientations: int = 12, line_length: int = 15, 
                         line_gap: int = 3, threshold_sigma: float = 2.0,
                         histogram_equalization: bool = True) -> Dict:
    """
    LEI (Linear Enhancement Inspector) scratch detection method.
    
    This method creates linear detectors at multiple orientations to identify
    scratch-like defects. It compares intensity along the scratch (red branch)
    with intensity parallel to the scratch (gray branches).
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask to limit analysis region
        orientations: Number of orientations to test (default: 12)
        line_length: Length of the linear detector (default: 15)
        line_gap: Gap between red and gray branches (default: 3)
        threshold_sigma: Statistical threshold in standard deviations (default: 2.0)
        histogram_equalization: Apply histogram equalization preprocessing (default: True)
    
    Returns:
        Dictionary containing:
            - 'scratches': Binary scratch mask
            - 'strength_map': Scratch strength at each pixel
            - 'orientation_map': Dominant orientation for each scratch
            - 'statistics': Detection statistics
            - 'preprocessed': Preprocessed image (if equalization applied)
    """
    # Input validation
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # Preprocessing: histogram equalization (as per LEI paper)
    if histogram_equalization:
        preprocessed = cv2.equalizeHist(image)
        print("Applied histogram equalization preprocessing")
    else:
        preprocessed = image.copy()
    
    # Initialize outputs
    height, width = image.shape
    scratch_strength = np.zeros_like(image, dtype=np.float32)
    orientation_map = np.zeros_like(image, dtype=np.float32)
    
    print(f"Starting LEI detection with {orientations} orientations...")
    
    # Process each orientation
    for i in range(orientations):
        angle = i * 180.0 / orientations
        angle_rad = np.radians(angle)
        
        print(f"  Processing orientation {i+1}/{orientations} (angle: {angle:.1f}°)")
        
        # Calculate orientation-specific strength
        orientation_strength = _calculate_lei_strength(
            preprocessed, mask, angle_rad, line_length, line_gap
        )
        
        # Update maximum strength and corresponding orientation
        better_mask = orientation_strength > scratch_strength
        scratch_strength[better_mask] = orientation_strength[better_mask]
        orientation_map[better_mask] = angle
    
    # Statistical thresholding
    if np.sum(mask) == 0:
        print("Warning: Empty mask provided")
        threshold_value = 0
        scratches = np.zeros_like(image, dtype=bool)
    else:
        masked_strength = scratch_strength[mask]
        mean_strength = np.mean(masked_strength)
        std_strength = np.std(masked_strength)
        threshold_value = mean_strength + threshold_sigma * std_strength
        
        # Create scratch mask
        scratches = (scratch_strength > threshold_value) & mask
    
    # Post-processing: connect broken scratches
    print("Applying post-processing...")
    scratches = morphology.closing(scratches, morphology.disk(2))
    scratches = morphology.remove_small_objects(scratches, min_size=5)
    
    # Compile statistics
    statistics = {
        'orientations_tested': orientations,
        'line_length': line_length,
        'line_gap': line_gap,
        'threshold_sigma': threshold_sigma,
        'threshold_value': threshold_value,
        'mean_strength': mean_strength if np.sum(mask) > 0 else 0,
        'std_strength': std_strength if np.sum(mask) > 0 else 0,
        'scratch_count': np.sum(scratches),
        'scratch_percentage': (np.sum(scratches) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0,
        'max_strength': np.max(scratch_strength),
        'min_strength': np.min(scratch_strength)
    }
    
    print(f"LEI Detection Results:")
    print(f"  Threshold: {threshold_value:.2f}")
    print(f"  Scratches found: {statistics['scratch_count']}")
    print(f"  Scratch percentage: {statistics['scratch_percentage']:.2f}%")
    
    result = {
        'scratches': scratches,
        'strength_map': scratch_strength,
        'orientation_map': orientation_map,
        'statistics': statistics
    }
    
    if histogram_equalization:
        result['preprocessed'] = preprocessed
    
    return result


def _calculate_lei_strength(image: np.ndarray, mask: np.ndarray, angle_rad: float,
                           line_length: int, line_gap: int) -> np.ndarray:
    """
    Calculate LEI strength for a specific orientation.
    
    Args:
        image: Preprocessed image
        mask: Binary mask
        angle_rad: Angle in radians
        line_length: Length of linear detector
        line_gap: Gap between branches
    
    Returns:
        Strength map for this orientation
    """
    height, width = image.shape
    strength = np.zeros_like(image, dtype=np.float32)
    
    # Precompute trigonometric values
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Process each pixel
    for y in range(line_gap, height - line_gap):
        for x in range(line_gap, width - line_gap):
            if not mask[y, x]:
                continue
            
            # Sample points along the red branch (along the line)
            red_values = []
            gray_values = []
            
            # Red branch: sample along the line direction
            for t in range(-line_length // 2, line_length // 2 + 1):
                # Red branch point
                rx = int(x + t * cos_angle)
                ry = int(y + t * sin_angle)
                
                if 0 <= rx < width and 0 <= ry < height:
                    red_values.append(image[ry, rx])
                
                # Gray branch points (offset perpendicular to line)
                for offset in [-line_gap, line_gap]:
                    gx = int(x + t * cos_angle + offset * sin_angle)
                    gy = int(y + t * sin_angle - offset * cos_angle)
                    
                    if 0 <= gx < width and 0 <= gy < height:
                        gray_values.append(image[gy, gx])
            
            # Calculate LEI strength: 2 * red_avg - gray_avg
            if len(red_values) > 0 and len(gray_values) > 0:
                red_avg = np.mean(red_values)
                gray_avg = np.mean(gray_values)
                strength[y, x] = 2 * red_avg - gray_avg
    
    return strength


def enhanced_lei_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                          multi_scale: bool = True, 
                          scales: List[Tuple[int, int]] = [(10, 2), (15, 3), (20, 4)],
                          adaptive_threshold: bool = True,
                          orientation_refinement: bool = True) -> Dict:
    """
    Enhanced LEI detection with multi-scale analysis and adaptive thresholding.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        multi_scale: Use multiple scales (default: True)
        scales: List of (line_length, line_gap) tuples (default: [(10,2), (15,3), (20,4)])
        adaptive_threshold: Use adaptive thresholding (default: True)
        orientation_refinement: Refine orientations around peaks (default: True)
    
    Returns:
        Enhanced detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    if not multi_scale:
        # Single scale detection with default parameters
        return lei_scratch_detection(image, mask)
    
    print("Performing enhanced multi-scale LEI detection...")
    
    # Store results for each scale
    scale_results = []
    combined_scratches = np.zeros_like(image, dtype=bool)
    combined_strength = np.zeros_like(image, dtype=np.float32)
    
    for i, (line_length, line_gap) in enumerate(scales):
        print(f"\nScale {i+1}/{len(scales)}: length={line_length}, gap={line_gap}")
        
        # Determine number of orientations based on scale
        if orientation_refinement:
            orientations = 18 if line_length >= 15 else 12
        else:
            orientations = 12
        
        # Adaptive threshold sigma based on scale
        if adaptive_threshold:
            # Larger scales use more conservative thresholds
            threshold_sigma = 1.5 + (line_length - 10) * 0.1
        else:
            threshold_sigma = 2.0
        
        result = lei_scratch_detection(
            image, mask,
            orientations=orientations,
            line_length=line_length,
            line_gap=line_gap,
            threshold_sigma=threshold_sigma
        )
        
        scale_results.append({
            'scale': (line_length, line_gap),
            'orientations': orientations,
            'threshold_sigma': threshold_sigma,
            'result': result
        })
        
        # Combine results
        combined_scratches |= result['scratches']
        combined_strength = np.maximum(combined_strength, result['strength_map'])
    
    # Advanced post-processing for combined results
    print("\nApplying advanced post-processing...")
    
    # Remove isolated pixels
    combined_scratches = morphology.opening(combined_scratches, morphology.disk(1))
    
    # Connect nearby scratch segments
    combined_scratches = morphology.closing(combined_scratches, morphology.disk(3))
    
    # Remove very small objects
    combined_scratches = morphology.remove_small_objects(combined_scratches, min_size=10)
    
    # Fill small gaps in scratch lines
    combined_scratches = morphology.closing(combined_scratches, morphology.rectangle(1, 5))
    combined_scratches = morphology.closing(combined_scratches, morphology.rectangle(5, 1))
    
    # Compile comprehensive statistics
    comprehensive_stats = {
        'multi_scale': True,
        'scales_used': scales,
        'total_scratches': np.sum(combined_scratches),
        'scratch_percentage': (np.sum(combined_scratches) / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0,
        'max_combined_strength': np.max(combined_strength),
        'scale_results': scale_results
    }
    
    print(f"\nEnhanced LEI Results:")
    print(f"  Total scratches: {comprehensive_stats['total_scratches']}")
    print(f"  Scratch percentage: {comprehensive_stats['scratch_percentage']:.2f}%")
    
    return {
        'scratches': combined_scratches,
        'strength_map': combined_strength,
        'scale_results': scale_results,
        'statistics': comprehensive_stats
    }


def analyze_scratch_properties(scratches: np.ndarray, orientation_map: np.ndarray) -> Dict:
    """
    Analyze properties of detected scratches.
    
    Args:
        scratches: Binary scratch mask
        orientation_map: Orientation map from LEI detection
    
    Returns:
        Dictionary with scratch analysis results
    """
    # Label individual scratch objects
    labeled_scratches, num_scratches = ndimage.label(scratches)
    
    if num_scratches == 0:
        return {'num_scratches': 0, 'scratch_properties': []}
    
    print(f"Analyzing {num_scratches} detected scratches...")
    
    scratch_properties = []
    
    for i in range(1, num_scratches + 1):
        scratch_mask = labeled_scratches == i
        
        # Basic properties
        area = np.sum(scratch_mask)
        if area == 0:
            continue
        
        # Centroid
        y_coords, x_coords = np.where(scratch_mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        # Bounding box
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        # Dimensions
        length = np.sqrt((max_y - min_y)**2 + (max_x - min_x)**2)
        width = area / (length + 1e-10)
        
        # Aspect ratio
        aspect_ratio = length / (width + 1e-10)
        
        # Average orientation
        scratch_orientations = orientation_map[scratch_mask]
        avg_orientation = np.mean(scratch_orientations)
        orientation_std = np.std(scratch_orientations)
        
        # Linearity measure (how well it fits a line)
        # Fit line to scratch points
        if len(y_coords) >= 2:
            coeffs = np.polyfit(x_coords, y_coords, 1)
            fitted_y = np.polyval(coeffs, x_coords)
            linearity = 1.0 - np.mean(np.abs(y_coords - fitted_y)) / (length + 1e-10)
        else:
            linearity = 0.0
        
        properties = {
            'id': i,
            'area': area,
            'centroid': (centroid_x, centroid_y),
            'bounding_box': (min_x, min_y, max_x, max_y),
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'avg_orientation': avg_orientation,
            'orientation_std': orientation_std,
            'linearity': linearity
        }
        
        scratch_properties.append(properties)
    
    return {
        'num_scratches': num_scratches,
        'scratch_properties': scratch_properties
    }


def visualize_lei_results(image: np.ndarray, results: Dict, save_path: Optional[str] = None):
    """
    Visualize LEI detection results.
    
    Args:
        image: Original input image
        results: Results from lei_scratch_detection or enhanced_lei_detection
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
        
        # Combined results
        axes[1, 0].imshow(results['strength_map'], cmap='viridis')
        axes[1, 0].set_title('Combined Strength')
        axes[1, 0].axis('off')
        
        axes[2, 0].imshow(results['scratches'], cmap='hot')
        axes[2, 0].set_title('Combined Scratches')
        axes[2, 0].axis('off')
        
        # Individual scale results
        for i, scale_result in enumerate(results['scale_results']):
            scale = scale_result['scale']
            result = scale_result['result']
            
            axes[0, i + 1].imshow(result.get('preprocessed', image), cmap='gray')
            axes[0, i + 1].set_title(f'Preprocessed (L={scale[0]}, G={scale[1]})')
            axes[0, i + 1].axis('off')
            
            axes[1, i + 1].imshow(result['strength_map'], cmap='viridis')
            axes[1, i + 1].set_title(f'Strength (L={scale[0]}, G={scale[1]})')
            axes[1, i + 1].axis('off')
            
            axes[2, i + 1].imshow(result['scratches'], cmap='hot')
            axes[2, i + 1].set_title(f'Scratches (L={scale[0]}, G={scale[1]})')
            axes[2, i + 1].axis('off')
    
    else:
        # Single scale results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        if 'preprocessed' in results:
            axes[0, 1].imshow(results['preprocessed'], cmap='gray')
            axes[0, 1].set_title('Preprocessed (Hist. Eq.)')
        else:
            axes[0, 1].imshow(image, cmap='gray')
            axes[0, 1].set_title('Input Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['strength_map'], cmap='viridis')
        axes[0, 2].set_title('Scratch Strength Map')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(results['orientation_map'], cmap='hsv')
        axes[1, 0].set_title('Orientation Map')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['scratches'], cmap='hot')
        axes[1, 1].set_title('Detected Scratches')
        axes[1, 1].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        overlay[results['scratches']] = [255, 0, 0]  # Red scratches
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Scratches Overlay')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


# Demo and test code
if __name__ == "__main__":
    print("LEI (Linear Enhancement Inspector) Detection Module - Demo")
    print("=" * 60)
    
    # Create synthetic test image with scratches
    print("Creating synthetic test image with scratches...")
    test_image = np.random.randint(100, 150, (300, 300), dtype=np.uint8)
    
    # Add linear scratches at different orientations
    # Horizontal scratch
    test_image[50:52, 30:150] = 80
    
    # Vertical scratch
    test_image[80:200, 200:202] = 70
    
    # Diagonal scratch
    for i in range(100):
        y = 100 + i
        x = 50 + i
        if y < 300 and x < 300:
            test_image[y:y+2, x:x+2] = 60
    
    # Another diagonal (opposite direction)
    for i in range(80):
        y = 200 + i
        x = 250 - i
        if y < 300 and x >= 0:
            test_image[y:y+2, x:x+2] = 90
    
    # Add some noise
    noise = np.random.normal(0, 3, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create circular mask
    center = (150, 150)
    radius = 120
    y, x = np.ogrid[:300, :300]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Added 4 artificial scratches")
    print(f"Mask coverage: {np.sum(mask)} pixels")
    
    # Test single-scale LEI
    print("\n1. Testing single-scale LEI detection...")
    single_results = lei_scratch_detection(
        test_image, mask,
        orientations=12,
        line_length=15,
        line_gap=3,
        threshold_sigma=2.0
    )
    
    # Test multi-scale LEI
    print("\n2. Testing enhanced multi-scale LEI detection...")
    multi_results = enhanced_lei_detection(
        test_image, mask,
        multi_scale=True,
        scales=[(10, 2), (15, 3), (20, 4)],
        adaptive_threshold=True,
        orientation_refinement=True
    )
    
    # Analyze scratch properties
    print("\n3. Analyzing scratch properties...")
    if 'orientation_map' in single_results:
        scratch_analysis = analyze_scratch_properties(
            single_results['scratches'],
            single_results['orientation_map']
        )
        
        print(f"Found {scratch_analysis['num_scratches']} individual scratches")
        for prop in scratch_analysis['scratch_properties'][:5]:  # Show first 5
            print(f"  Scratch {prop['id']}: area={prop['area']}, aspect_ratio={prop['aspect_ratio']:.1f}, "
                  f"orientation={prop['avg_orientation']:.1f}°")
    
    # Visualize results
    print("\n4. Visualizing results...")
    
    # Single scale visualization
    fig1 = visualize_lei_results(test_image, single_results)
    plt.suptitle('Single-Scale LEI Detection', fontsize=16)
    
    # Multi-scale visualization
    fig2 = visualize_lei_results(test_image, multi_results)
    plt.suptitle('Multi-Scale LEI Detection', fontsize=16)
    
    # Performance comparison
    print("\n5. Performance comparison:")
    print(f"Single-scale scratches: {single_results['statistics']['scratch_count']}")
    print(f"Multi-scale scratches: {multi_results['statistics']['total_scratches']}")
    improvement = multi_results['statistics']['total_scratches'] - single_results['statistics']['scratch_count']
    print(f"Improvement: {improvement} additional scratch pixels")
    
    print("\nDemo completed successfully!")
