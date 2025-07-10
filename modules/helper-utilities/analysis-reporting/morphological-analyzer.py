#!/usr/bin/env python3
"""
Morphological Analysis Module for Fiber Optic Defect Detection
===============================================================

This module implements morphological image processing methods for detecting
various types of defects in fiber optic end-face images.

Functions:
- Top-hat and bottom-hat transforms
- Morphological gradient
- Opening and closing operations
- Hit-or-miss transforms
- Watershed segmentation
- Skeletonization

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
try:
    from skimage import morphology, measure, segmentation
    from skimage.morphology import skeletonize, medial_axis
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some morphological operations will use manual implementations.")
    SKIMAGE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def tophat_bottomhat_analysis(image, kernel_sizes=[3, 5, 7, 9], mask=None):
    """
    Top-hat and bottom-hat morphological transforms for defect detection.
    
    Top-hat: Detects bright defects (brighter than local background)
    Bottom-hat: Detects dark defects (darker than local background)
    
    Args:
        image (np.ndarray): Input grayscale image
        kernel_sizes (list): List of structuring element sizes
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Top-hat and bottom-hat results for all kernel sizes
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    for kernel_size in kernel_sizes:
        # Create elliptical structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Top-hat transform (bright defects)
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Bottom-hat transform (dark defects)
        bottomhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold for defect detection
        tophat_threshold = np.percentile(tophat[mask], 95) if np.sum(mask) > 0 else np.percentile(tophat, 95)
        bottomhat_threshold = np.percentile(bottomhat[mask], 95) if np.sum(mask) > 0 else np.percentile(bottomhat, 95)
        
        # Detect defects
        bright_defects = (tophat > tophat_threshold) & mask
        dark_defects = (bottomhat > bottomhat_threshold) & mask
        
        results[f'kernel_{kernel_size}'] = {
            'tophat': tophat,
            'bottomhat': bottomhat,
            'bright_defects': bright_defects,
            'dark_defects': dark_defects,
            'tophat_threshold': tophat_threshold,
            'bottomhat_threshold': bottomhat_threshold,
            'bright_count': np.sum(bright_defects),
            'dark_count': np.sum(dark_defects)
        }
    
    # Combine results from all kernel sizes
    combined_bright = np.zeros_like(mask, dtype=bool)
    combined_dark = np.zeros_like(mask, dtype=bool)
    
    for result in results.values():
        combined_bright |= result['bright_defects']
        combined_dark |= result['dark_defects']
    
    results['combined'] = {
        'bright_defects': combined_bright,
        'dark_defects': combined_dark,
        'all_defects': combined_bright | combined_dark,
        'bright_count': np.sum(combined_bright),
        'dark_count': np.sum(combined_dark),
        'total_count': np.sum(combined_bright | combined_dark)
    }
    
    return results


def morphological_gradient_analysis(image, kernel_sizes=[3, 5, 7], mask=None):
    """
    Morphological gradient for edge and boundary detection.
    
    Gradient = Dilation - Erosion (highlights boundaries)
    
    Args:
        image (np.ndarray): Input grayscale image
        kernel_sizes (list): List of structuring element sizes
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Gradient results and detected edges
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    for kernel_size in kernel_sizes:
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Morphological gradient
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        # Internal gradient (erosion - original)
        eroded = cv2.erode(image, kernel)
        internal_gradient = image - eroded
        
        # External gradient (dilation - original)
        dilated = cv2.dilate(image, kernel)
        external_gradient = dilated - image
        
        # Threshold for edge detection
        gradient_threshold = np.percentile(gradient[mask], 95) if np.sum(mask) > 0 else np.percentile(gradient, 95)
        
        # Detect edges
        edges = (gradient > gradient_threshold) & mask
        
        results[f'kernel_{kernel_size}'] = {
            'gradient': gradient,
            'internal_gradient': internal_gradient,
            'external_gradient': external_gradient,
            'edges': edges,
            'threshold': gradient_threshold,
            'edge_count': np.sum(edges)
        }
    
    # Combine gradients from all scales
    combined_gradient = np.zeros_like(image, dtype=np.float32)
    combined_edges = np.zeros_like(mask, dtype=bool)
    
    for result in results.values():
        combined_gradient = np.maximum(combined_gradient, result['gradient'].astype(np.float32))
        combined_edges |= result['edges']
    
    results['combined'] = {
        'gradient': combined_gradient,
        'edges': combined_edges,
        'edge_count': np.sum(combined_edges)
    }
    
    return results


def opening_closing_analysis(image, kernel_sizes=[3, 5, 7], mask=None):
    """
    Morphological opening and closing for noise removal and defect detection.
    
    Opening: Erosion followed by dilation (removes small bright objects)
    Closing: Dilation followed by erosion (fills small dark holes)
    
    Args:
        image (np.ndarray): Input grayscale image
        kernel_sizes (list): List of structuring element sizes
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Opening and closing results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    for kernel_size in kernel_sizes:
        # Create structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Opening operation
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing operation
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Differences reveal defects
        open_diff = image - opened  # Small bright objects removed by opening
        close_diff = closed - image  # Small dark holes filled by closing
        
        # Threshold differences
        open_threshold = np.percentile(open_diff[mask], 95) if np.sum(mask) > 0 else np.percentile(open_diff, 95)
        close_threshold = np.percentile(close_diff[mask], 95) if np.sum(mask) > 0 else np.percentile(close_diff, 95)
        
        # Detect defects
        bright_noise = (open_diff > open_threshold) & mask
        dark_holes = (close_diff > close_threshold) & mask
        
        results[f'kernel_{kernel_size}'] = {
            'opened': opened,
            'closed': closed,
            'open_diff': open_diff,
            'close_diff': close_diff,
            'bright_noise': bright_noise,
            'dark_holes': dark_holes,
            'open_threshold': open_threshold,
            'close_threshold': close_threshold,
            'bright_count': np.sum(bright_noise),
            'dark_count': np.sum(dark_holes)
        }
    
    return results


def hit_or_miss_detection(image, templates=None, mask=None):
    """
    Hit-or-miss transform for detecting specific patterns.
    
    Args:
        image (np.ndarray): Input binary or grayscale image
        templates (list, optional): List of structuring element pairs [(hit, miss), ...]
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Hit-or-miss results for each template
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Convert to binary
    binary_image = image > np.mean(image)
    
    if templates is None:
        # Default templates for common defect patterns
        templates = []
        
        # Cross pattern (for detecting cross-shaped defects)
        cross_hit = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        cross_miss = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
        templates.append(('cross', cross_hit, cross_miss))
        
        # Line patterns (for detecting linear defects)
        line_h_hit = np.array([[1, 1, 1]], dtype=np.uint8)
        line_h_miss = np.array([[0, 0, 0]], dtype=np.uint8)
        templates.append(('line_horizontal', line_h_hit, line_h_miss))
        
        line_v_hit = np.array([[1], [1], [1]], dtype=np.uint8)
        line_v_miss = np.array([[0], [0], [0]], dtype=np.uint8)
        templates.append(('line_vertical', line_v_hit, line_v_miss))
        
        # Corner pattern
        corner_hit = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        corner_miss = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        templates.append(('corner', corner_hit, corner_miss))
    
    results = {}
    
    for template_name, hit_kernel, miss_kernel in templates:
        # Perform hit-or-miss transform
        hit_result = cv2.erode(binary_image.astype(np.uint8), hit_kernel)
        miss_result = cv2.erode((~binary_image).astype(np.uint8), miss_kernel)
        
        # Combine results
        hmt_result = cv2.bitwise_and(hit_result, miss_result)
        
        # Apply mask
        hmt_result = hmt_result.astype(bool) & mask
        
        results[template_name] = {
            'hit_or_miss': hmt_result,
            'detection_count': np.sum(hmt_result),
            'hit_kernel': hit_kernel,
            'miss_kernel': miss_kernel
        }
    
    return results


def watershed_segmentation(image, mask=None, min_distance=10):
    """
    Watershed segmentation for separating touching defects.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
        min_distance (int): Minimum distance between peaks
    
    Returns:
        dict: Watershed segmentation results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Prepare image for watershed
    # Use distance transform to find local maxima
    binary_mask = mask.astype(np.uint8)
    
    # Distance transform
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    if SKIMAGE_AVAILABLE:
        from skimage.feature import peak_local_maxima
        from skimage.segmentation import watershed
        
        # Find local maxima
        local_maxima = peak_local_maxima(distance, min_distance=min_distance, threshold_abs=0.3*distance.max())
        
        # Create markers
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary_mask)
        
    else:
        # Manual implementation using OpenCV watershed
        # Find contours as approximate markers
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create markers
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i+1, -1)
        
        # Convert image to 3-channel for watershed
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Apply watershed
        cv2.watershed(image_3ch, markers)
        labels = markers
    
    # Count segments
    num_segments = len(np.unique(labels)) - 1  # Subtract background
    
    return {
        'labels': labels,
        'distance_transform': distance,
        'num_segments': num_segments,
        'binary_mask': binary_mask
    }


def skeletonization_analysis(image, mask=None):
    """
    Skeletonization for analyzing linear defects and structures.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Skeleton analysis results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Create binary image
    binary_image = image > np.mean(image[mask]) if np.sum(mask) > 0 else image > np.mean(image)
    binary_image = binary_image & mask
    
    if SKIMAGE_AVAILABLE:
        # Skeletonization
        skeleton = skeletonize(binary_image)
        
        # Medial axis
        medial_axis_img, distance = medial_axis(binary_image, return_distance=True)
        
    else:
        # Manual skeletonization using morphological operations
        skeleton = manual_skeletonize(binary_image)
        medial_axis_img = skeleton
        distance = cv2.distanceTransform(binary_image.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Analyze skeleton properties
    skeleton_pixels = np.sum(skeleton)
    
    # Find branch points and endpoints
    branch_points, endpoints = analyze_skeleton_topology(skeleton)
    
    return {
        'skeleton': skeleton,
        'medial_axis': medial_axis_img,
        'distance_map': distance,
        'skeleton_pixels': skeleton_pixels,
        'branch_points': branch_points,
        'endpoints': endpoints,
        'num_branches': len(branch_points),
        'num_endpoints': len(endpoints)
    }


def manual_skeletonize(binary_image, max_iterations=100):
    """
    Manual skeletonization using morphological thinning.
    
    Args:
        binary_image (np.ndarray): Binary input image
        max_iterations (int): Maximum number of iterations
    
    Returns:
        np.ndarray: Skeletonized image
    """
    # Zhang-Suen thinning algorithm (simplified)
    img = binary_image.astype(np.uint8)
    
    # Define 8-connectivity kernels for thinning
    kernels = []
    
    # Create thinning kernels
    for i in range(8):
        kernel = np.zeros((3, 3), dtype=np.uint8)
        if i == 0:  # Top
            kernel = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)
        elif i == 1:  # Top-right
            kernel = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.uint8)
        # Add more kernels for complete implementation...
        kernels.append(kernel)
    
    # Simplified thinning (basic version)
    prev_img = img.copy()
    
    for iteration in range(max_iterations):
        # Apply erosion with cross-shaped kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(img, kernel, iterations=1)
        
        # If no change, stop
        if np.array_equal(eroded, prev_img):
            break
        
        prev_img = img.copy()
        img = eroded
    
    return img.astype(bool)


def analyze_skeleton_topology(skeleton):
    """
    Analyze topology of skeleton to find branch points and endpoints.
    
    Args:
        skeleton (np.ndarray): Binary skeleton image
    
    Returns:
        tuple: (branch_points, endpoints) as lists of (y, x) coordinates
    """
    # Create a kernel to count neighbors
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    
    # Count neighbors for each skeleton pixel
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    # Find branch points (more than 2 neighbors) and endpoints (1 neighbor)
    branch_points = []
    endpoints = []
    
    skeleton_coords = np.where(skeleton)
    
    for i in range(len(skeleton_coords[0])):
        y, x = skeleton_coords[0][i], skeleton_coords[1][i]
        neighbors = neighbor_count[y, x]
        
        if neighbors > 2:
            branch_points.append((y, x))
        elif neighbors == 1:
            endpoints.append((y, x))
    
    return branch_points, endpoints


def comprehensive_morphological_analysis(image, mask=None):
    """
    Comprehensive morphological analysis combining all methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Combined morphological analysis results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    print("Running comprehensive morphological analysis...")
    
    # 1. Top-hat and bottom-hat analysis
    print("  - Top-hat and bottom-hat transforms...")
    tophat_result = tophat_bottomhat_analysis(image, mask=mask)
    results['tophat_bottomhat'] = tophat_result
    
    # 2. Morphological gradient
    print("  - Morphological gradient analysis...")
    gradient_result = morphological_gradient_analysis(image, mask=mask)
    results['gradient'] = gradient_result
    
    # 3. Opening and closing
    print("  - Opening and closing analysis...")
    opening_closing_result = opening_closing_analysis(image, mask=mask)
    results['opening_closing'] = opening_closing_result
    
    # 4. Hit-or-miss transform
    print("  - Hit-or-miss pattern detection...")
    hmt_result = hit_or_miss_detection(image, mask=mask)
    results['hit_or_miss'] = hmt_result
    
    # 5. Watershed segmentation
    print("  - Watershed segmentation...")
    watershed_result = watershed_segmentation(image, mask=mask)
    results['watershed'] = watershed_result
    
    # 6. Skeletonization
    print("  - Skeletonization analysis...")
    skeleton_result = skeletonization_analysis(image, mask=mask)
    results['skeleton'] = skeleton_result
    
    # Combine all morphological defect detections
    combined_defects = np.zeros_like(mask, dtype=bool)
    
    # Add top-hat/bottom-hat defects
    combined_defects |= tophat_result['combined']['all_defects']
    
    # Add gradient edges
    combined_defects |= gradient_result['combined']['edges']
    
    # Add opening/closing defects (from largest kernel)
    largest_kernel = f"kernel_{max([3, 5, 7])}"
    if largest_kernel in opening_closing_result:
        combined_defects |= opening_closing_result[largest_kernel]['bright_noise']
        combined_defects |= opening_closing_result[largest_kernel]['dark_holes']
    
    # Add hit-or-miss detections
    for pattern_result in hmt_result.values():
        combined_defects |= pattern_result['hit_or_miss']
    
    results['combined_defects'] = combined_defects
    results['defect_count'] = np.sum(combined_defects)
    results['defect_percentage'] = (np.sum(combined_defects) / np.sum(mask) * 100) if np.sum(mask) > 0 else 0
    
    return results


def visualize_morphological_results(image, results, save_path=None):
    """
    Visualize morphological analysis results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from morphological analysis
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Morphological Analysis Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top-hat defects
    if 'tophat_bottomhat' in results:
        combined = results['tophat_bottomhat']['combined']
        axes[0, 1].imshow(combined['bright_defects'], cmap='hot')
        axes[0, 1].set_title(f'Top-hat Defects ({combined["bright_count"]})')
        axes[0, 1].axis('off')
    
    # Bottom-hat defects
    if 'tophat_bottomhat' in results:
        combined = results['tophat_bottomhat']['combined']
        axes[0, 2].imshow(combined['dark_defects'], cmap='hot')
        axes[0, 2].set_title(f'Bottom-hat Defects ({combined["dark_count"]})')
        axes[0, 2].axis('off')
    
    # Morphological gradient
    if 'gradient' in results:
        axes[1, 0].imshow(results['gradient']['combined']['gradient'], cmap='viridis')
        axes[1, 0].set_title('Morphological Gradient')
        axes[1, 0].axis('off')
    
    # Watershed segmentation
    if 'watershed' in results:
        axes[1, 1].imshow(results['watershed']['labels'], cmap='tab20')
        num_segments = results['watershed']['num_segments']
        axes[1, 1].set_title(f'Watershed Segments ({num_segments})')
        axes[1, 1].axis('off')
    
    # Skeleton
    if 'skeleton' in results:
        axes[1, 2].imshow(results['skeleton']['skeleton'], cmap='hot')
        num_branches = results['skeleton']['num_branches']
        num_endpoints = results['skeleton']['num_endpoints']
        axes[1, 2].set_title(f'Skeleton\nBranches: {num_branches}, Ends: {num_endpoints}')
        axes[1, 2].axis('off')
    
    # Hit-or-miss results
    if 'hit_or_miss' in results and results['hit_or_miss']:
        pattern_name = list(results['hit_or_miss'].keys())[0]
        pattern_result = results['hit_or_miss'][pattern_name]
        axes[2, 0].imshow(pattern_result['hit_or_miss'], cmap='hot')
        count = pattern_result['detection_count']
        axes[2, 0].set_title(f'{pattern_name.title()} Pattern ({count})')
        axes[2, 0].axis('off')
    
    # Combined defects
    if 'combined_defects' in results:
        axes[2, 1].imshow(results['combined_defects'], cmap='hot')
        count = results['defect_count']
        percentage = results['defect_percentage']
        axes[2, 1].set_title(f'Combined Defects\n{count} pixels ({percentage:.2f}%)')
        axes[2, 1].axis('off')
    
    # Summary statistics
    axes[2, 2].axis('off')
    if 'tophat_bottomhat' in results and 'gradient' in results:
        summary_text = "Morphological Summary:\n\n"
        summary_text += f"Bright defects: {results['tophat_bottomhat']['combined']['bright_count']}\n"
        summary_text += f"Dark defects: {results['tophat_bottomhat']['combined']['dark_count']}\n"
        summary_text += f"Edge pixels: {results['gradient']['combined']['edge_count']}\n"
        if 'watershed' in results:
            summary_text += f"Segments: {results['watershed']['num_segments']}\n"
        if 'skeleton' in results:
            summary_text += f"Skeleton length: {results['skeleton']['skeleton_pixels']}\n"
        
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        axes[2, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of morphological analysis functions.
    """
    # Create a test image with various morphological features
    test_image = np.random.normal(128, 15, (200, 200)).astype(np.uint8)
    
    # Add bright spots (particles)
    for _ in range(10):
        x, y = np.random.randint(20, 180, 2)
        cv2.circle(test_image, (x, y), np.random.randint(2, 6), 255, -1)
    
    # Add dark holes
    for _ in range(8):
        x, y = np.random.randint(20, 180, 2)
        cv2.circle(test_image, (x, y), np.random.randint(3, 8), 50, -1)
    
    # Add linear defects
    cv2.line(test_image, (50, 50), (150, 150), 200, 2)
    cv2.line(test_image, (100, 20), (100, 180), 80, 1)
    
    # Add rectangular defect
    cv2.rectangle(test_image, (20, 150), (60, 180), 220, -1)
    
    print("Testing Morphological Analysis Module")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_morphological_analysis(test_image)
    
    # Print summary
    print(f"\nMorphological Analysis Summary:")
    if 'tophat_bottomhat' in results:
        combined = results['tophat_bottomhat']['combined']
        print(f"Bright defects (top-hat): {combined['bright_count']}")
        print(f"Dark defects (bottom-hat): {combined['dark_count']}")
    
    if 'gradient' in results:
        print(f"Edge pixels (gradient): {results['gradient']['combined']['edge_count']}")
    
    if 'watershed' in results:
        print(f"Watershed segments: {results['watershed']['num_segments']}")
    
    if 'skeleton' in results:
        skeleton_info = results['skeleton']
        print(f"Skeleton pixels: {skeleton_info['skeleton_pixels']}")
        print(f"Branch points: {skeleton_info['num_branches']}")
        print(f"Endpoints: {skeleton_info['num_endpoints']}")
    
    print(f"Total combined defects: {results['defect_count']} ({results['defect_percentage']:.2f}%)")
    
    # Visualize results
    visualize_morphological_results(test_image, results, 'morphological_analysis_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
