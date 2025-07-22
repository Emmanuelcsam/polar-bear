#!/usr/bin/env python3
"""
Blob Detection Module for Fiber Optic Defect Analysis
=====================================================

This module implements sophisticated blob detection methods for identifying
circular defects, pits, digs, and particles in fiber optic end-face images.

Functions:
- Laplacian of Gaussian (LoG) blob detection
- Determinant of Hessian blob detection
- Difference of Gaussians (DoG) blob detection
- MSER (Maximally Stable Extremal Regions) detection
- Multi-scale blob analysis

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
try:
    from skimage import feature, measure
    from skimage.feature import blob_log, blob_dog, blob_doh
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Using manual implementations.")
    SKIMAGE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def laplacian_of_gaussian_blobs(image, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.1, mask=None):
    """
    Laplacian of Gaussian (LoG) blob detection.
    
    Detects blobs by finding local maxima in scale-normalized LoG responses.
    
    Args:
        image (np.ndarray): Input grayscale image
        min_sigma (float): Minimum standard deviation for Gaussian kernel
        max_sigma (float): Maximum standard deviation for Gaussian kernel
        num_sigma (int): Number of intermediate values of standard deviations
        threshold (float): Detection threshold
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: LoG blob detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    # Generate sigma values
    sigma_list = np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma)
    
    # Store LoG responses at different scales
    scale_responses = {}
    log_stack = np.zeros((img_float.shape[0], img_float.shape[1], len(sigma_list)))
    
    for i, sigma in enumerate(sigma_list):
        # Gaussian smoothing
        smoothed = ndimage.gaussian_filter(img_float, sigma)
        
        # Laplacian (using ndimage for better accuracy)
        laplacian = ndimage.laplace(smoothed)
        
        # Scale normalization
        log_response = sigma**2 * laplacian
        
        log_stack[:, :, i] = log_response
        scale_responses[f'sigma_{sigma:.2f}'] = {
            'response': log_response,
            'sigma': sigma
        }
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image implementation for blob detection
        blobs_log = blob_log(img_float, min_sigma=min_sigma, max_sigma=max_sigma, 
                            num_sigma=num_sigma, threshold=threshold)
        
        # Convert to mask
        blob_mask = np.zeros_like(mask, dtype=bool)
        blob_centers = []
        
        for blob in blobs_log:
            y, x, r = blob
            blob_centers.append((int(y), int(x), r))
            # Draw filled circle
            cv2.circle(blob_mask, (int(x), int(y)), int(r), True, -1)
        
    else:
        # Manual blob detection
        blob_mask, blob_centers = manual_blob_detection(log_stack, sigma_list, threshold)
    
    # Apply original mask
    blob_mask = blob_mask & mask
    
    return {
        'scale_responses': scale_responses,
        'log_stack': log_stack,
        'blob_mask': blob_mask,
        'blob_centers': blob_centers,
        'num_blobs': len(blob_centers),
        'sigma_range': (min_sigma, max_sigma),
        'threshold': threshold
    }


def determinant_of_hessian_blobs(image, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.01, mask=None):
    """
    Determinant of Hessian (DoH) blob detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        min_sigma (float): Minimum sigma for Gaussian kernel
        max_sigma (float): Maximum sigma for Gaussian kernel
        num_sigma (int): Number of sigma values
        threshold (float): Detection threshold
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: DoH blob detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    # Generate sigma values
    sigma_list = np.logspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma)
    
    scale_responses = {}
    doh_stack = np.zeros((img_float.shape[0], img_float.shape[1], len(sigma_list)))
    
    for i, sigma in enumerate(sigma_list):
        # Compute Hessian matrix components
        Ixx = ndimage.gaussian_filter(img_float, sigma, order=[0, 2])
        Iyy = ndimage.gaussian_filter(img_float, sigma, order=[2, 0])
        Ixy = ndimage.gaussian_filter(img_float, sigma, order=[1, 1])
        
        # Determinant of Hessian
        det_hessian = Ixx * Iyy - Ixy**2
        
        # Scale normalization
        det_hessian_normalized = sigma**4 * det_hessian
        
        doh_stack[:, :, i] = det_hessian_normalized
        scale_responses[f'sigma_{sigma:.2f}'] = {
            'det_hessian': det_hessian_normalized,
            'sigma': sigma,
            'Ixx': Ixx,
            'Iyy': Iyy,
            'Ixy': Ixy
        }
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image implementation
        blobs_doh = blob_doh(img_float, min_sigma=min_sigma, max_sigma=max_sigma,
                            num_sigma=num_sigma, threshold=threshold)
        
        blob_mask = np.zeros_like(mask, dtype=bool)
        blob_centers = []
        
        for blob in blobs_doh:
            y, x, r = blob
            blob_centers.append((int(y), int(x), r))
            cv2.circle(blob_mask, (int(x), int(y)), int(r), True, -1)
    else:
        # Manual blob detection
        blob_mask, blob_centers = manual_blob_detection(doh_stack, sigma_list, threshold)
    
    # Apply original mask
    blob_mask = blob_mask & mask
    
    return {
        'scale_responses': scale_responses,
        'doh_stack': doh_stack,
        'blob_mask': blob_mask,
        'blob_centers': blob_centers,
        'num_blobs': len(blob_centers),
        'sigma_range': (min_sigma, max_sigma),
        'threshold': threshold
    }


def difference_of_gaussians_blobs(image, min_sigma=1, max_sigma=10, sigma_ratio=1.6, threshold=0.1, mask=None):
    """
    Difference of Gaussians (DoG) blob detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        min_sigma (float): Minimum sigma
        max_sigma (float): Maximum sigma
        sigma_ratio (float): Ratio between successive sigma values
        threshold (float): Detection threshold
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: DoG blob detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    # Generate sigma values
    sigma_list = []
    sigma = min_sigma
    while sigma <= max_sigma:
        sigma_list.append(sigma)
        sigma *= sigma_ratio
    
    scale_responses = {}
    dog_stack = []
    
    # Compute DoG at each scale
    prev_gaussian = None
    
    for i, sigma in enumerate(sigma_list):
        # Gaussian smoothing
        gaussian = ndimage.gaussian_filter(img_float, sigma)
        
        if prev_gaussian is not None:
            # Difference of Gaussians
            dog = gaussian - prev_gaussian
            dog_stack.append(dog)
            
            scale_responses[f'sigma_{sigma:.2f}'] = {
                'dog': dog,
                'sigma': sigma,
                'prev_sigma': sigma_list[i-1]
            }
        
        prev_gaussian = gaussian
    
    if len(dog_stack) > 0:
        dog_array = np.stack(dog_stack, axis=-1)
    else:
        dog_array = np.zeros((img_float.shape[0], img_float.shape[1], 1))
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image implementation
        blobs_dog = blob_dog(img_float, min_sigma=min_sigma, max_sigma=max_sigma,
                            sigma_ratio=sigma_ratio, threshold=threshold)
        
        blob_mask = np.zeros_like(mask, dtype=bool)
        blob_centers = []
        
        for blob in blobs_dog:
            y, x, r = blob
            blob_centers.append((int(y), int(x), r))
            cv2.circle(blob_mask, (int(x), int(y)), int(r), True, -1)
    else:
        # Manual blob detection
        blob_mask, blob_centers = manual_blob_detection(dog_array, sigma_list[1:], threshold)
    
    # Apply original mask
    blob_mask = blob_mask & mask
    
    return {
        'scale_responses': scale_responses,
        'dog_stack': dog_array,
        'blob_mask': blob_mask,
        'blob_centers': blob_centers,
        'num_blobs': len(blob_centers),
        'sigma_list': sigma_list,
        'threshold': threshold
    }


def manual_blob_detection(response_stack, sigma_list, threshold):
    """
    Manual implementation of blob detection from scale-space responses.
    
    Args:
        response_stack (np.ndarray): 3D array of responses at different scales
        sigma_list (list): List of sigma values
        threshold (float): Detection threshold
    
    Returns:
        tuple: (blob_mask, blob_centers)
    """
    h, w, n_scales = response_stack.shape
    blob_mask = np.zeros((h, w), dtype=bool)
    blob_centers = []
    
    # Find local maxima in 3D space (x, y, scale)
    for s in range(1, n_scales - 1):  # Exclude first and last scales
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                current_val = response_stack[y, x, s]
                
                # Check if it's above threshold
                if current_val < threshold:
                    continue
                
                # Check if it's a local maximum in 3D neighborhood
                is_maximum = True
                for ds in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if ds == 0 and dy == 0 and dx == 0:
                                continue
                            
                            neighbor_val = response_stack[y + dy, x + dx, s + ds]
                            if neighbor_val >= current_val:
                                is_maximum = False
                                break
                        if not is_maximum:
                            break
                    if not is_maximum:
                        break
                
                if is_maximum:
                    # Found a blob
                    sigma = sigma_list[s]
                    radius = sigma * np.sqrt(2)  # Convert sigma to radius
                    blob_centers.append((y, x, radius))
                    
                    # Draw on mask
                    cv2.circle(blob_mask, (x, y), int(radius), True, -1)
    
    return blob_mask, blob_centers


def mser_blob_detection(image, mask=None):
    """
    MSER (Maximally Stable Extremal Regions) detection for blob-like regions.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: MSER detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[~mask] = 0
    
    # Create MSER detector
    mser = cv2.MSER_create(
        _delta=5,           # Delta parameter
        _min_area=30,       # Minimum area
        _max_area=14400,    # Maximum area
        _max_variation=0.25, # Maximum variation
        _min_diversity=0.2,  # Minimum diversity
        _max_evolution=200,  # Maximum evolution
        _area_threshold=1.01, # Area threshold
        _min_margin=0.003,   # Minimum margin
        _edge_blur_size=5    # Edge blur size
    )
    
    # Detect MSER regions
    regions, bboxes = mser.detectRegions(masked_image)
    
    # Create mask and extract features
    mser_mask = np.zeros_like(mask, dtype=bool)
    blob_info = []
    
    for i, region in enumerate(regions):
        # Create region mask
        region_mask = np.zeros_like(image, dtype=np.uint8)
        region_mask[region[:, 1], region[:, 0]] = 255
        
        # Filter out regions outside our mask
        if not np.any(region_mask & mask.astype(np.uint8)):
            continue
        
        # Calculate region properties
        moments = cv2.moments(region_mask)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            area = moments['m00']
            
            # Estimate equivalent radius
            radius = np.sqrt(area / np.pi)
            
            blob_info.append({
                'center': (cy, cx),
                'area': area,
                'radius': radius,
                'bbox': bboxes[i] if i < len(bboxes) else None
            })
            
            # Add to combined mask
            mser_mask |= region_mask.astype(bool)
    
    # Apply original mask
    mser_mask = mser_mask & mask
    
    return {
        'mser_mask': mser_mask,
        'regions': regions,
        'blob_info': blob_info,
        'num_blobs': len(blob_info),
        'bboxes': bboxes
    }


def connected_component_blob_analysis(image, min_area=10, max_area=1000, mask=None):
    """
    Connected component analysis for blob detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        min_area (int): Minimum blob area
        max_area (int): Maximum blob area
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Connected component analysis results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Create binary image using adaptive thresholding
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply mask
    binary = binary & mask.astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter components by area
    blob_mask = np.zeros_like(mask, dtype=bool)
    blob_info = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_area <= area <= max_area:
            # Extract component mask
            component_mask = (labels == i)
            blob_mask |= component_mask
            
            # Calculate properties
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            cx, cy = centroids[i]
            
            # Estimate radius
            radius = np.sqrt(area / np.pi)
            
            blob_info.append({
                'center': (int(cy), int(cx)),
                'area': area,
                'radius': radius,
                'bbox': (x, y, w, h),
                'aspect_ratio': w / h if h > 0 else 0,
                'extent': area / (w * h) if w * h > 0 else 0
            })
    
    return {
        'blob_mask': blob_mask,
        'blob_info': blob_info,
        'num_blobs': len(blob_info),
        'labels': labels,
        'stats': stats,
        'centroids': centroids
    }


def comprehensive_blob_detection(image, mask=None):
    """
    Comprehensive blob detection combining multiple methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Combined blob detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    print("Running comprehensive blob detection...")
    
    # 1. Laplacian of Gaussian
    print("  - Laplacian of Gaussian blob detection...")
    log_result = laplacian_of_gaussian_blobs(image, mask=mask)
    results['log_blobs'] = log_result
    
    # 2. Determinant of Hessian
    print("  - Determinant of Hessian blob detection...")
    doh_result = determinant_of_hessian_blobs(image, mask=mask)
    results['doh_blobs'] = doh_result
    
    # 3. Difference of Gaussians
    print("  - Difference of Gaussians blob detection...")
    dog_result = difference_of_gaussians_blobs(image, mask=mask)
    results['dog_blobs'] = dog_result
    
    # 4. MSER detection
    print("  - MSER blob detection...")
    mser_result = mser_blob_detection(image, mask=mask)
    results['mser_blobs'] = mser_result
    
    # 5. Connected components
    print("  - Connected component analysis...")
    cc_result = connected_component_blob_analysis(image, mask=mask)
    results['cc_blobs'] = cc_result
    
    # Combine all blob detections
    combined_blobs = np.zeros_like(mask, dtype=bool)
    combined_blobs |= log_result['blob_mask']
    combined_blobs |= doh_result['blob_mask']
    combined_blobs |= dog_result['blob_mask']
    combined_blobs |= mser_result['mser_mask']
    combined_blobs |= cc_result['blob_mask']
    
    # Collect all blob centers
    all_blob_centers = []
    all_blob_centers.extend(log_result['blob_centers'])
    all_blob_centers.extend(doh_result['blob_centers'])
    all_blob_centers.extend(dog_result['blob_centers'])
    for blob in mser_result['blob_info']:
        all_blob_centers.append((blob['center'][0], blob['center'][1], blob['radius']))
    for blob in cc_result['blob_info']:
        all_blob_centers.append((blob['center'][0], blob['center'][1], blob['radius']))
    
    results['combined_blobs'] = combined_blobs
    results['all_blob_centers'] = all_blob_centers
    results['blob_count'] = np.sum(combined_blobs)
    results['blob_percentage'] = (np.sum(combined_blobs) / np.sum(mask) * 100) if np.sum(mask) > 0 else 0
    
    return results


def visualize_blob_results(image, results, save_path=None):
    """
    Visualize blob detection results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from blob detection
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Blob Detection Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # LoG blobs
    if 'log_blobs' in results:
        axes[0, 1].imshow(results['log_blobs']['blob_mask'], cmap='hot')
        count = results['log_blobs']['num_blobs']
        axes[0, 1].set_title(f'LoG Blobs ({count})')
        axes[0, 1].axis('off')
    
    # DoH blobs
    if 'doh_blobs' in results:
        axes[0, 2].imshow(results['doh_blobs']['blob_mask'], cmap='hot')
        count = results['doh_blobs']['num_blobs']
        axes[0, 2].set_title(f'DoH Blobs ({count})')
        axes[0, 2].axis('off')
    
    # DoG blobs
    if 'dog_blobs' in results:
        axes[1, 0].imshow(results['dog_blobs']['blob_mask'], cmap='hot')
        count = results['dog_blobs']['num_blobs']
        axes[1, 0].set_title(f'DoG Blobs ({count})')
        axes[1, 0].axis('off')
    
    # MSER blobs
    if 'mser_blobs' in results:
        axes[1, 1].imshow(results['mser_blobs']['mser_mask'], cmap='hot')
        count = results['mser_blobs']['num_blobs']
        axes[1, 1].set_title(f'MSER Blobs ({count})')
        axes[1, 1].axis('off')
    
    # Connected component blobs
    if 'cc_blobs' in results:
        axes[1, 2].imshow(results['cc_blobs']['blob_mask'], cmap='hot')
        count = results['cc_blobs']['num_blobs']
        axes[1, 2].set_title(f'CC Blobs ({count})')
        axes[1, 2].axis('off')
    
    # Combined blobs
    if 'combined_blobs' in results:
        axes[2, 0].imshow(results['combined_blobs'], cmap='hot')
        count = results['blob_count']
        percentage = results['blob_percentage']
        axes[2, 0].set_title(f'Combined Blobs\n{count} pixels ({percentage:.2f}%)')
        axes[2, 0].axis('off')
    
    # Blob centers overlay
    if 'all_blob_centers' in results:
        overlay = display_img.copy()
        for y, x, r in results['all_blob_centers']:
            cv2.circle(overlay, (int(x), int(y)), max(1, int(r)), (255, 0, 0), 1)
        axes[2, 1].imshow(overlay)
        axes[2, 1].set_title('Detected Blob Centers')
        axes[2, 1].axis('off')
    
    # Summary statistics
    axes[2, 2].axis('off')
    if 'log_blobs' in results:
        summary_text = "Blob Detection Summary:\n\n"
        summary_text += f"LoG blobs: {results['log_blobs']['num_blobs']}\n"
        summary_text += f"DoH blobs: {results['doh_blobs']['num_blobs']}\n"
        summary_text += f"DoG blobs: {results['dog_blobs']['num_blobs']}\n"
        summary_text += f"MSER blobs: {results['mser_blobs']['num_blobs']}\n"
        summary_text += f"CC blobs: {results['cc_blobs']['num_blobs']}\n"
        summary_text += f"Total centers: {len(results['all_blob_centers'])}"
        
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        axes[2, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of blob detection functions.
    """
    # Create a test image with various blob-like features
    test_image = np.random.normal(128, 20, (200, 200)).astype(np.uint8)
    
    # Add bright circular blobs (particles)
    for _ in range(15):
        x, y = np.random.randint(20, 180, 2)
        radius = np.random.randint(3, 8)
        cv2.circle(test_image, (x, y), radius, 255, -1)
    
    # Add dark circular blobs (pits)
    for _ in range(10):
        x, y = np.random.randint(20, 180, 2)
        radius = np.random.randint(2, 6)
        cv2.circle(test_image, (x, y), radius, 50, -1)
    
    # Add some elliptical blobs
    for _ in range(5):
        x, y = np.random.randint(30, 170, 2)
        axes_length = (np.random.randint(8, 15), np.random.randint(5, 10))
        angle = np.random.randint(0, 180)
        cv2.ellipse(test_image, (x, y), axes_length, angle, 0, 360, 200, -1)
    
    print("Testing Blob Detection Module")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_blob_detection(test_image)
    
    # Print summary
    print(f"\nBlob Detection Summary:")
    print(f"LoG blobs: {results['log_blobs']['num_blobs']}")
    print(f"DoH blobs: {results['doh_blobs']['num_blobs']}")
    print(f"DoG blobs: {results['dog_blobs']['num_blobs']}")
    print(f"MSER blobs: {results['mser_blobs']['num_blobs']}")
    print(f"Connected component blobs: {results['cc_blobs']['num_blobs']}")
    print(f"Total combined blob pixels: {results['blob_count']} ({results['blob_percentage']:.2f}%)")
    print(f"Total detected blob centers: {len(results['all_blob_centers'])}")
    
    # Visualize results
    visualize_blob_results(test_image, results, 'blob_detection_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
