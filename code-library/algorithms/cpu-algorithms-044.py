#!/usr/bin/env python3
"""
Advanced Scratch Detection Module for Fiber Optic Defect Analysis
=================================================================

This module implements sophisticated methods for detecting linear defects (scratches)
in fiber optic end-face images. Extracted from defect_analysis.py.

Functions:
- Hessian ridge detection
- Frangi vesselness filter
- Radon transform line detection
- Directional filter banks
- Phase congruency
- Multi-scale line detection

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
try:
    from skimage import filters, transform
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some filters will use manual implementations.")
    SKIMAGE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def hessian_ridge_detection(image, scales=[1, 2, 3, 4], mask=None):
    """
    Multi-scale Hessian ridge detection for linear structures.
    
    Uses eigenvalues of the Hessian matrix to detect ridge-like structures
    that are characteristic of scratches and linear defects.
    
    Args:
        image (np.ndarray): Input grayscale image
        scales (list): List of scales for multi-scale analysis
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Ridge detection results at multiple scales
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    scale_responses = {}
    max_response = np.zeros_like(img_float)
    
    for scale in scales:
        # Gaussian smoothing at current scale
        smoothed = ndimage.gaussian_filter(img_float, scale)
        
        # Compute Hessian matrix components
        Hxx = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=1)
        Hyy = ndimage.sobel(ndimage.sobel(smoothed, axis=0), axis=0)
        Hxy = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=0)
        
        # Compute eigenvalues of Hessian
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy * Hxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        
        lambda1 = 0.5 * (trace + discriminant)  # Larger eigenvalue
        lambda2 = 0.5 * (trace - discriminant)  # Smaller eigenvalue
        
        # Ridge measure (based on Frangi's approach)
        # For ridges, we want lambda2 < 0 and |lambda2| >> |lambda1|
        Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)  # Ratio of eigenvalues
        S = np.sqrt(lambda1**2 + lambda2**2)              # Magnitude measure
        
        # Ridge response function
        beta = 0.5  # Controls sensitivity to blob-like structures
        c = 0.5 * np.max(S)  # Half maximum of S
        
        response = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
        
        # Only keep responses where lambda2 < 0 (ridge-like)
        response[lambda2 > 0] = 0
        
        # Scale normalization
        response *= scale**2
        
        scale_responses[f'scale_{scale}'] = {
            'response': response,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'ridge_direction': np.arctan2(Hxy, Hxx - lambda2),  # Ridge orientation
            'ridge_strength': np.abs(lambda2)
        }
        
        # Update maximum response across scales
        max_response = np.maximum(max_response, response)
    
    # Threshold for ridge detection
    if np.sum(mask) > 0:
        threshold = np.percentile(max_response[mask], 95)
    else:
        threshold = np.percentile(max_response, 95)
    
    ridges = (max_response > threshold) & mask
    
    return {
        'scale_responses': scale_responses,
        'max_response': max_response,
        'ridges': ridges,
        'threshold': threshold,
        'scales': scales
    }


def frangi_vesselness_filter(image, scales=np.arange(1, 4, 0.5), mask=None):
    """
    Frangi vesselness filter for detecting tubular structures (scratches).
    
    Args:
        image (np.ndarray): Input grayscale image
        scales (np.ndarray): Array of scales for analysis
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Vesselness detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    vesselness = np.zeros_like(img_float)
    
    scale_responses = {}
    
    for scale in scales:
        # Gaussian derivatives at current scale
        smoothed = ndimage.gaussian_filter(img_float, scale)
        
        # Second order derivatives (Hessian)
        Hxx = ndimage.gaussian_filter(smoothed, scale, order=[0, 2])
        Hyy = ndimage.gaussian_filter(smoothed, scale, order=[2, 0])
        Hxy = ndimage.gaussian_filter(smoothed, scale, order=[1, 1])
        
        # Eigenvalues of Hessian
        tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
        lambda1 = 0.5 * (Hxx + Hyy + tmp)
        lambda2 = 0.5 * (Hxx + Hyy - tmp)
        
        # Sort eigenvalues by absolute value
        idx = np.abs(lambda1) < np.abs(lambda2)
        lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
        
        # Vesselness measures
        Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2)
        
        # Frangi parameters
        beta = 0.5   # Controls sensitivity to blob-like structures
        gamma = 15   # Controls sensitivity to background noise
        
        # Vesselness response
        v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
        v[lambda2 > 0] = 0  # Only dark ridges (lambda2 < 0)
        
        scale_responses[f'scale_{scale:.1f}'] = {
            'vesselness': v,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'direction': np.arctan2(2*Hxy, Hxx - Hyy)
        }
        
        # Update maximum vesselness
        vesselness = np.maximum(vesselness, v)
    
    # Threshold for vessel detection
    if np.sum(mask) > 0:
        threshold = np.percentile(vesselness[mask], 95)
    else:
        threshold = np.percentile(vesselness, 95)
    
    vessels = (vesselness > threshold) & mask
    
    return {
        'scale_responses': scale_responses,
        'vesselness': vesselness,
        'vessels': vessels,
        'threshold': threshold,
        'scales': scales
    }


def radon_line_detection(image, theta_range=180, mask=None):
    """
    Radon transform for detecting linear structures.
    
    Args:
        image (np.ndarray): Input grayscale image
        theta_range (int): Number of angles to test (0 to 180 degrees)
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Radon transform results and detected lines
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[~mask] = 0
    
    # Edge detection for better line detection
    edges = cv2.Canny(masked_image, 50, 150)
    
    # Compute Radon transform
    theta = np.linspace(0, 180, theta_range, endpoint=False)
    
    if SKIMAGE_AVAILABLE:
        sinogram = transform.radon(edges, theta=theta, circle=True)
    else:
        sinogram = manual_radon_transform(edges, theta)
    
    # Find peaks in Radon space
    line_mask = np.zeros_like(image, dtype=bool)
    detected_lines = []
    
    # Threshold for peak detection
    threshold = np.mean(sinogram) + 2 * np.std(sinogram)
    
    # Find significant peaks
    peak_coords = np.where(sinogram > threshold)
    
    for i in range(len(peak_coords[0])):
        rho_idx = peak_coords[0][i]
        theta_idx = peak_coords[1][i]
        
        # Convert back to image coordinates
        rho = rho_idx - sinogram.shape[0] // 2
        angle = theta[theta_idx] * np.pi / 180
        
        # Draw line on mask
        draw_line_from_radon(line_mask, rho, angle, image.shape)
        
        detected_lines.append({
            'rho': rho,
            'theta': theta[theta_idx],
            'strength': sinogram[rho_idx, theta_idx]
        })
    
    # Apply original mask
    line_mask = line_mask & mask
    
    return {
        'sinogram': sinogram,
        'theta': theta,
        'line_mask': line_mask,
        'detected_lines': detected_lines,
        'num_lines': len(detected_lines),
        'threshold': threshold
    }


def manual_radon_transform(image, theta):
    """
    Manual implementation of Radon transform.
    
    Args:
        image (np.ndarray): Input image
        theta (np.ndarray): Array of angles in degrees
    
    Returns:
        np.ndarray: Radon transform (sinogram)
    """
    # Pad image to handle rotation
    diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    pad_x = (diagonal - image.shape[1]) // 2
    pad_y = (diagonal - image.shape[0]) // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    
    # Initialize sinogram
    sinogram = np.zeros((diagonal, len(theta)))
    
    # Compute projections
    for i, angle in enumerate(theta):
        rotated = ndimage.rotate(padded, angle, reshape=False, order=1)
        sinogram[:, i] = np.sum(rotated, axis=1)
    
    return sinogram


def draw_line_from_radon(mask, rho, theta, shape):
    """
    Draw line from Radon parameters onto mask.
    
    Args:
        mask (np.ndarray): Output mask to draw on
        rho (float): Distance from origin
        theta (float): Angle in radians
        shape (tuple): Image shape (height, width)
    """
    h, w = shape
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Handle edge cases
    epsilon = 1e-10
    
    if abs(sin_t) < epsilon:
        # Nearly horizontal line
        x = int(rho / cos_t) if abs(cos_t) > epsilon else 0
        if 0 <= x < w:
            mask[:, x] = True
    elif abs(cos_t) < epsilon:
        # Nearly vertical line
        y = int(rho / sin_t) if abs(sin_t) > epsilon else 0
        if 0 <= y < h:
            mask[y, :] = True
    else:
        # General case
        for x in range(w):
            y = int((rho - x * cos_t) / sin_t)
            if 0 <= y < h:
                mask[y, x] = True


def directional_filter_bank(image, n_orientations=16, mask=None):
    """
    Directional filter bank using steerable filters for line detection.
    
    Args:
        image (np.ndarray): Input grayscale image
        n_orientations (int): Number of orientations to test
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Directional filtering results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64) / 255.0
    
    responses = {}
    max_response = np.zeros_like(img_float)
    orientation_map = np.zeros_like(img_float)
    
    for i in range(n_orientations):
        angle = i * np.pi / n_orientations
        
        # Create steerable filter
        kernel = create_steerable_filter(angle)
        
        # Apply filter
        response = cv2.filter2D(img_float, -1, kernel)
        response = np.abs(response)
        
        # Non-maximum suppression along perpendicular direction
        suppressed = directional_nms(response, angle + np.pi/2)
        
        responses[f'angle_{angle:.2f}'] = {
            'response': response,
            'suppressed': suppressed,
            'angle': angle
        }
        
        # Update maximum response and orientation
        update_mask = suppressed > max_response
        max_response[update_mask] = suppressed[update_mask]
        orientation_map[update_mask] = angle
    
    # Threshold for line detection
    if np.sum(mask) > 0:
        threshold = np.percentile(max_response[mask], 95)
    else:
        threshold = np.percentile(max_response, 95)
    
    lines = (max_response > threshold) & mask
    
    return {
        'responses': responses,
        'max_response': max_response,
        'orientation_map': orientation_map,
        'lines': lines,
        'threshold': threshold,
        'n_orientations': n_orientations
    }


def create_steerable_filter(angle, size=15, sigma=2):
    """
    Create steerable derivative filter at specified angle.
    
    Args:
        angle (float): Filter orientation in radians
        size (int): Kernel size
        sigma (float): Gaussian standard deviation
    
    Returns:
        np.ndarray: Steerable filter kernel
    """
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(size) - size//2, 
                      np.arange(size) - size//2)
    
    # Rotate coordinates
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)
    
    # Second derivative of Gaussian in x-direction
    g = np.exp(-(x_rot**2 + y_rot**2) / (2*sigma**2))
    kernel = (x_rot**2 / sigma**4 - 1/sigma**2) * g
    
    # Normalize
    kernel = kernel / np.sum(np.abs(kernel))
    
    return kernel.astype(np.float32)


def directional_nms(response, angle):
    """
    Non-maximum suppression along specified direction.
    
    Args:
        response (np.ndarray): Filter response
        angle (float): Direction for suppression (radians)
    
    Returns:
        np.ndarray: Suppressed response
    """
    h, w = response.shape
    suppressed = response.copy()
    
    # Direction vectors
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # Interpolate along direction
            val1 = bilinear_interpolate(response, x + dx, y + dy)
            val2 = bilinear_interpolate(response, x - dx, y - dy)
            
            # Suppress if not maximum
            if response[y, x] < val1 or response[y, x] < val2:
                suppressed[y, x] = 0
    
    return suppressed


def bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation at floating point coordinates.
    
    Args:
        img (np.ndarray): Input image
        x (float): X coordinate
        y (float): Y coordinate
    
    Returns:
        float: Interpolated value
    """
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    
    # Boundary check
    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)
    
    # Interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    return wa * img[y0, x0] + wb * img[y0, x1] + \
           wc * img[y1, x0] + wd * img[y1, x1]


def comprehensive_scratch_detection(image, mask=None):
    """
    Comprehensive scratch detection combining multiple methods.
    
    Args:
        image (np.ndarray): Input grayscale image
        mask (np.ndarray, optional): Binary mask for region of interest
    
    Returns:
        dict: Combined scratch detection results
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    results = {}
    
    print("Running comprehensive scratch detection...")
    
    # 1. Hessian ridge detection
    print("  - Hessian ridge detection...")
    hessian_result = hessian_ridge_detection(image, mask=mask)
    results['hessian_ridges'] = hessian_result
    
    # 2. Frangi vesselness
    print("  - Frangi vesselness filter...")
    frangi_result = frangi_vesselness_filter(image, mask=mask)
    results['frangi_vessels'] = frangi_result
    
    # 3. Radon line detection
    print("  - Radon line detection...")
    radon_result = radon_line_detection(image, mask=mask)
    results['radon_lines'] = radon_result
    
    # 4. Directional filter bank
    print("  - Directional filter bank...")
    directional_result = directional_filter_bank(image, mask=mask)
    results['directional_filters'] = directional_result
    
    # Combine all scratch detections
    combined_scratches = np.zeros_like(mask, dtype=bool)
    combined_scratches |= hessian_result['ridges']
    combined_scratches |= frangi_result['vessels']
    combined_scratches |= radon_result['line_mask']
    combined_scratches |= directional_result['lines']
    
    # Weight the combination based on method confidence
    confidence_map = np.zeros_like(image, dtype=np.float32)
    
    # Add weighted responses
    confidence_map += 0.3 * hessian_result['max_response']
    confidence_map += 0.3 * frangi_result['vesselness']
    confidence_map += 0.2 * directional_result['max_response']
    
    # Normalize confidence
    if confidence_map.max() > 0:
        confidence_map = confidence_map / confidence_map.max()
    
    results['combined_scratches'] = combined_scratches
    results['confidence_map'] = confidence_map
    results['scratch_count'] = np.sum(combined_scratches)
    results['scratch_percentage'] = (np.sum(combined_scratches) / np.sum(mask) * 100) if np.sum(mask) > 0 else 0
    
    return results


def visualize_scratch_results(image, results, save_path=None):
    """
    Visualize scratch detection results.
    
    Args:
        image (np.ndarray): Original image
        results (dict): Results from scratch detection
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Scratch Detection Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Hessian ridges
    if 'hessian_ridges' in results:
        axes[0, 1].imshow(results['hessian_ridges']['max_response'], cmap='viridis')
        axes[0, 1].set_title('Hessian Ridge Response')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['hessian_ridges']['ridges'], cmap='hot')
        count = np.sum(results['hessian_ridges']['ridges'])
        axes[0, 2].set_title(f'Hessian Ridges ({count})')
        axes[0, 2].axis('off')
    
    # Frangi vesselness
    if 'frangi_vessels' in results:
        axes[1, 0].imshow(results['frangi_vessels']['vesselness'], cmap='viridis')
        axes[1, 0].set_title('Frangi Vesselness')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['frangi_vessels']['vessels'], cmap='hot')
        count = np.sum(results['frangi_vessels']['vessels'])
        axes[1, 1].set_title(f'Frangi Vessels ({count})')
        axes[1, 1].axis('off')
    
    # Radon lines
    if 'radon_lines' in results:
        axes[1, 2].imshow(results['radon_lines']['line_mask'], cmap='hot')
        count = results['radon_lines']['num_lines']
        axes[1, 2].set_title(f'Radon Lines ({count})')
        axes[1, 2].axis('off')
    
    # Directional filter response
    if 'directional_filters' in results:
        axes[2, 0].imshow(results['directional_filters']['max_response'], cmap='viridis')
        axes[2, 0].set_title('Directional Filter Response')
        axes[2, 0].axis('off')
    
    # Combined scratches
    if 'combined_scratches' in results:
        axes[2, 1].imshow(results['combined_scratches'], cmap='hot')
        count = results['scratch_count']
        percentage = results['scratch_percentage']
        axes[2, 1].set_title(f'Combined Scratches\n{count} pixels ({percentage:.2f}%)')
        axes[2, 1].axis('off')
    
    # Confidence map
    if 'confidence_map' in results:
        axes[2, 2].imshow(results['confidence_map'], cmap='plasma')
        axes[2, 2].set_title('Scratch Confidence Map')
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of scratch detection functions.
    """
    # Create a test image with linear defects
    test_image = np.random.normal(128, 15, (200, 200)).astype(np.uint8)
    
    # Add scratches at different orientations
    # Horizontal scratch
    test_image[50, 30:170] = 80
    test_image[51, 30:170] = 80
    
    # Vertical scratch
    test_image[30:170, 100] = 200
    test_image[30:170, 101] = 200
    
    # Diagonal scratch
    for i in range(150):
        x, y = 25 + i, 25 + i
        if x < 200 and y < 200:
            test_image[y, x] = 60
            if x+1 < 200:
                test_image[y, x+1] = 60
    
    # Curved scratch (approximated by short line segments)
    for i in range(100):
        x = 50 + i
        y = int(150 + 20 * np.sin(i * 0.1))
        if 0 <= x < 200 and 0 <= y < 200:
            test_image[y, x] = 220
    
    print("Testing Scratch Detection Module")
    print("=" * 50)
    
    # Run comprehensive analysis
    results = comprehensive_scratch_detection(test_image)
    
    # Print summary
    print(f"\nScratch Detection Summary:")
    if 'hessian_ridges' in results:
        print(f"Hessian ridges: {np.sum(results['hessian_ridges']['ridges'])}")
    if 'frangi_vessels' in results:
        print(f"Frangi vessels: {np.sum(results['frangi_vessels']['vessels'])}")
    if 'radon_lines' in results:
        print(f"Radon lines: {results['radon_lines']['num_lines']}")
    if 'directional_filters' in results:
        print(f"Directional lines: {np.sum(results['directional_filters']['lines'])}")
    
    print(f"Total combined scratches: {results['scratch_count']} ({results['scratch_percentage']:.2f}%)")
    
    # Visualize results
    visualize_scratch_results(test_image, results, 'scratch_detection_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
