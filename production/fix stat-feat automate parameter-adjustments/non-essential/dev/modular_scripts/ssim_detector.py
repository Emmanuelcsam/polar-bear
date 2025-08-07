#!/usr/bin/env python3
"""
SSIM (Structural Similarity Index) difference detection module.
Works independently with fallback to simple difference if scikit-image unavailable.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

# Check if scikit-image is available
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using fallback difference method.")


def compute_ssim_difference(ref_img: np.ndarray, 
                           live_img: np.ndarray,
                           threshold: float = 0.95) -> Tuple[Optional[np.ndarray], float]:
    """
    Computes the SSIM difference map and returns a binary mask of defects.
    
    Args:
        ref_img: Reference grayscale image (uint8)
        live_img: Live/test grayscale image (uint8)
        threshold: SSIM threshold above which images are considered too similar (default: 0.95)
        
    Returns:
        Tuple containing:
            - Binary mask of defects (uint8) or None if images are too similar
            - SSIM similarity score (float, 0-1, 1=identical)
    """
    # Ensure both images have the same size
    if ref_img.shape != live_img.shape:
        live_img = cv2.resize(live_img, (ref_img.shape[1], ref_img.shape[0]))
        logging.debug(f"Resized live image to match reference: {ref_img.shape}")
    
    if SKIMAGE_AVAILABLE:
        # Use scikit-image SSIM implementation
        (score, diff) = ssim(ref_img, live_img, full=True)
        diff = (diff * 255).astype("uint8")
        
        if score > threshold:
            # Images are too similar - no significant defects
            return None, score

        # Threshold the difference image to get a binary mask
        _, thresh = cv2.threshold(diff, 0, 255, 
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return thresh, score
    else:
        # Fallback to simple absolute difference
        diff = cv2.absdiff(ref_img, live_img)
        score = 1.0 - (np.mean(diff) / 255.0)
        
        if score > threshold:
            return None, score
            
        # Fixed threshold for simple difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return thresh, score


def compute_ssim_manual(ref_img: np.ndarray, 
                        live_img: np.ndarray,
                        window_size: int = 11) -> Tuple[float, np.ndarray]:
    """
    Manual SSIM implementation using OpenCV.
    
    Args:
        ref_img: Reference grayscale image (uint8)
        live_img: Live/test grayscale image (uint8) 
        window_size: Size of Gaussian window for local statistics (default: 11)
        
    Returns:
        Tuple containing:
            - SSIM index (float, 0-1)
            - SSIM map showing local similarity (float32, 0-1)
    """
    # SSIM constants to stabilize division
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    # Create Gaussian window
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    # Convert to float for calculations
    img1 = ref_img.astype(float)
    img2 = live_img.astype(float)
    
    # Compute local means
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    # Compute local statistics
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    # SSIM components
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)
    
    # Combine components
    ssim_map = luminance * contrast * structure
    ssim_index = np.mean(ssim_map)
    
    return float(ssim_index), ssim_map.astype(np.float32)


def find_difference_regions(diff_mask: np.ndarray, 
                           min_area: int = 50) -> list:
    """
    Find and analyze regions of difference in a binary mask.
    
    Args:
        diff_mask: Binary difference mask (uint8)
        min_area: Minimum area for a valid region (default: 50)
        
    Returns:
        List of dictionaries containing region information:
            - bbox: (x, y, width, height) bounding box
            - area: Area in pixels
            - centroid: (cx, cy) center point
    """
    regions = []
    
    if diff_mask is None:
        return regions
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        diff_mask, connectivity=8)
    
    # Process each component (skip background at index 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area >= min_area:
            regions.append({
                'bbox': (x, y, w, h),
                'area': int(area),
                'centroid': (int(centroids[i][0]), int(centroids[i][1]))
            })
    
    return regions


def visualize_ssim_comparison(ref_img: np.ndarray,
                             live_img: np.ndarray,
                             diff_mask: Optional[np.ndarray] = None,
                             ssim_score: Optional[float] = None) -> np.ndarray:
    """
    Create visualization of SSIM comparison results.
    
    Args:
        ref_img: Reference grayscale image
        live_img: Live/test grayscale image
        diff_mask: Binary difference mask (optional)
        ssim_score: SSIM similarity score (optional)
        
    Returns:
        Composite visualization image
    """
    # Create 2x2 grid visualization
    h, w = ref_img.shape[:2]
    
    # Create output image (2x2 grid)
    output = np.zeros((h*2, w*2), dtype=np.uint8)
    
    # Top-left: Reference image
    output[0:h, 0:w] = ref_img
    
    # Top-right: Live image
    output[0:h, w:w*2] = live_img
    
    # Bottom-left: Absolute difference
    abs_diff = cv2.absdiff(ref_img, live_img)
    output[h:h*2, 0:w] = abs_diff
    
    # Bottom-right: Binary mask or SSIM map
    if diff_mask is not None:
        output[h:h*2, w:w*2] = diff_mask
    else:
        # Show enhanced difference
        enhanced = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX)
        output[h:h*2, w:w*2] = enhanced
    
    # Convert to BGR for text overlay
    output_bgr = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    cv2.putText(output_bgr, "Reference", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_bgr, "Live", (w+10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_bgr, "Difference", (10, h+30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_bgr, "Mask/Enhanced", (w+10, h+30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add SSIM score if available
    if ssim_score is not None:
        score_text = f"SSIM: {ssim_score:.3f}"
        cv2.putText(output_bgr, score_text, (w-150, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return output_bgr


def main():
    """Standalone test function."""
    print("SSIM Detector Module - Standalone Test")
    print("-" * 40)
    print(f"Scikit-image available: {SKIMAGE_AVAILABLE}")
    
    # Create synthetic test images
    # Reference image - clean circle
    ref_image = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(ref_image, (150, 150), 80, 255, -1)
    
    # Live image - circle with defects
    live_image = ref_image.copy()
    # Add a scratch
    cv2.line(live_image, (100, 100), (200, 200), 0, 3)
    # Add a blob defect
    cv2.circle(live_image, (180, 120), 10, 128, -1)
    # Add noise
    noise = np.random.randint(-10, 10, live_image.shape, dtype=np.int16)
    live_image = np.clip(live_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Compute SSIM difference
    print("\nComputing SSIM difference...")
    diff_mask, score = compute_ssim_difference(ref_image, live_image)
    print(f"SSIM Score: {score:.3f}")
    
    if diff_mask is not None:
        print("Defects detected!")
        
        # Find difference regions
        regions = find_difference_regions(diff_mask)
        print(f"\nFound {len(regions)} difference regions:")
        for i, region in enumerate(regions, 1):
            print(f"  Region {i}:")
            print(f"    Bounding box: {region['bbox']}")
            print(f"    Area: {region['area']} pixels")
            print(f"    Centroid: {region['centroid']}")
    else:
        print("Images are too similar - no significant defects.")
    
    # Test manual SSIM implementation
    print("\nTesting manual SSIM implementation...")
    manual_score, ssim_map = compute_ssim_manual(ref_image, live_image)
    print(f"Manual SSIM Score: {manual_score:.3f}")
    
    # Create visualization
    viz = visualize_ssim_comparison(ref_image, live_image, diff_mask, score)
    output_path = "ssim_comparison_test.png"
    cv2.imwrite(output_path, viz)
    print(f"\nVisualization saved to: {output_path}")
    
    # Save SSIM map
    if ssim_map is not None:
        ssim_map_viz = (ssim_map * 255).astype(np.uint8)
        ssim_map_path = "ssim_map_test.png"
        cv2.imwrite(ssim_map_path, ssim_map_viz)
        print(f"SSIM map saved to: {ssim_map_path}")


if __name__ == "__main__":
    main()
