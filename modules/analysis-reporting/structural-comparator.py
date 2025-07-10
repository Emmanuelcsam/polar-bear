import numpy as np
import cv2
from skimage.metrics import structural_similarity
from typing import Dict, Any

def compute_image_structural_comparison(img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
    """Compute structural similarity between images."""
    if img1.shape != img2.shape:
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Ensure images are grayscale
    if len(img1.shape) > 2: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SSIM analysis
    ssim_index, ssim_map = structural_similarity(img1, img2, full=True, win_size=11)
    
    # Multi-scale SSIM
    ms_ssim_values = []
    for scale in [1, 2, 4]:
        if img1.shape[0] < scale * 11 or img1.shape[1] < scale * 11: continue
        img1_s = cv2.resize(img1, (img1.shape[1]//scale, img1.shape[0]//scale))
        img2_s = cv2.resize(img2, (img2.shape[1]//scale, img2.shape[0]//scale))
        ms_ssim_values.append(structural_similarity(img1_s, img2_s, win_size=min(7, min(img1_s.shape))))

    return {
        'ssim': float(ssim_index),
        'ssim_map': ssim_map,
        'ms_ssim': ms_ssim_values,
    }

if __name__ == '__main__':
    # Create two sample images
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), 255, -1)
    
    img2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img2, (22, 22), (78, 78), 255, -1) # Slightly different
    
    print("Running image structural comparison...")
    results = compute_image_structural_comparison(img1, img2)
    
    print("\nComparison Results:")
    print(f"  SSIM Index: {results['ssim']:.4f}")
    print(f"  Multi-scale SSIM: {[f'{v:.4f}' for v in results['ms_ssim']]}")
    
    # Save the SSIM map for visualization
    ssim_map_visual = (results['ssim_map'] * 255).astype(np.uint8)
    cv2.imwrite("ssim_map.png", ssim_map_visual)
    print("\nSaved 'ssim_map.png' for visual inspection.")
