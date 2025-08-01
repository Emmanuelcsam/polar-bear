"""
SSIM (Structural Similarity Index) difference detection module.
"""

import cv2
import numpy as np
import logging
from config.system_config import SystemConfig

# Check if scikit-image is available
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM functionality disabled.")


def compute_ssim_difference(ref_img, live_img):
    """Computes the SSIM difference map and returns a binary mask of defects."""
    # Ensure both images have the same size
    if ref_img.shape != live_img.shape:
        live_img = cv2.resize(live_img, (ref_img.shape[1], 
                                        ref_img.shape[0]))
        logging.debug(f"Resized live image to match reference: "
                    f"{ref_img.shape}")
    
    if SKIMAGE_AVAILABLE:
        (score, diff) = ssim(ref_img, live_img, full=True)
        diff = (diff * 255).astype("uint8")
        
        if score > SystemConfig.SSIM_THRESHOLD:
            # If images are very similar, there are no significant defects.
            # Return an empty mask to save processing time.
            return None, score

        # Threshold the difference image to get a binary mask of the defects
        _, thresh = cv2.threshold(diff, 0, 255, 
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return thresh, score
    else:
        # Fallback to simple difference when scikit-image is not available
        diff = cv2.absdiff(ref_img, live_img)
        score = 1.0 - (np.mean(diff) / 255.0)
        
        if score > SystemConfig.SSIM_THRESHOLD:
            return None, score
            
        # Threshold the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return thresh, score 