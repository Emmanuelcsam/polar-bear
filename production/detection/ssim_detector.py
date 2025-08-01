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
    if ref_img.shape != live_img.shape:  # Check if dimensions don't match between reference and live images
        live_img = cv2.resize(live_img, (ref_img.shape[1],  # Resize live image width to match reference
                                        ref_img.shape[0]))  # Resize live image height to match reference
        logging.debug(f"Resized live image to match reference: "  # Log the resizing operation for debugging
                    f"{ref_img.shape}")  # Include new dimensions in log message
    
    if SKIMAGE_AVAILABLE:  # Use scikit-image SSIM if library is installed
        (score, diff) = ssim(ref_img, live_img, full=True)  # Calculate SSIM score and full difference map
        diff = (diff * 255).astype("uint8")  # Convert difference map from float range [0,1] to uint8 range [0,255]
        
        if score > SystemConfig.SSIM_THRESHOLD:  # Check if images are too similar to contain significant defects
            # If images are very similar, there are no significant defects.
            # Return an empty mask to save processing time.
            return None, score  # Return None mask and high similarity score to skip further processing

        # Threshold the difference image to get a binary mask of the defects
        _, thresh = cv2.threshold(diff, 0, 255,  # Apply automatic threshold to difference map
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Use Otsu's method with inverted binary output
        return thresh, score  # Return binary defect mask and similarity score
    else:
        # Fallback to simple difference when scikit-image is not available
        diff = cv2.absdiff(ref_img, live_img)  # Calculate absolute pixel-wise difference between images
        score = 1.0 - (np.mean(diff) / 255.0)  # Convert mean difference to similarity score (1.0 = identical, 0.0 = completely different)
        
        if score > SystemConfig.SSIM_THRESHOLD:  # Apply same similarity threshold check as SSIM method
            return None, score  # Return None mask if images are too similar for defect detection
            
        # Threshold the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # Apply fixed threshold of 30 to create binary defect mask
        return thresh, score  # Return binary defect mask and calculated similarity score 