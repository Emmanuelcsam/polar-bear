#!/usr/bin/env python3
"""
Scharr X - More accurate Sobel
Category: filtering
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def scharr_x(image: np.ndarray) -> np.ndarray:
    """
    Applies the Scharr operator to an image in the x-direction.
    
    Args:
        image: Input image.
        
    Returns:
        Filtered image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    result = np.uint8(np.absolute(scharrx))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply the Scharr operator and save the result."""
    logger.info(f"Starting script: {os.path.basename(__file__)}")
    logger.info(f"Received arguments: {args}")

    try:
        # Load image
        img = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"Failed to load image from: {args.input_path}")
            return

        logger.info(f"Successfully loaded image from: {args.input_path}")

        # Process image
        result = scharr_x(img)
        logger.info("Applied Scharr operator in the x-direction")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the Scharr operator to an image in the x-direction.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="scharr_x_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)