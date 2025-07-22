#!/usr/bin/env python3
"""
Apply histogram equalization for contrast enhancement
Category: enhancement
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        image: Input image
    
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # For color images, convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        # For grayscale images
        return cv2.equalizeHist(image)

def main(args):
    """Main function to apply histogram equalization and save the result."""
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
        result = histogram_equalization(img)
        logger.info("Applied histogram equalization")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply histogram equalization to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="histogram_equalization_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)