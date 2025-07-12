#!/usr/bin/env python3
"""
To Zero Thresholding
Category: thresholding
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def threshold_tozero(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Applies 'to zero' thresholding to an image.
    
    Args:
        image: Input image.
        threshold: The threshold value.
        
    Returns:
        Thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO)
    return result

def main(args):
    """Main function to apply 'to zero' thresholding and save the result."""
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
        result = threshold_tozero(img, args.threshold)
        logger.info(f"Applied 'to zero' thresholding with threshold={args.threshold}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply 'to zero' thresholding to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="threshold_tozero_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold", type=int, default=127, help="The threshold value.")
    
    args = parser.parse_args()
    main(args)