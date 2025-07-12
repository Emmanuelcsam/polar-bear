#!/usr/bin/env python3
"""
Color Thresholding
Category: threshold
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def color_threshold(image: np.ndarray, threshold_value: int) -> np.ndarray:
    """
    Applies a simple binary threshold to a grayscale image.
    
    Args:
        image: Input image.
        threshold_value: Threshold value.
        
    Returns:
        Thresholded image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, result = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return result

def main(args):
    """Main function to apply color thresholding and save the result."""
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
        result = color_threshold(img, args.threshold_value)
        logger.info(f"Applied color thresholding with threshold_value={args.threshold_value}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply color thresholding to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="color_threshold_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold_value", type=int, default=127, help="Threshold value.")
    
    args = parser.parse_args()
    main(args)