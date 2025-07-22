#!/usr/bin/env python3
"""
Convolution High Pass Filter
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

def convolution_hpfilter(image: np.ndarray) -> np.ndarray:
    """
    Applies a convolution high pass filter to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Filtered image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    result = cv2.filter2D(gray, -1, kernel)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply a convolution high pass filter and save the result."""
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
        result = convolution_hpfilter(img)
        logger.info("Applied convolution high pass filter")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a convolution high pass filter to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="convolution_hpfilter_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)