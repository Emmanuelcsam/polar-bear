#!/usr/bin/env python3
"""
Morphological Dilation - Expand bright regions
Category: morphology
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def dilation(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """
    Applies morphological dilation to an image.
    
    Args:
        image: Input image.
        kernel_size: Size of the morphological kernel.
        iterations: Number of times to apply the dilation.
        
    Returns:
        Processed image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.dilate(image, kernel, iterations=iterations)
    return result

def main(args):
    """Main function to apply morphological dilation and save the result."""
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
        result = dilation(img, args.kernel_size, args.iterations)
        logger.info(f"Applied morphological dilation with kernel_size={args.kernel_size} and iterations={args.iterations}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply morphological dilation to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="dilation_output.png", help="Path to save the output image.")
    parser.add_argument("--kernel_size", type=int, default=5, help="Size of the morphological kernel.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to apply the dilation.")
    
    args = parser.parse_args()
    main(args)