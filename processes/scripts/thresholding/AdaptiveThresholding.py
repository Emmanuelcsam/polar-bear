#!/usr/bin/env python3
"""
Adaptive Thresholding
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

def adaptive_thresholding(image: np.ndarray, block_size: int, C: int) -> np.ndarray:
    """
    Applies adaptive thresholding to an image.
    
    Args:
        image: Input image.
        block_size: The size of the neighborhood area.
        C: A constant subtracted from the mean or weighted mean.
        
    Returns:
        Thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

def main(args):
    """Main function to apply adaptive thresholding and save the result."""
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
        result = adaptive_thresholding(img, args.block_size, args.C)
        logger.info(f"Applied adaptive thresholding with block_size={args.block_size} and C={args.C}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply adaptive thresholding to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="adaptive_thresholding_output.png", help="Path to save the output image.")
    parser.add_argument("--block_size", type=int, default=11, help="The size of the neighborhood area.")
    parser.add_argument("-C", type=int, default=2, help="A constant subtracted from the mean or weighted mean.")
    
    args = parser.parse_args()
    main(args)