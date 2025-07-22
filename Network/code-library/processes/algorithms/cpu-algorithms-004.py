#!/usr/bin/env python3
"""
Adaptive Mean Thresholding
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

def adaptive_mean_threshold(image: np.ndarray, block_size: int, C: int) -> np.ndarray:
    """
    Applies adaptive mean thresholding to a grayscale image.
    
    Args:
        image: Input grayscale image.
        block_size: Size of a pixel neighborhood that is used to calculate a threshold value.
        C: Constant subtracted from the mean.
        
    Returns:
        Thresholded image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Block size must be odd
    if block_size % 2 == 0:
        block_size += 1
        
    result = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )
    return result

def main(args):
    """Main function to apply adaptive mean thresholding and save the result."""
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
        result = adaptive_mean_threshold(img, args.block_size, args.C)
        logger.info(f"Applied adaptive mean thresholding with block_size={args.block_size} and C={args.C}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply adaptive mean thresholding to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="adaptive_mean_output.png", help="Path to save the output image.")
    parser.add_argument("--block_size", type=int, default=11, help="Size of a pixel neighborhood.")
    parser.add_argument("-C", type=int, default=2, help="Constant subtracted from the mean.")
    
    args = parser.parse_args()
    main(args)