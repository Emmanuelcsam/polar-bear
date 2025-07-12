#!/usr/bin/env python3
"""
Top Hat - Extract small bright features
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

def tophat(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies morphological tophat to an image.
    
    Args:
        image: Input image.
        kernel_size: Size of the morphological kernel.
        
    Returns:
        Processed image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return result

def main(args):
    """Main function to apply morphological tophat and save the result."""
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
        result = tophat(img, args.kernel_size)
        logger.info(f"Applied morphological tophat with kernel_size={args.kernel_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply morphological tophat to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="tophat_output.png", help="Path to save the output image.")
    parser.add_argument("--kernel_size", type=int, default=9, help="Size of the morphological kernel.")
    
    args = parser.parse_args()
    main(args)