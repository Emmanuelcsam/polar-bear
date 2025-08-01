#!/usr/bin/env python3
"""
Motion Blur - Simulate motion effect
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

def motion_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a motion blur to an image.
    
    Args:
        image: Input image.
        kernel_size: The size of the motion blur kernel.
        
    Returns:
        Blurred image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernel, 1)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def main(args):
    """Main function to apply a motion blur and save the result."""
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
        result = motion_blur(img, args.kernel_size)
        logger.info(f"Applied motion blur with kernel_size={args.kernel_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a motion blur to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="motion_blur_output.png", help="Path to save the output image.")
    parser.add_argument("--kernel_size", type=int, default=15, help="The size of the motion blur kernel.")
    
    args = parser.parse_args()
    main(args)