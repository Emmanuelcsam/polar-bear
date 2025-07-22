#!/usr/bin/env python3
"""
Apply Gaussian blur to smooth the image
Category: blur
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def main(args):
    """Main function to apply Gaussian blur and save the result."""
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
        result = gaussian_blur(img, args.kernel_size, args.sigma)
        logger.info(f"Applied Gaussian blur with kernel_size={args.kernel_size} and sigma={args.sigma}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Gaussian blur to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="gaussian_blur_output.png", help="Path to save the output image.")
    parser.add_argument("--kernel_size", type=int, default=5, help="Size of the Gaussian kernel (must be odd).")
    parser.add_argument("--sigma", type=float, default=1.0, help="Standard deviation for Gaussian kernel.")
    
    args = parser.parse_args()
    main(args)