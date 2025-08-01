#!/usr/bin/env python3
"""
Bilateral Filter
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

def bilateral_filter(image: np.ndarray, d: int, sigma_color: int, sigma_space: int) -> np.ndarray:
    """
    Applies a bilateral filter to an image.
    
    Args:
        image: Input image.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
        
    Returns:
        Filtered image.
    """
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return result

def main(args):
    """Main function to apply a bilateral filter and save the result."""
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
        result = bilateral_filter(img, args.d, args.sigma_color, args.sigma_space)
        logger.info(f"Applied bilateral filter with d={args.d}, sigma_color={args.sigma_color}, sigma_space={args.sigma_space}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a bilateral filter to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="bilateral_filter_output.png", help="Path to save the output image.")
    parser.add_argument("-d", type=int, default=9, help="Diameter of each pixel neighborhood.")
    parser.add_argument("--sigma_color", type=int, default=75, help="Filter sigma in the color space.")
    parser.add_argument("--sigma_space", type=int, default=75, help="Filter sigma in the coordinate space.")
    
    args = parser.parse_args()
    main(args)