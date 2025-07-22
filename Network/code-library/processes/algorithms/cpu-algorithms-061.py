#!/usr/bin/env python3
"""
Gaussian Pyramid (Downscale)
Category: effects
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def gaussian_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    """
    Creates a Gaussian pyramid from an image.
    
    Args:
        image: Input image.
        levels: The number of pyramid levels.
        
    Returns:
        Image from the specified pyramid level.
    """
    result = image.copy()
    for i in range(levels):
        result = cv2.pyrDown(result)
    return result

def main(args):
    """Main function to create a Gaussian pyramid and save the result."""
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
        result = gaussian_pyramid(img, args.levels)
        logger.info(f"Created Gaussian pyramid with {args.levels} levels")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Gaussian pyramid from an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="gaussian_pyramid_output.png", help="Path to save the output image.")
    parser.add_argument("--levels", type=int, default=2, help="The number of pyramid levels.")
    
    args = parser.parse_args()
    main(args)