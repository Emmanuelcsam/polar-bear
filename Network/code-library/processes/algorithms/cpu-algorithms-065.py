#!/usr/bin/env python3
"""
Laplacian Pyramid
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

def laplacian_pyramid(image: np.ndarray) -> np.ndarray:
    """
    Creates a Laplacian pyramid from an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image from the Laplacian pyramid.
    """
    gaussian = cv2.pyrDown(image)
    gaussian_up = cv2.pyrUp(gaussian, dstsize=(image.shape[1], image.shape[0]))
    result = cv2.subtract(image, gaussian_up)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

def main(args):
    """Main function to create a Laplacian pyramid and save the result."""
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
        result = laplacian_pyramid(img)
        logger.info("Created Laplacian pyramid")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Laplacian pyramid from an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="laplacian_pyramid_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)