#!/usr/bin/env python3
"""
Convert RGB to HSV
Category: transformations
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Converts an image from RGB to HSV color space.
    
    Args:
        image: Input image.
        
    Returns:
        Image in HSV color space.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def main(args):
    """Main function to convert an image to HSV and save the result."""
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
        result = rgb_to_hsv(img)
        logger.info("Converted image to HSV color space")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image from RGB to HSV color space.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="rgb_to_hsv_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)