#!/usr/bin/env python3
"""
Cartesian to Polar Transform
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

def polar_transform(image: np.ndarray) -> np.ndarray:
    """
    Transforms an image from Cartesian to polar coordinates.
    
    Args:
        image: Input image.
        
    Returns:
        Transformed image.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    maxRadius = min(center[0], center[1])
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    return cv2.warpPolar(image, (width, height), center, maxRadius, flags)

def main(args):
    """Main function to transform an image to polar coordinates and save the result."""
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
        result = polar_transform(img)
        logger.info("Applied polar transformation")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform an image from Cartesian to polar coordinates.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="polar_transform_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)