#!/usr/bin/env python3
"""
Affine Transformation
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

def affine_transform(image: np.ndarray) -> np.ndarray:
    """
    Applies an affine transformation to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Transformed image.
    """
    height, width = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, matrix, (width, height))

def main(args):
    """Main function to apply an affine transformation and save the result."""
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
        result = affine_transform(img)
        logger.info("Applied affine transformation")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply an affine transformation to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="affine_transform_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)