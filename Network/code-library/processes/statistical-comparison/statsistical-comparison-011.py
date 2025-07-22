#!/usr/bin/env python3
"""
Compare Images
Category: analysis
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def compare_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compares two images and returns the difference.
    
    Args:
        image1: The first image.
        image2: The second image.
        
    Returns:
        The difference between the two images.
    """
    if image1.shape != image2.shape:
        logger.error("Images must have the same dimensions.")
        return None
    
    return cv2.absdiff(image1, image2)

def main(args):
    """Main function to compare two images and save the result."""
    logger.info(f"Starting script: {os.path.basename(__file__)}")
    logger.info(f"Received arguments: {args}")

    try:
        # Load images
        img1 = cv2.imread(args.input_path1, cv2.IMREAD_UNCHANGED)
        if img1 is None:
            logger.error(f"Failed to load image from: {args.input_path1}")
            return
        
        img2 = cv2.imread(args.input_path2, cv2.IMREAD_UNCHANGED)
        if img2 is None:
            logger.error(f"Failed to load image from: {args.input_path2}")
            return

        logger.info(f"Successfully loaded images from: {args.input_path1} and {args.input_path2}")

        # Process image
        result = compare_images(img1, img2)
        if result is not None:
            logger.info("Compared images")

            # Save result
            cv2.imwrite(args.output_path, result)
            logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two images.")
    parser.add_argument("input_path1", type=str, help="Path to the first input image.")
    parser.add_argument("input_path2", type=str, help="Path to the second input image.")
    parser.add_argument("--output_path", type=str, default="compare_images_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)