#!/usr/bin/env python3
"""
Flip Image Vertically
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

def flip_vertical(image: np.ndarray) -> np.ndarray:
    """
    Flips an image vertically.
    
    Args:
        image: Input image.
        
    Returns:
        Vertically flipped image.
    """
    return cv2.flip(image, 0)

def main(args):
    """Main function to flip an image vertically and save the result."""
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
        result = flip_vertical(img)
        logger.info("Flipped image vertically")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip an image vertically.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="flip_vertical_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)