#!/usr/bin/env python3
"""
Kaleidoscope Effect
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

def kaleidoscope(image: np.ndarray) -> np.ndarray:
    """
    Applies a kaleidoscope effect to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with kaleidoscope effect.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Get quadrant
    quadrant = image[:h//2, :w//2]
    # Create mirrored sections
    image[:h//2, :w//2] = quadrant
    image[:h//2, w//2:] = cv2.flip(quadrant, 1)
    image[h//2:, :w//2] = cv2.flip(quadrant, 0)
    image[h//2:, w//2:] = cv2.flip(quadrant, -1)
    return image

def main(args):
    """Main function to apply a kaleidoscope effect and save the result."""
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
        result = kaleidoscope(img)
        logger.info("Applied kaleidoscope effect")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a kaleidoscope effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="kaleidoscope_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)