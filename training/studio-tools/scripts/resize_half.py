#!/usr/bin/env python3
"""
Resize Image to Half Size
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

def resize_half(image: np.ndarray) -> np.ndarray:
    """
    Resizes an image to half its original size.
    
    Args:
        image: Input image.
        
    Returns:
        Resized image.
    """
    height, width = image.shape[:2]
    new_height, new_width = height // 2, width // 2
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def main(args):
    """Main function to resize an image to half size and save the result."""
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
        result = resize_half(img)
        logger.info("Resized image to half size")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image to half its original size.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="resize_half_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)