#!/usr/bin/env python3
"""
Pixelate Effect
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

def pixelate_effect(image: np.ndarray, pixel_size: int) -> np.ndarray:
    """
    Applies a pixelate effect to an image.
    
    Args:
        image: Input image.
        pixel_size: The size of the pixels.
        
    Returns:
        Image with pixelate effect.
    """
    h, w = image.shape[:2]
    # Resize down
    temp = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    # Resize up
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def main(args):
    """Main function to apply a pixelate effect and save the result."""
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
        result = pixelate_effect(img, args.pixel_size)
        logger.info(f"Applied pixelate effect with pixel_size={args.pixel_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a pixelate effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="pixelate_effect_output.png", help="Path to save the output image.")
    parser.add_argument("--pixel_size", type=int, default=10, help="The size of the pixels.")
    
    args = parser.parse_args()
    main(args)