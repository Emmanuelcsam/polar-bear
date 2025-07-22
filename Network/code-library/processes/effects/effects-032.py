#!/usr/bin/env python3
"""
Posterize Effect
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

def posterize_effect(image: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Applies a posterize effect to an image.
    
    Args:
        image: Input image.
        n_bits: The number of bits to keep for each color channel.
        
    Returns:
        Image with posterize effect.
    """
    mask = 256 - (1 << n_bits)
    return cv2.bitwise_and(image, mask)

def main(args):
    """Main function to apply a posterize effect and save the result."""
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
        result = posterize_effect(img, args.n_bits)
        logger.info(f"Applied posterize effect with n_bits={args.n_bits}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a posterize effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="posterize_effect_output.png", help="Path to save the output image.")
    parser.add_argument("--n_bits", type=int, default=4, help="The number of bits to keep for each color channel.")
    
    args = parser.parse_args()
    main(args)