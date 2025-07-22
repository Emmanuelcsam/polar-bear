#!/usr/bin/env python3
"""
Solarize Effect
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

def solarize_effect(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Applies a solarize effect to an image.
    
    Args:
        image: Input image.
        threshold: The threshold value.
        
    Returns:
        Image with solarize effect.
    """
    result = image.copy()
    mask = result > threshold
    result[mask] = 255 - result[mask]
    return result

def main(args):
    """Main function to apply a solarize effect and save the result."""
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
        result = solarize_effect(img, args.threshold)
        logger.info(f"Applied solarize effect with threshold={args.threshold}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a solarize effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="solarize_effect_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold", type=int, default=128, help="The threshold value.")
    
    args = parser.parse_args()
    main(args)