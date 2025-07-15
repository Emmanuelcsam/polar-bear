#!/usr/bin/env python3
"""
Sepia Tone Effect
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

def sepia_effect(image: np.ndarray) -> np.ndarray:
    """
    Applies a sepia tone effect to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with sepia tone effect.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    result = cv2.transform(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)

def main(args):
    """Main function to apply a sepia tone effect and save the result."""
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
        result = sepia_effect(img)
        logger.info("Applied sepia tone effect")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a sepia tone effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="sepia_effect_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)