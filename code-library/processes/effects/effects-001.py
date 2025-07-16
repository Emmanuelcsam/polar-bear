#!/usr/bin/env python3
"""
Cartoon Effect
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

def cartoon_effect(image: np.ndarray) -> np.ndarray:
    """
    Applies a cartoon effect to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with cartoon effect.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Apply bilateral filter
    smooth = cv2.bilateralFilter(image, 15, 80, 80)
    # Get edges
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    # Convert edges to color
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Combine
    return cv2.bitwise_and(smooth, edges)

def main(args):
    """Main function to apply a cartoon effect and save the result."""
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
        result = cartoon_effect(img)
        logger.info("Applied cartoon effect")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a cartoon effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="cartoon_effect_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)