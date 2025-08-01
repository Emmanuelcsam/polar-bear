#!/usr/bin/env python3
"""
Apply All Colormaps
Category: color
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def apply_all_colormaps(image: np.ndarray) -> np.ndarray:
    """
    Applies all available colormaps to an image and returns a grid of the results.
    
    Args:
        image: Input image.
        
    Returns:
        A grid of images with all available colormaps applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    colormaps = [
        cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_WINTER,
        cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING,
        cv2.COLORMAP_COOL, cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT
    ]
    
    results = []
    for cmap in colormaps:
        results.append(cv2.applyColorMap(gray, cmap))
        
    return np.vstack([np.hstack(results[:4]), np.hstack(results[4:8]), np.hstack(results[8:])])

def main(args):
    """Main function to apply all colormaps and save the result."""
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
        result = apply_all_colormaps(img)
        logger.info("Applied all colormaps")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply all colormaps to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="apply_all_colormaps_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)