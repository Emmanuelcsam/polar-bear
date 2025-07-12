#!/usr/bin/env python3
"""
Laplacian Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def laplacian_edges(image: np.ndarray, ksize: int) -> np.ndarray:
    """
    Applies Laplacian edge detection to an image.
    
    Args:
        image: Input image.
        ksize: Aperture size used to compute the second-derivative filters.
        
    Returns:
        Edge-detected image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    result = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    result = cv2.convertScaleAbs(result)
    return result

def main(args):
    """Main function to apply Laplacian edge detection and save the result."""
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
        result = laplacian_edges(img, args.ksize)
        logger.info(f"Applied Laplacian edge detection with ksize={args.ksize}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Laplacian edge detection to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="laplacian_edges_output.png", help="Path to save the output image.")
    parser.add_argument("--ksize", type=int, default=3, help="Aperture size for the Laplacian operator.")
    
    args = parser.parse_args()
    main(args)