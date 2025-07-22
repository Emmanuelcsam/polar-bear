#!/usr/bin/env python3
"""
Guided Filter - Edge-preserving smoothing
Category: filtering
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration
import cv2.ximgproc

# Logger setup
logger = get_logger(__file__)

def guided_filter(image: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Applies a guided filter to an image.
    
    Args:
        image: Input image.
        radius: Radius of the guided filter.
        eps: Regularization parameter.
        
    Returns:
        Filtered image.
    """
    return cv2.ximgproc.guidedFilter(image, image, radius, eps)

def main(args):
    """Main function to apply a guided filter and save the result."""
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
        result = guided_filter(img, args.radius, args.eps)
        logger.info(f"Applied guided filter with radius={args.radius} and eps={args.eps}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a guided filter to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="guided_filter_output.png", help="Path to save the output image.")
    parser.add_argument("--radius", type=int, default=8, help="Radius of the guided filter.")
    parser.add_argument("--eps", type=float, default=0.2, help="Regularization parameter.")
    
    args = parser.parse_args()
    main(args)