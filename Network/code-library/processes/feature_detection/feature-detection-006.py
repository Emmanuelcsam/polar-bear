#!/usr/bin/env python3
"""
Harris Corner Detection
Category: features
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def harris_corners(image: np.ndarray, block_size: int, ksize: int, k: float) -> np.ndarray:
    """
    Detects corners in an image using the Harris corner detector.
    
    Args:
        image: Input image.
        block_size: Neighborhood size.
        ksize: Aperture parameter for the Sobel operator.
        k: Harris detector free parameter.
        
    Returns:
        Image with detected corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners = cv2.cornerHarris(gray, block_size, ksize, k)
    # Dilate corner points
    corners = cv2.dilate(corners, None)
    # Threshold and mark corners
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    result_color[corners > 0.01 * corners.max()] = [0, 0, 255]
    return result_color

def main(args):
    """Main function to detect corners and save the result."""
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
        result = harris_corners(img, args.block_size, args.ksize, args.k)
        logger.info(f"Applied Harris corner detection with block_size={args.block_size}, ksize={args.ksize}, k={args.k}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect corners in an image using the Harris corner detector.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="harris_corners_output.png", help="Path to save the output image.")
    parser.add_argument("--block_size", type=int, default=2, help="Neighborhood size.")
    parser.add_argument("--ksize", type=int, default=3, help="Aperture parameter for the Sobel operator.")
    parser.add_argument("-k", type=float, default=0.04, help="Harris detector free parameter.")
    
    args = parser.parse_args()
    main(args)