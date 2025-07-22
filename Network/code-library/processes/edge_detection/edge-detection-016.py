#!/usr/bin/env python3
"""
Detect edges using Canny edge detection
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

def edge_detection(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Canny edge detection
    return cv2.Canny(blurred, low_threshold, high_threshold)

def main(args):
    """Main function to detect edges and save the result."""
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
        result = edge_detection(img, args.low_threshold, args.high_threshold)
        logger.info(f"Applied edge detection with low_threshold={args.low_threshold} and high_threshold={args.high_threshold}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect edges in an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="edge_detection_output.png", help="Path to save the output image.")
    parser.add_argument("--low_threshold", type=int, default=50, help="Lower threshold for edge detection.")
    parser.add_argument("--high_threshold", type=int, default=150, help="Upper threshold for edge detection.")
    
    args = parser.parse_args()
    main(args)