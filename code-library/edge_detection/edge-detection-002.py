#!/usr/bin/env python3
"""
Canny Edge Detection
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

def canny_edges(image: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Applies Canny edge detection to an image.
    
    Args:
        image: Input image.
        threshold1: First threshold for the hysteresis procedure.
        threshold2: Second threshold for the hysteresis procedure.
        
    Returns:
        Edge-detected image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    result = cv2.Canny(gray, threshold1, threshold2)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply Canny edge detection and save the result."""
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
        result = canny_edges(img, args.threshold1, args.threshold2)
        logger.info(f"Applied Canny edge detection with threshold1={args.threshold1} and threshold2={args.threshold2}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Canny edge detection to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="canny_edges_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold1", type=int, default=50, help="First threshold for the hysteresis procedure.")
    parser.add_argument("--threshold2", type=int, default=150, help="Second threshold for the hysteresis procedure.")
    
    args = parser.parse_args()
    main(args)