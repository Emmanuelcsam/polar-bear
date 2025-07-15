#!/usr/bin/env python3
"""
Prewitt Y - Alternative edge detection
Category: filtering
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def prewitt_y(image: np.ndarray) -> np.ndarray:
    """
    Applies Prewitt Y operator to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Edge-detected image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    result = cv2.filter2D(gray, -1, kernel)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply Prewitt Y and save the result."""
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
        result = prewitt_y(img)
        logger.info("Applied Prewitt Y operator")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Prewitt Y operator to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="prewitt_y_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)