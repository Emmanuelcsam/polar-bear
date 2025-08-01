#!/usr/bin/env python3
"""
Hit or Miss - Detect specific patterns
Category: morphology
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def hitmiss(image: np.ndarray) -> np.ndarray:
    """
    Applies the hit-or-miss transform to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Transformed image.
    """
    kernel = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], dtype=np.int8)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    
    return result

def main(args):
    """Main function to apply the hit-or-miss transform and save the result."""
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
        result = hitmiss(img)
        logger.info("Applied hit-or-miss transform")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the hit-or-miss transform to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="hitmiss_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)