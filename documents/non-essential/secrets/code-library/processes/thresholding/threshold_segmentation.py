#!/usr/bin/env python3
"""
Threshold Segmentation
Category: segmentation
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def threshold_segmentation(image: np.ndarray, threshold_value: int) -> np.ndarray:
    """
    Segments an image using a simple threshold.
    
    Args:
        image: Input image.
        threshold_value: The threshold value.
        
    Returns:
        Segmented image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, result = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return result

def main(args):
    """Main function to segment an image and save the result."""
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
        result = threshold_segmentation(img, args.threshold_value)
        logger.info(f"Applied threshold segmentation with threshold_value={args.threshold_value}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an image using a simple threshold.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="threshold_segmentation_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold_value", type=int, default=127, help="The threshold value.")
    
    args = parser.parse_args()
    main(args)
