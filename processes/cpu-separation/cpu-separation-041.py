#!/usr/bin/env python3
"""
MSER Region Detection
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

def mser_regions(image: np.ndarray) -> np.ndarray:
    """
    Detects MSER regions in an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with detected regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(result_color, hulls, 1, (0, 255, 0), 2)
    return result_color

def main(args):
    """Main function to detect MSER regions and save the result."""
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
        result = mser_regions(img)
        logger.info("Applied MSER region detection")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect MSER regions in an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="mser_regions_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)