#!/usr/bin/env python3
"""
Morphological Skeleton - Thin objects to lines
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

def skeleton(image: np.ndarray) -> np.ndarray:
    """
    Computes the morphological skeleton of an image.
    
    Args:
        image: Input image.
        
    Returns:
        Skeletonized image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    size = np.size(gray)
    skel = np.zeros(gray.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(gray, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(gray, temp)
        skel = cv2.bitwise_or(skel, temp)
        gray = eroded.copy()
        zeros = size - cv2.countNonZero(gray)
        if zeros == size:
            done = True
    result = skel
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to compute the morphological skeleton and save the result."""
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
        result = skeleton(img)
        logger.info("Computed morphological skeleton")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the morphological skeleton of an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="skeleton_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)