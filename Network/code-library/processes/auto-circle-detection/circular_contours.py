#!/usr/bin/env python3
"""
Circular Contours
Category: analysis
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def circular_contours(image: np.ndarray, min_circularity: float) -> np.ndarray:
    """
    Finds circular contours in an image.
    
    Args:
        image: Input image.
        min_circularity: The minimum circularity of a contour.
        
    Returns:
        Image with circular contours drawn.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > min_circularity:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
            
    return result

def main(args):
    """Main function to find circular contours in an image and save the result."""
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
        result = circular_contours(img, args.min_circularity)
        logger.info(f"Found circular contours with min_circularity={args.min_circularity}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find circular contours in an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="circular_contours_output.png", help="Path to save the output image.")
    parser.add_argument("--min_circularity", type=float, default=0.8, help="The minimum circularity of a contour.")
    
    args = parser.parse_args()
    main(args)