#!/usr/bin/env python3
"""
Filter Contours by Area
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

def filter_contours_area(image: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """
    Filters contours in an image by area.
    
    Args:
        image: Input image.
        min_area: The minimum area of a contour.
        max_area: The maximum area of a contour.
        
    Returns:
        Image with filtered contours drawn.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            filtered_contours.append(contour)
            
    result = cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 0), 3)
    return result

def main(args):
    """Main function to filter contours by area and save the result."""
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
        result = filter_contours_area(img, args.min_area, args.max_area)
        logger.info(f"Filtered contours by area with min_area={args.min_area} and max_area={args.max_area}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter contours by area.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="filter_contours_area_output.png", help="Path to save the output image.")
    parser.add_argument("--min_area", type=int, default=100, help="The minimum area of a contour.")
    parser.add_argument("--max_area", type=int, default=1000, help="The maximum area of a contour.")
    
    args = parser.parse_args()
    main(args)