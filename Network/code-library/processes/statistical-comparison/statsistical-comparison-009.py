#!/usr/bin/env python3
"""
Component Analysis
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

def comp_analysis(image: np.ndarray) -> np.ndarray:
    """
    Performs component analysis on an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with component analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        mask = labels == i
        result[mask] = np.random.randint(0, 255, size=3).tolist()
        
    return result

def main(args):
    """Main function to perform component analysis and save the result."""
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
        result = comp_analysis(img)
        logger.info("Performed component analysis")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform component analysis on an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="comp_analysis_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)