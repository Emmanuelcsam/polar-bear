#!/usr/bin/env python3
"""
Shi-Tomasi Corner Detection
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

def shi_tomasi(image: np.ndarray, max_corners: int, quality_level: float, min_distance: int) -> np.ndarray:
    """
    Detects corners in an image using the Shi-Tomasi corner detector.
    
    Args:
        image: Input image.
        max_corners: Maximum number of corners to return.
        quality_level: Parameter characterizing the minimal accepted quality of image corners.
        min_distance: Minimum possible Euclidean distance between the returned corners.
        
    Returns:
        Image with detected corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result_color, (x, y), 3, (0, 255, 0), -1)
    return result_color

def main(args):
    """Main function to detect corners and save the result."""
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
        result = shi_tomasi(img, args.max_corners, args.quality_level, args.min_distance)
        logger.info(f"Applied Shi-Tomasi corner detection with max_corners={args.max_corners}, quality_level={args.quality_level}, min_distance={args.min_distance}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect corners in an image using the Shi-Tomasi corner detector.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="shi_tomasi_output.png", help="Path to save the output image.")
    parser.add_argument("--max_corners", type=int, default=100, help="Maximum number of corners to return.")
    parser.add_argument("--quality_level", type=float, default=0.01, help="Parameter characterizing the minimal accepted quality of image corners.")
    parser.add_argument("--min_distance", type=int, default=10, help="Minimum possible Euclidean distance between the returned corners.")
    
    args = parser.parse_args()
    main(args)