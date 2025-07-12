#!/usr/bin/env python3
"""
Hough Circle Detection
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

def hough_circles(image: np.ndarray, dp: float, min_dist: int, param1: int, param2: int, min_radius: int, max_radius: int) -> np.ndarray:
    """
    Detects circles in an image using the Hough transform.
    
    Args:
        image: Input image.
        dp: Inverse ratio of the accumulator resolution to the image resolution.
        min_dist: Minimum distance between the centers of the detected circles.
        param1: First method-specific parameter.
        param2: Second method-specific parameter.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        
    Returns:
        Image with detected circles.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
    return result_color

def main(args):
    """Main function to detect circles and save the result."""
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
        result = hough_circles(img, args.dp, args.min_dist, args.param1, args.param2, args.min_radius, args.max_radius)
        logger.info(f"Applied Hough circle detection with dp={args.dp}, min_dist={args.min_dist}, param1={args.param1}, param2={args.param2}, min_radius={args.min_radius}, max_radius={args.max_radius}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect circles in an image using the Hough transform.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="hough_circles_output.png", help="Path to save the output image.")
    parser.add_argument("--dp", type=float, default=1, help="Inverse ratio of the accumulator resolution to the image resolution.")
    parser.add_argument("--min_dist", type=int, default=50, help="Minimum distance between the centers of the detected circles.")
    parser.add_argument("--param1", type=int, default=50, help="First method-specific parameter.")
    parser.add_argument("--param2", type=int, default=30, help="Second method-specific parameter.")
    parser.add_argument("--min_radius", type=int, default=0, help="Minimum circle radius.")
    parser.add_argument("--max_radius", type=int, default=0, help="Maximum circle radius.")
    
    args = parser.parse_args()
    main(args)