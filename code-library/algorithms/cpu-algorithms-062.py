#!/usr/bin/env python3
"""
Hough Line Detection
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

def hough_lines(image: np.ndarray, rho: float, theta: float, threshold: int, min_line_length: int, max_line_gap: int) -> np.ndarray:
    """
    Detects lines in an image using the Hough transform.
    
    Args:
        image: Input image.
        rho: Distance resolution of the accumulator in pixels.
        theta: Angle resolution of the accumulator in radians.
        threshold: Accumulator threshold parameter.
        min_line_length: Minimum line length.
        max_line_gap: Maximum allowed gap between points on the same line to link them.
        
    Returns:
        Image with detected lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result_color

def main(args):
    """Main function to detect lines and save the result."""
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
        result = hough_lines(img, args.rho, np.pi/180, args.threshold, args.min_line_length, args.max_line_gap)
        logger.info(f"Applied Hough line detection with rho={args.rho}, theta={np.pi/180}, threshold={args.threshold}, min_line_length={args.min_line_length}, max_line_gap={args.max_line_gap}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect lines in an image using the Hough transform.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="hough_lines_output.png", help="Path to save the output image.")
    parser.add_argument("--rho", type=float, default=1, help="Distance resolution of the accumulator in pixels.")
    parser.add_argument("--threshold", type=int, default=50, help="Accumulator threshold parameter.")
    parser.add_argument("--min_line_length", type=int, default=50, help="Minimum line length.")
    parser.add_argument("--max_line_gap", type=int, default=10, help="Maximum allowed gap between points on the same line to link them.")
    
    args = parser.parse_args()
    main(args)