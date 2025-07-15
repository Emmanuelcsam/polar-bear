#!/usr/bin/env python3
"""
Sauvola's Local Thresholding
Category: thresholding
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def sauvola_threshold(image: np.ndarray, window_size: int, k: float, R: float) -> np.ndarray:
    """
    Applies Sauvola's local thresholding to an image.
    
    Args:
        image: Input image.
        window_size: The size of the local window.
        k: The Sauvola parameter.
        R: The dynamic range of standard deviation.
        
    Returns:
        Thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Sauvola's method: T = mean * (1 + k * ((std / R) - 1))
    mean = cv2.boxFilter(gray.astype(np.float32), -1, (window_size, window_size))
    sqmean = cv2.boxFilter((gray.astype(np.float32))**2, -1, (window_size, window_size))
    std = np.sqrt(sqmean - mean**2)
    threshold = mean * (1 + k * ((std / R) - 1))
    result = (gray > threshold).astype(np.uint8) * 255
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply Sauvola's local thresholding and save the result."""
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
        result = sauvola_threshold(img, args.window_size, args.k, args.R)
        logger.info(f"Applied Sauvola's local thresholding with window_size={args.window_size}, k={args.k}, and R={args.R}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Sauvola's local thresholding to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="sauvola_threshold_output.png", help="Path to save the output image.")
    parser.add_argument("--window_size", type=int, default=25, help="The size of the local window.")
    parser.add_argument("-k", type=float, default=0.2, help="The Sauvola parameter.")
    parser.add_argument("-R", type=float, default=128, help="The dynamic range of standard deviation.")
    
    args = parser.parse_args()
    main(args)