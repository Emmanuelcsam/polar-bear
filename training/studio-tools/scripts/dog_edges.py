#!/usr/bin/env python3
"""
Difference of Gaussians Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def dog_edges(image: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """
    Applies Difference of Gaussians edge detection to an image.
    
    Args:
        image: Input image.
        sigma1: The standard deviation of the first Gaussian kernel.
        sigma2: The standard deviation of the second Gaussian kernel.
        
    Returns:
        Edge-detected image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Apply two Gaussian blurs with different sigmas
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    # Compute difference
    dog = g1.astype(np.float32) - g2.astype(np.float32)
    # Normalize and threshold
    result = np.uint8(np.clip(np.abs(dog) * 10, 0, 255))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to apply Difference of Gaussians edge detection and save the result."""
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
        result = dog_edges(img, args.sigma1, args.sigma2)
        logger.info(f"Applied Difference of Gaussians edge detection with sigma1={args.sigma1} and sigma2={args.sigma2}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Difference of Gaussians edge detection to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="dog_edges_output.png", help="Path to save the output image.")
    parser.add_argument("--sigma1", type=float, default=1.0, help="The standard deviation of the first Gaussian kernel.")
    parser.add_argument("--sigma2", type=float, default=2.0, help="The standard deviation of the second Gaussian kernel.")
    
    args = parser.parse_args()
    main(args)