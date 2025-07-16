#!/usr/bin/env python3
"""
Add Gaussian Noise
Category: noise
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def add_gaussian_noise(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """
    Adds Gaussian Noise to an image.
    
    Args:
        image: Input image (grayscale or color).
        mean: Mean of the Gaussian distribution.
        sigma: Standard deviation of the Gaussian distribution.
        
    Returns:
        Image with added Gaussian noise.
    """
    result = image.copy()
    
    # Add Gaussian noise
    noise = np.random.normal(mean, sigma, result.shape).astype(np.float32)
    result = cv2.add(result.astype(np.float32), noise)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def main(args):
    """Main function to add Gaussian noise and save the result."""
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
        result = add_gaussian_noise(img, args.mean, args.sigma)
        logger.info(f"Applied Gaussian noise with mean={args.mean} and sigma={args.sigma}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Gaussian noise to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="add_gaussian_noise_output.png", help="Path to save the output image.")
    parser.add_argument("--mean", type=float, default=0, help="Mean of the Gaussian distribution.")
    parser.add_argument("--sigma", type=float, default=25, help="Standard deviation of the Gaussian distribution.")
    
    args = parser.parse_args()
    main(args)