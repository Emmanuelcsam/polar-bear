#!/usr/bin/env python3
"""
Unsharp Masking - Sharpen image
Category: filtering
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def unsharp_mask(image: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    Applies an unsharp mask to an image.
    
    Args:
        image: Input image.
        sigma: The standard deviation of the Gaussian kernel.
        strength: The strength of the sharpening.
        
    Returns:
        Sharpened image.
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

def main(args):
    """Main function to apply an unsharp mask and save the result."""
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
        result = unsharp_mask(img, args.sigma, args.strength)
        logger.info(f"Applied unsharp mask with sigma={args.sigma} and strength={args.strength}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply an unsharp mask to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="unsharp_mask_output.png", help="Path to save the output image.")
    parser.add_argument("--sigma", type=float, default=2.0, help="The standard deviation of the Gaussian kernel.")
    parser.add_argument("--strength", type=float, default=0.5, help="The strength of the sharpening.")
    
    args = parser.parse_args()
    main(args)