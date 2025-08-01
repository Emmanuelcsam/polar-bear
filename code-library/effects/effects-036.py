#!/usr/bin/env python3
"""
Vignette Effect
Category: effects
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def vignette_effect(image: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Applies a vignette effect to an image.
    
    Args:
        image: Input image.
        sigma_x: The standard deviation of the Gaussian kernel in the x-direction.
        sigma_y: The standard deviation of the Gaussian kernel in the y-direction.
        
    Returns:
        Image with vignette effect.
    """
    rows, cols = image.shape[:2]
    # Create vignette mask
    kernel_x = cv2.getGaussianKernel(cols, sigma_x)
    kernel_y = cv2.getGaussianKernel(rows, sigma_y)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    # Apply to each channel
    if len(image.shape) == 3:
        for i in range(3):
            image[:,:,i] = image[:,:,i] * mask
    else:
        image = (image * mask).astype(np.uint8)
    
    return image

def main(args):
    """Main function to apply a vignette effect and save the result."""
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
        result = vignette_effect(img, args.sigma_x, args.sigma_y)
        logger.info(f"Applied vignette effect with sigma_x={args.sigma_x} and sigma_y={args.sigma_y}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a vignette effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="vignette_effect_output.png", help="Path to save the output image.")
    parser.add_argument("--sigma_x", type=float, default=150, help="The standard deviation of the Gaussian kernel in the x-direction.")
    parser.add_argument("--sigma_y", type=float, default=150, help="The standard deviation of the Gaussian kernel in the y-direction.")
    
    args = parser.parse_args()
    main(args)