#!/usr/bin/env python3
"""
Vintage Photo Effect
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

def vintage_effect(image: np.ndarray) -> np.ndarray:
    """
    Applies a vintage photo effect to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with vintage photo effect.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Add yellow tint
    image[:,:,0] = np.clip(image[:,:,0] * 0.7, 0, 255)  # Reduce blue
    # Add noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # Add vignette
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
    kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    for i in range(3):
        image[:,:,i] = (image[:,:,i] * mask).astype(np.uint8)
    
    return image

def main(args):
    """Main function to apply a vintage photo effect and save the result."""
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
        result = vintage_effect(img)
        logger.info("Applied vintage photo effect")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a vintage photo effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="vintage_effect_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)