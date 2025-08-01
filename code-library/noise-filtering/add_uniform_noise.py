#!/usr/bin/env python3
"""
Add Uniform Noise
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

def add_uniform_noise(image: np.ndarray, amount: int) -> np.ndarray:
    """
    Adds uniform noise to an image.
    
    Args:
        image: Input image.
        amount: The amount of noise to add.
        
    Returns:
        Image with added uniform noise.
    """
    result = image.copy()
    noise = np.random.uniform(-amount, amount, result.shape)
    result = np.add(result, noise)
    return np.clip(result, 0, 255).astype(np.uint8)

def main(args):
    """Main function to add uniform noise and save the result."""
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
        result = add_uniform_noise(img, args.amount)
        logger.info(f"Applied uniform noise with amount={args.amount}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add uniform noise to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="add_uniform_noise_output.png", help="Path to save the output image.")
    parser.add_argument("--amount", type=int, default=50, help="The amount of noise to add.")
    
    args = parser.parse_args()
    main(args)