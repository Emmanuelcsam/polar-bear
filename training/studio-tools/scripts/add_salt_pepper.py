#!/usr/bin/env python3
"""
Add Salt and Pepper Noise
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

def add_salt_pepper_noise(image: np.ndarray, salt_vs_pepper: float, amount: float) -> np.ndarray:
    """
    Adds salt and pepper noise to an image.
    
    Args:
        image: Input image.
        salt_vs_pepper: Ratio of salt to pepper.
        amount: Amount of noise to add.
        
    Returns:
        Image with added salt and pepper noise.
    """
    result = image.copy()
    num_salt = np.ceil(amount * result.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * result.size * (1.0 - salt_vs_pepper))
    
    # Add Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    result[coords[0], coords[1]] = 255
    
    # Add Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    result[coords[0], coords[1]] = 0
    
    return result

def main(args):
    """Main function to add salt and pepper noise and save the result."""
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
        result = add_salt_pepper_noise(img, args.salt_vs_pepper, args.amount)
        logger.info(f"Applied salt and pepper noise with salt_vs_pepper={args.salt_vs_pepper} and amount={args.amount}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add salt and pepper noise to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="add_salt_pepper_output.png", help="Path to save the output image.")
    parser.add_argument("--salt_vs_pepper", type=float, default=0.5, help="Ratio of salt to pepper.")
    parser.add_argument("--amount", type=float, default=0.05, help="Amount of noise to add.")
    
    args = parser.parse_args()
    main(args)