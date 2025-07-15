#!/usr/bin/env python3
"""
Color Quantization
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

def color_quantization(image: np.ndarray, n_colors: int) -> np.ndarray:
    """
    Reduces the number of colors in an image.
    
    Args:
        image: Input image.
        n_colors: Number of colors to reduce to.
        
    Returns:
        Quantized image.
    """
    if len(image.shape) == 3:
        data = image.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        result = quantized.reshape(image.shape)
    else:
        # For grayscale, use simple quantization
        levels = np.linspace(0, 255, n_colors)
        result = np.digitize(image, levels) * (255 // (n_colors - 1))
        result = result.astype(np.uint8)
    
    return result

def main(args):
    """Main function to apply color quantization and save the result."""
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
        result = color_quantization(img, args.n_colors)
        logger.info(f"Applied color quantization with n_colors={args.n_colors}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply color quantization to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="color_quantization_output.png", help="Path to save the output image.")
    parser.add_argument("--n_colors", type=int, default=8, help="Number of colors to reduce to.")
    
    args = parser.parse_args()
    main(args)