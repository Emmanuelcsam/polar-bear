#!/usr/bin/env python3
"""
Morphology with Cross Kernel
Category: morphology
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def cross_kernel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Applies a morphological operation with a cross kernel to an image.
    
    Args:
        image: Input image.
        kernel_size: The size of the kernel.
        
    Returns:
        Processed image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def main(args):
    """Main function to apply a morphological operation with a cross kernel and save the result."""
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
        result = cross_kernel(img, args.kernel_size)
        logger.info(f"Applied morphological operation with a cross kernel of size {args.kernel_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a morphological operation with a cross kernel to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="cross_kernel_output.png", help="Path to save the output image.")
    parser.add_argument("--kernel_size", type=int, default=5, help="The size of the kernel.")
    
    args = parser.parse_args()
    main(args)