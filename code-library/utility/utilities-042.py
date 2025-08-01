#!/usr/bin/env python3
"""
Pad Image with Border
Category: transformations
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def pad_image(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """
    Pads an image with a border.
    
    Args:
        image: Input image.
        top: The top border size.
        bottom: The bottom border size.
        left: The left border size.
        right: The right border size.
        
    Returns:
        Padded image.
    """
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def main(args):
    """Main function to pad an image and save the result."""
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
        result = pad_image(img, args.top, args.bottom, args.left, args.right)
        logger.info(f"Padded image with top={args.top}, bottom={args.bottom}, left={args.left}, right={args.right}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pad an image with a border.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="pad_image_output.png", help="Path to save the output image.")
    parser.add_argument("--top", type=int, default=20, help="The top border size.")
    parser.add_argument("--bottom", type=int, default=20, help="The bottom border size.")
    parser.add_argument("--left", type=int, default=20, help="The left border size.")
    parser.add_argument("--right", type=int, default=20, help="The right border size.")
    
    args = parser.parse_args()
    main(args)