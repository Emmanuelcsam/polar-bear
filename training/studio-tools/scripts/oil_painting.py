#!/usr/bin/env python3
"""
Oil Painting Effect
Category: effects
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration
import cv2.xphoto

# Logger setup
logger = get_logger(__file__)

def oil_painting(image: np.ndarray, size: int, dyn_ratio: int) -> np.ndarray:
    """
    Applies an oil painting effect to an image.
    
    Args:
        image: Input image.
        size: Size of the neighborhood.
        dyn_ratio: Ratio of the dynamic range.
        
    Returns:
        Image with oil painting effect.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.xphoto.oilPainting(image, size, dyn_ratio)

def main(args):
    """Main function to apply an oil painting effect and save the result."""
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
        result = oil_painting(img, args.size, args.dyn_ratio)
        logger.info(f"Applied oil painting effect with size={args.size} and dyn_ratio={args.dyn_ratio}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply an oil painting effect to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="oil_painting_output.png", help="Path to save the output image.")
    parser.add_argument("--size", type=int, default=7, help="Size of the neighborhood.")
    parser.add_argument("--dyn_ratio", type=int, default=1, help="Ratio of the dynamic range.")
    
    args = parser.parse_args()
    main(args)