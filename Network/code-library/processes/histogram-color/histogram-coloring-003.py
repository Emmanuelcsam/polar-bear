#!/usr/bin/env python3
"""
Apply Custom Colormap
Category: color
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def apply_custom_colormap(image: np.ndarray) -> np.ndarray:
    """
    Applies a custom colormap to an image.
    
    Args:
        image: Input image.
        
    Returns:
        Image with custom colormap applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Create a custom colormap (e.g., blue to yellow)
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        colormap[i, 0, 0] = 255 - i  # Blue
        colormap[i, 0, 1] = i        # Green
        colormap[i, 0, 2] = 0        # Red
        
    return cv2.applyColorMap(gray, colormap)

def main(args):
    """Main function to apply a custom colormap and save the result."""
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
        result = apply_custom_colormap(img)
        logger.info("Applied custom colormap")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a custom colormap to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="apply_custom_colormap_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)