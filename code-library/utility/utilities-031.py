#!/usr/bin/env python3
"""
Inpainting (Remove center region)
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

def inpainting(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Applies inpainting to an image.
    
    Args:
        image: Input image.
        radius: The radius of the circular region to inpaint.
        
    Returns:
        Inpainted image.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), radius, 255, -1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def main(args):
    """Main function to apply inpainting and save the result."""
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
        result = inpainting(img, args.radius)
        logger.info(f"Applied inpainting with radius={args.radius}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply inpainting to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="inpainting_output.png", help="Path to save the output image.")
    parser.add_argument("--radius", type=int, default=50, help="The radius of the circular region to inpaint.")
    
    args = parser.parse_args()
    main(args)