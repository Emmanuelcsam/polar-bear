#!/usr/bin/env python3
"""
Crop Center Region
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

def crop_center(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Crops the center region of an image.
    
    Args:
        image: Input image.
        crop_width: The width of the cropped region.
        crop_height: The height of the cropped region.
        
    Returns:
        Cropped image.
    """
    height, width = image.shape[:2]
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    return image[start_y:start_y+crop_height, start_x:start_x+crop_width]

def main(args):
    """Main function to crop the center of an image and save the result."""
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
        result = crop_center(img, args.crop_width, args.crop_height)
        logger.info(f"Cropped center of image with width={args.crop_width} and height={args.crop_height}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the center of an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="crop_center_output.png", help="Path to save the output image.")
    parser.add_argument("--crop_width", type=int, default=50, help="The width of the cropped region.")
    parser.add_argument("--crop_height", type=int, default=50, help="The height of the cropped region.")
    
    args = parser.parse_args()
    main(args)