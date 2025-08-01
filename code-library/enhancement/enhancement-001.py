#!/usr/bin/env python3
"""
CLAHE - Contrast Limited Adaptive Histogram Equalization
Category: histogram
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    """
    Applies CLAHE to an image.
    
    Args:
        image: Input image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.
        
    Returns:
        Processed image.
    """
    if len(image.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        result = clahe.apply(image)
    
    return result

def main(args):
    """Main function to apply CLAHE and save the result."""
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
        result = clahe(img, args.clip_limit, args.tile_grid_size)
        logger.info(f"Applied CLAHE with clip_limit={args.clip_limit} and tile_grid_size={args.tile_grid_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply CLAHE to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="clahe_output.png", help="Path to save the output image.")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="Threshold for contrast limiting.")
    parser.add_argument("--tile_grid_size", type=int, default=8, help="Size of grid for histogram equalization.")
    
    args = parser.parse_args()
    main(args)