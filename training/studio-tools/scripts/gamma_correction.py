#!/usr/bin/env python3
"""
Gamma Correction
Category: enhancement
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies gamma correction to an image.
    
    Args:
        image: Input image.
        gamma: Gamma value.
        
    Returns:
        Corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def main(args):
    """Main function to apply gamma correction and save the result."""
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
        result = gamma_correction(img, args.gamma)
        logger.info(f"Applied gamma correction with gamma={args.gamma}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply gamma correction to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="gamma_correction_output.png", help="Path to save the output image.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value.")
    
    args = parser.parse_args()
    main(args)