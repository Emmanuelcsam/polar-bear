#!/usr/bin/env python3
"""
Non-Local Means Denoising
Category: filtering
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def nlm_denoise(image: np.ndarray, h: float, template_window_size: int, search_window_size: int) -> np.ndarray:
    """
    Applies non-local means denoising to an image.
    
    Args:
        image: Input image.
        h: Parameter regulating filter strength.
        template_window_size: Size in pixels of the template patch that is used to compute weights.
        search_window_size: Size in pixels of the window that is used to compute the weighted average for a given pixel.
        
    Returns:
        Denoised image.
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
    else:
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)

def main(args):
    """Main function to apply non-local means denoising and save the result."""
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
        result = nlm_denoise(img, args.H, args.template_window_size, args.search_window_size)
        logger.info(f"Applied non-local means denoising with h={args.H}, template_window_size={args.template_window_size}, search_window_size={args.search_window_size}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply non-local means denoising to an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="nlm_denoise_output.png", help="Path to save the output image.")
    parser.add_argument("-H", type=float, default=10, help="Parameter regulating filter strength.")
    parser.add_argument("--template_window_size", type=int, default=7, help="Size in pixels of the template patch that is used to compute weights.")
    parser.add_argument("--search_window_size", type=int, default=21, help="Size in pixels of the window that is used to compute the weighted average for a given pixel.")
    
    args = parser.parse_args()
    main(args)