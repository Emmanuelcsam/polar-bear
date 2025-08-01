#!/usr/bin/env python3
"""
GrabCut Foreground Extraction
Category: features
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration

# Logger setup
logger = get_logger(__file__)

def grabcut_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Extracts the foreground of an image using the GrabCut algorithm.
    
    Args:
        image: Input image.
        
    Returns:
        Segmented image.
    """
    h, w = image.shape[:2]
    # Define rectangle around center
    rect = (w//4, h//4, w//2, h//2)
    # Initialize mask
    mask = np.zeros((h, w), np.uint8)
    # Initialize foreground and background models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Apply GrabCut
    if len(image.shape) == 3:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        result = cv2.bitwise_and(image, image, mask=mask2)
    else:
        # GrabCut needs color image
        result_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(result_color, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        result = cv2.bitwise_and(image, image, mask=mask2)
    
    return result

def main(args):
    """Main function to extract the foreground of an image and save the result."""
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
        result = grabcut_segmentation(img)
        logger.info("Applied GrabCut segmentation")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the foreground of an image using the GrabCut algorithm.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="grabcut_segmentation_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)