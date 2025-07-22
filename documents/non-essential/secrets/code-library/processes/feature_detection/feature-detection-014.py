#!/usr/bin/env python3
"""
Template Matching (using center region)
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

def template_matching(image: np.ndarray, template_path: str, threshold: float) -> np.ndarray:
    """
    Finds a template in an image.
    
    Args:
        image: Input image.
        template_path: Path to the template image.
        threshold: Threshold for matching.
        
    Returns:
        Image with matched template.
    """
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        logger.error(f"Failed to load template from: {template_path}")
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    
    result_color = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result_color, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)
        
    return result_color

def main(args):
    """Main function to find a template in an image and save the result."""
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
        result = template_matching(img, args.template_path, args.threshold)
        logger.info(f"Applied template matching with template_path={args.template_path} and threshold={args.threshold}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find a template in an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("template_path", type=str, help="Path to the template image.")
    parser.add_argument("--output_path", type=str, default="template_matching_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for matching.")
    
    args = parser.parse_args()
    main(args)