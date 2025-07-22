#!/usr/bin/env python3
"""
FAST Feature Detection
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

def fast_features(image: np.ndarray, threshold: int, nonmax_suppression: bool) -> np.ndarray:
    """
    Detects corners in an image using the FAST algorithm.
    
    Args:
        image: Input image.
        threshold: Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel.
        nonmax_suppression: If true, non-maximum suppression is applied to detected corners.
        
    Returns:
        Image with detected features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
    keypoints = fast.detect(gray, None)
    result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))
    return result

def main(args):
    """Main function to detect features and save the result."""
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
        result = fast_features(img, args.threshold, args.nonmax_suppression)
        logger.info(f"Applied FAST feature detection with threshold={args.threshold} and nonmax_suppression={args.nonmax_suppression}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect features in an image using the FAST algorithm.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="fast_features_output.png", help="Path to save the output image.")
    parser.add_argument("--threshold", type=int, default=10, help="Threshold on difference between intensity of the central pixel and pixels on a circle around this pixel.")
    parser.add_argument("--nonmax_suppression", type=bool, default=True, help="If true, non-maximum suppression is applied to detected corners.")
    
    args = parser.parse_args()
    main(args)