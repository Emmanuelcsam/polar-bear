#!/usr/bin/env python3
"""
ORB Feature Detection
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

def orb_features(image: np.ndarray, n_features: int) -> np.ndarray:
    """
    Detects features in an image using the ORB algorithm.
    
    Args:
        image: Input image.
        n_features: The maximum number of features to retain.
        
    Returns:
        Image with detected features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
        result = orb_features(img, args.n_features)
        logger.info(f"Applied ORB feature detection with n_features={args.n_features}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect features in an image using the ORB algorithm.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="orb_features_output.png", help="Path to save the output image.")
    parser.add_argument("--n_features", type=int, default=500, help="The maximum number of features to retain.")
    
    args = parser.parse_args()
    main(args)