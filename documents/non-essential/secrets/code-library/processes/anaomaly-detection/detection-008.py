#!/usr/bin/env python3
"""
Blob Detection
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

def blob_detection(image: np.ndarray, min_area: int, min_circularity: float) -> np.ndarray:
    """
    Detects blobs in an image.
    
    Args:
        image: Input image.
        min_area: The minimum area of a blob.
        min_circularity: The minimum circularity of a blob.
        
    Returns:
        Image with detected blobs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    result = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

def main(args):
    """Main function to detect blobs and save the result."""
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
        result = blob_detection(img, args.min_area, args.min_circularity)
        logger.info(f"Applied blob detection with min_area={args.min_area} and min_circularity={args.min_circularity}")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect blobs in an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="blob_detection_output.png", help="Path to save the output image.")
    parser.add_argument("--min_area", type=int, default=100, help="The minimum area of a blob.")
    parser.add_argument("--min_circularity", type=float, default=0.1, help="The minimum circularity of a blob.")
    
    args = parser.parse_args()
    main(args)