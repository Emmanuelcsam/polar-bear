#!/usr/bin/env python3
"""
Masked Histogram
Category: analysis
"""
import cv2
import numpy as np
import argparse
import os
from logging_utils import get_logger
import aps # Placeholder for aps.py integration
import matplotlib.pyplot as plt

# Logger setup
logger = get_logger(__file__)

def masked_histogram(image: np.ndarray) -> np.ndarray:
    """
    Displays the histogram of a masked region of an image.
    
    Args:
        image: Input image.
        
    Returns:
        The histogram plot as a NumPy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[gray > 127] = 255
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
    
    fig, ax = plt.subplots()
    ax.plot(hist)
    ax.set_title("Masked Histogram")
    ax.set_xlabel("Bins")
    ax.set_ylabel("# of Pixels")
    
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())

def main(args):
    """Main function to display the masked histogram of an image and save the result."""
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
        result = masked_histogram(img)
        logger.info("Generated masked histogram")

        # Save result
        cv2.imwrite(args.output_path, result)
        logger.info(f"Saved processed image to: {args.output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the masked histogram of an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, default="masked_histogram_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()
    main(args)