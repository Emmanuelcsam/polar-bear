#!/usr/bin/env python3
"""
Create a test BMP image with blobs for blob detection testing.
This script loads good.bmp and adds synthetic blobs to it.
"""

import cv2
import numpy as np
from pathlib import Path


def create_blob_test_image(input_path="good.bmp", output_path="blob_test.bmp"):
    """
    Create a test image with synthetic blobs added to the original image.

    Args:
        input_path (str): Path to the input BMP file
        output_path (str): Path for the output BMP file with blobs
    """
    # Load the original image
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found!")
        return False

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image from {input_path}")
        return False

    print(f"Loaded image: {image.shape}")

    # Create a copy for modification
    blob_image = image.copy()

    # Get image dimensions
    height, width = image.shape[:2]

    # Define blob parameters
    blobs = [
        # (center_x, center_y, radius, color(BGR), fill_type)
        (width // 4, height // 4, 30, (0, 0, 0), True),  # Black filled blob (top-left)
        (3 * width // 4, height // 4, 25, (255, 255, 255), True),  # White filled blob (top-right)
        (width // 2, height // 2, 35, (128, 128, 128), True),  # Gray filled blob (center)
        (width // 4, 3 * height // 4, 20, (0, 255, 0), True),  # Green filled blob (bottom-left)
        (3 * width // 4, 3 * height // 4, 28, (255, 0, 0), True),  # Blue filled blob (bottom-right)

        # Additional smaller blobs
        (width // 6, height // 6, 15, (0, 0, 255), True),  # Red small blob
        (5 * width // 6, height // 6, 18, (255, 255, 0), True),  # Cyan small blob
        (width // 6, 5 * height // 6, 12, (128, 0, 128), True),  # Purple small blob
        (5 * width // 6, 5 * height // 6, 22, (0, 128, 255), True),  # Orange small blob
    ]

    # Add blobs to the image
    for center_x, center_y, radius, color, filled in blobs:
        if filled:
            # Filled circle (blob)
            cv2.circle(blob_image, (center_x, center_y), radius, color, -1)
        else:
            # Circle outline
            cv2.circle(blob_image, (center_x, center_y), radius, color, 2)

    # Add some irregular blobs using ellipses
    ellipses = [
        # (center, axes, angle, color, fill)
        ((width // 8, height // 2), (40, 25), 30, (64, 64, 64), True),  # Dark gray ellipse
        ((7 * width // 8, height // 2), (30, 45), -15, (192, 192, 192), True),  # Light gray ellipse
    ]

    for center, axes, angle, color, filled in ellipses:
        thickness = -1 if filled else 2
        cv2.ellipse(blob_image, center, axes, angle, 0, 360, color, thickness)

    # Save the result
    success = cv2.imwrite(output_path, blob_image)
    if success:
        print(f"Successfully created blob test image: {output_path}")
        print(f"Added {len(blobs)} circular blobs and {len(ellipses)} elliptical blobs")
        return True
    else:
        print(f"Error: Failed to save image to {output_path}")
        return False


def validate_blob_image(image_path):
    """Validate that the blob test image was created correctly."""
    if not Path(image_path).exists():
        print(f"Error: Test image {image_path} not found!")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load test image from {image_path}")
        return False

    print(f"Validation successful: {image_path}")
    print(f"Image dimensions: {image.shape}")
    return True


if __name__ == "__main__":
    # Create the blob test image
    if create_blob_test_image():
        # Validate the created image
        validate_blob_image("blob_test.bmp")
        print("\nBlob test image creation completed successfully!")
        print("You can now use 'blob_test.bmp' to test the blob detection emulator.")
    else:
        print("Failed to create blob test image.")
