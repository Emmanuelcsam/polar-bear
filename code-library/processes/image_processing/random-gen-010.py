#!/usr/bin/env python3
"""
Create sample test images for the image processing system.
This script generates simple test images to demonstrate the system functionality.
"""

import os
import numpy as np
import sys

def create_test_images():
    """Create sample test images using only numpy (no OpenCV dependency)."""

    # Ensure images_input directory exists
    os.makedirs('images_input', exist_ok=True)

    print("Creating test images...")

    # Try to use OpenCV if available
    try:
        import cv2
        use_cv2 = True
        print("Using OpenCV for image creation")
    except ImportError:
        use_cv2 = False
        print("OpenCV not available, creating simple image data files")

    # Create different types of test images
    images = [
        {
            'name': 'solid_gray.png',
            'width': 100,
            'height': 100,
            'pattern': 'solid',
            'value': 128
        },
        {
            'name': 'bright_image.png',
            'width': 80,
            'height': 80,
            'pattern': 'solid',
            'value': 200
        },
        {
            'name': 'dark_image.png',
            'width': 120,
            'height': 120,
            'pattern': 'solid',
            'value': 50
        },
        {
            'name': 'gradient.png',
            'width': 150,
            'height': 100,
            'pattern': 'gradient',
            'value': None
        },
        {
            'name': 'noise.png',
            'width': 90,
            'height': 90,
            'pattern': 'noise',
            'value': None
        }
    ]

    created_count = 0

    for img_config in images:
        filepath = os.path.join('images_input', img_config['name'])

        # Create image data
        if img_config['pattern'] == 'solid':
            img_data = np.full((img_config['height'], img_config['width']),
                             img_config['value'], dtype=np.uint8)
        elif img_config['pattern'] == 'gradient':
            # Create horizontal gradient
            img_data = np.zeros((img_config['height'], img_config['width']), dtype=np.uint8)
            for x in range(img_config['width']):
                img_data[:, x] = int(255 * x / img_config['width'])
        elif img_config['pattern'] == 'noise':
            # Create random noise
            np.random.seed(42)  # For reproducible results
            img_data = np.random.randint(0, 256,
                                       (img_config['height'], img_config['width']),
                                       dtype=np.uint8)
        else:
            # Default to solid gray
            img_data = np.full((img_config['height'], img_config['width']),
                             128, dtype=np.uint8)

        # Save image
        if use_cv2:
            try:
                cv2.imwrite(filepath, img_data)
                created_count += 1
                print(f"Created: {filepath} ({img_config['width']}x{img_config['height']})")
            except Exception as e:
                print(f"Failed to create {filepath}: {e}")
        else:
            # Save as numpy file if OpenCV not available
            np_filepath = filepath.replace('.png', '.npy')
            np.save(np_filepath, img_data)
            created_count += 1
            print(f"Created: {np_filepath} ({img_config['width']}x{img_config['height']}) [numpy format]")

    print(f"\nSuccessfully created {created_count} test images.")

    if not use_cv2:
        print("\nNote: Created .npy files instead of .png files because OpenCV is not available.")
        print("The system will still work, but some image processing features may be limited.")

    return created_count

if __name__ == "__main__":
    created_count = create_test_images()
    if created_count > 0:
        print(f"\n✅ Test images created successfully!")
        print("You can now run the image processing scripts:")
        print("  python 2_intensity_reader.py")
        print("  python 3_pattern_recognizer.py")
        print("  python run_all_tests.py")
    else:
        print("\n❌ No test images were created.")
        sys.exit(1)
