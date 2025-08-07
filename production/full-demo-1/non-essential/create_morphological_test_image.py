#!/usr/bin/env python3
"""
Create a test image for morphological features analysis based on good.bmp.
This image will include various shapes, textures, and features to test all aspects
of morphological analysis including defects, components, and complexity.
"""

import cv2
import numpy as np
from pathlib import Path


def create_morphological_test_image(base_image_path="pictures/good.bmp",
                                   output_path="pictures/morphological_test.bmp"):
    """
    Create a comprehensive test image for morphological features analysis.

    Args:
        base_image_path: Path to the base image (good.bmp)
        output_path: Path to save the test image
    """

    # Load base image if it exists
    base_path = Path(base_image_path)
    if base_path.exists():
        print(f"Loading base image: {base_image_path}")
        base_image = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
        if base_image is None:
            print(f"Could not load base image, creating new one")
            base_image = np.ones((600, 800), dtype=np.uint8) * 128
        else:
            # Resize if needed
            if base_image.shape[0] != 600 or base_image.shape[1] != 800:
                base_image = cv2.resize(base_image, (800, 600))
            print(f"Base image loaded: {base_image.shape}")
    else:
        print(f"Base image not found, creating new test image")
        base_image = np.ones((600, 800), dtype=np.uint8) * 128

    # Create test image starting with base
    test_image = base_image.copy()

    print("Adding morphological test features...")

    # 1. Various geometric shapes for component analysis
    print("  - Adding geometric shapes")
    # Large circle
    cv2.circle(test_image, (150, 150), 60, 255, -1)

    # Rectangle
    cv2.rectangle(test_image, (250, 100), (350, 200), 255, -1)

    # Ellipse
    cv2.ellipse(test_image, (450, 150), (50, 30), 45, 0, 360, 255, -1)

    # Triangle (using filled polygon)
    triangle_points = np.array([[550, 100], [600, 200], [500, 200]], np.int32)
    cv2.fillPoly(test_image, [triangle_points], 255)

    # Small circles for component counting
    small_circle_centers = [(100, 300), (150, 320), (200, 310), (250, 300), (300, 315)]
    for center in small_circle_centers:
        cv2.circle(test_image, center, 15, 255, -1)

    # 2. Complex shapes for skeleton analysis
    print("  - Adding complex shapes for skeleton analysis")
    # Star shape
    star_points = []
    center_x, center_y = 150, 450
    outer_radius, inner_radius = 40, 20
    for i in range(10):
        angle = i * np.pi / 5
        if i % 2 == 0:
            x = center_x + outer_radius * np.cos(angle)
            y = center_y + outer_radius * np.sin(angle)
        else:
            x = center_x + inner_radius * np.cos(angle)
            y = center_y + inner_radius * np.sin(angle)
        star_points.append([int(x), int(y)])
    cv2.fillPoly(test_image, [np.array(star_points, np.int32)], 255)

    # Branching structure
    cv2.line(test_image, (350, 400), (350, 500), 255, 8)
    cv2.line(test_image, (350, 450), (400, 400), 255, 6)
    cv2.line(test_image, (350, 450), (300, 400), 255, 6)
    cv2.line(test_image, (350, 480), (380, 520), 255, 4)
    cv2.line(test_image, (350, 480), (320, 520), 255, 4)

    # 3. Textural features for morphological analysis
    print("  - Adding textural features")
    # Random dots pattern
    np.random.seed(42)  # For reproducible results
    for _ in range(50):
        x = np.random.randint(450, 550)
        y = np.random.randint(300, 400)
        radius = np.random.randint(2, 8)
        intensity = np.random.randint(200, 255)
        cv2.circle(test_image, (x, y), radius, intensity, -1)

    # Grid pattern
    for i in range(580, 750, 20):
        for j in range(300, 430, 20):
            cv2.rectangle(test_image, (i, j), (i+10, j+10), 255, -1)

    # 4. Defects and artifacts
    print("  - Adding defects and artifacts")
    # Bright defects (spots)
    bright_defects = [(100, 500), (180, 480), (220, 520), (160, 540)]
    for center in bright_defects:
        cv2.circle(test_image, center, 8, 255, -1)
        # Add halo effect
        cv2.circle(test_image, center, 12, 200, 2)

    # Dark defects (holes/scratches)
    dark_defects = [(250, 450), (280, 470), (310, 460)]
    for center in dark_defects:
        cv2.circle(test_image, center, 6, 50, -1)

    # Linear defects (scratches)
    cv2.line(test_image, (400, 450), (500, 480), 80, 3)
    cv2.line(test_image, (420, 500), (480, 520), 60, 2)

    # 5. Holes and cavities for morphological opening/closing tests
    print("  - Adding holes and cavities")
    # Shape with hole
    cv2.circle(test_image, (600, 350), 40, 255, -1)
    cv2.circle(test_image, (600, 350), 20, 128, -1)  # Hole in the middle

    # Rectangle with multiple holes
    cv2.rectangle(test_image, (650, 400), (750, 500), 255, -1)
    hole_centers = [(670, 420), (690, 440), (710, 460), (730, 480)]
    for center in hole_centers:
        cv2.circle(test_image, center, 8, 128, -1)

    # 6. Rough edges for roughness analysis
    print("  - Adding rough edges")
    # Create jagged shape
    base_points = [(550, 450), (600, 430), (650, 450), (630, 500), (580, 520), (550, 480)]
    # Add noise to points
    noisy_points = []
    for point in base_points:
        noise_x = np.random.randint(-5, 5)
        noise_y = np.random.randint(-5, 5)
        noisy_points.append([point[0] + noise_x, point[1] + noise_y])
    cv2.fillPoly(test_image, [np.array(noisy_points, np.int32)], 255)

    # 7. Add some noise to test robustness
    print("  - Adding noise")
    noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise * 0.3, 0, 255).astype(np.uint8)

    # 8. Different intensity levels for multi-scale analysis
    print("  - Adding different intensity regions")
    # Dark region
    cv2.rectangle(test_image, (50, 50), (120, 120), 80, -1)
    # Medium region
    cv2.rectangle(test_image, (650, 50), (720, 120), 180, -1)
    # Add some small features in these regions
    cv2.circle(test_image, (85, 85), 8, 150, -1)
    cv2.circle(test_image, (685, 85), 8, 120, -1)

    # Save the test image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), test_image)
    if success:
        print(f"Test image saved to: {output_path}")
        print(f"Image size: {test_image.shape}")
        print(f"Intensity range: {test_image.min()} - {test_image.max()}")

        # Create a color version for better visualization
        color_output = str(output_path).replace('.bmp', '_color.bmp')
        test_image_color = cv2.applyColorMap(test_image, cv2.COLORMAP_JET)
        cv2.imwrite(color_output, test_image_color)
        print(f"Color version saved to: {color_output}")

        # Print some statistics
        unique_values = len(np.unique(test_image))
        print(f"Unique intensity values: {unique_values}")

    else:
        print(f"Failed to save test image to: {output_path}")

    return test_image


def create_simple_morphological_test():
    """Create a simple test image with basic shapes."""
    print("\nCreating simple morphological test image...")

    # Create simple test image
    simple_image = np.zeros((400, 600), dtype=np.uint8)

    # Add basic shapes
    cv2.circle(simple_image, (100, 100), 50, 255, -1)  # Circle
    cv2.rectangle(simple_image, (200, 50), (300, 150), 255, -1)  # Rectangle
    cv2.ellipse(simple_image, (400, 100), (60, 30), 0, 0, 360, 255, -1)  # Ellipse

    # Add some noise
    noise = np.random.randint(0, 30, simple_image.shape, dtype=np.uint8)
    simple_image = cv2.add(simple_image, noise)

    # Add defects
    cv2.circle(simple_image, (150, 200), 8, 255, -1)  # Bright defect
    cv2.circle(simple_image, (250, 200), 8, 0, -1)    # Dark defect

    # Save simple test image
    simple_path = "pictures/morphological_simple_test.bmp"
    cv2.imwrite(simple_path, simple_image)
    print(f"Simple test image saved to: {simple_path}")

    return simple_image


def main():
    """Create both comprehensive and simple test images."""
    print("Creating Morphological Features Test Images")
    print("=" * 50)

    # Create comprehensive test image
    test_image = create_morphological_test_image()

    # Create simple test image
    simple_image = create_simple_morphological_test()

    print("\nTest images created successfully!")
    print("You can now run the morphological features emulator with these test images.")

    # Also test the morphological features module directly
    print("\nTesting morphological features extraction...")

    # Import and test the morphological features
    try:
        from dev.morphological_features import extract_morphological_features

        features = extract_morphological_features(test_image)
        print(f"Extracted {len(features)} morphological features")

        # Print first few features
        feature_names = list(features.keys())[:5]
        for name in feature_names:
            print(f"  {name}: {features[name]:.3f}")

    except Exception as e:
        print(f"Error testing morphological features: {e}")


if __name__ == "__main__":
    main()
