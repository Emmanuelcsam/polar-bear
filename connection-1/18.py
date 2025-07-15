#!/usr/bin/env python3
"""
Demonstration script showing all modules working together.
"""
import os
import tempfile
import numpy as np
from PIL import Image

# Import all our modules
import data_store
import pixel_generator
import intensity_reader
import image_guided_generator
import pattern_recognizer
import anomaly_detector
import trend_recorder
import learner
import geometry_analyzer

def create_demo_image():
    """Create a simple demo image for testing."""
    # Create a 4x4 gradient image
    img_array = np.array([
        [50, 100, 150, 200],
        [60, 110, 160, 210],
        [70, 120, 170, 220],
        [80, 130, 180, 230]
    ], dtype=np.uint8)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img = Image.fromarray(img_array, mode='L')
    img.save(temp_file.name)
    temp_file.close()

    return temp_file.name

def main():
    """Run demonstration of all modules."""
    print("=== Image Processing Pipeline Demo ===\n")

    # Clear any existing events
    data_store.clear_events()

    # 1. Generate some random pixel data
    print("1. Generating random pixel data...")
    pixel_generator.generate_pixels(interval=0.001, max_iterations=10)

    # 2. Create and read a demo image
    print("\n2. Creating and reading demo image...")
    demo_image = create_demo_image()
    try:
        intensity_reader.read_image(demo_image)
        print(f"   Processed image: {demo_image}")
    finally:
        # Clean up
        if os.path.exists(demo_image):
            os.remove(demo_image)

    # 3. Show current data
    print("\n3. Current data summary:")
    events = data_store.load_events()
    print(f"   Total events: {len(events)}")

    # 4. Analyze patterns
    print("\n4. Pattern analysis:")
    patterns = pattern_recognizer.find_patterns()
    print(f"   Found {len(patterns)} unique values")

    # 5. Detect anomalies
    print("\n5. Anomaly detection:")
    anomalies = anomaly_detector.detect_anomalies(threshold=30)
    print(f"   Found {len(anomalies)} anomalies")

    # 6. Record trends
    print("\n6. Trend analysis:")
    trends = trend_recorder.record_trends()

    # 7. Generate image from intensities
    print("\n7. Generating image from logged intensities:")
    generated_image = image_guided_generator.generate_image(width=4, height=4)
    print(f"   Generated image shape: {generated_image.shape}")

    # 8. Analyze geometry
    print("\n8. Geometry analysis:")
    gx, gy = geometry_analyzer.analyze_geometry(width=4, height=4)
    if gx is not None:
        print(f"   Gradient X shape: {gx.shape}")
        print(f"   Gradient Y shape: {gy.shape}")

    # 9. Learn model
    print("\n9. Learning model:")
    model = learner.learn_model("demo_model.pkl")
    if model:
        print(f"   Model learned with {len(model)} unique values")

    # Clean up model file
    if os.path.exists("demo_model.pkl"):
        os.remove("demo_model.pkl")

    print("\n=== Demo Complete ===")
    print(f"Final event count: {len(data_store.load_events())}")

if __name__ == "__main__":
    main()
