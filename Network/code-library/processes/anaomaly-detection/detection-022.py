# 6_deviation_detector.py
import os
import cv2
import numpy as np
# Import configuration from 0_config.py
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
config = import_module('0_config')

def detect_deviations(threshold=50):
    """Highlights areas in an image that deviate from the dataset's average."""
    print("--- Deviation Detector Started ---")

    # Ensure input directory exists
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        print("No images found to build an average or to compare.")
        return None

    # Step 1: Create an average image from the entire dataset
    print("Calculating average image from dataset...")
    first_img = cv2.imread(os.path.join(config.INPUT_DIR, image_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print(f"Could not read image: {image_files[0]}")
        return None

    avg_img = np.zeros_like(first_img, dtype=np.float32)

    for f in image_files:
        img = cv2.imread(os.path.join(config.INPUT_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            avg_img += img / len(image_files)
    avg_img = avg_img.astype(np.uint8)

    # Step 2: Compare the first image to the average to find anomalies
    target_img = first_img
    print(f"Comparing '{image_files[0]}' against the average.")

    # Calculate the absolute difference and find contours on a thresholded map
    diff_img = cv2.absdiff(target_img, avg_img)
    _, thresh_img = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)

    # Create a color image to draw red rectangles on
    output_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # Draw red box

    output_path = os.path.join(config.OUTPUT_DIR, "anomaly_detection.png")
    cv2.imwrite(output_path, output_img)
    print(f"--- Anomaly map saved to {output_path} ---")
    return output_img

if __name__ == "__main__":
    detect_deviations()
