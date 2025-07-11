#!/usr/bin/env python3
"""
Simple OpenCV demo: edge‑detect a given image.
Usage: python opencv_module.py path/to/image.png
"""
import sys
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[OpenCV] OpenCV not available")

def cv_analyze(path):
    """Analyze image using OpenCV edge detection."""
    if not OPENCV_AVAILABLE:
        print("[OpenCV] OpenCV not available, skipping")
        return None

    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[OpenCV] Error: Could not load image {path}")
            return None

        edges = cv2.Canny(img, 100, 200)
        print(f"[OpenCV] edges shape: {edges.shape}")
        return edges
    except Exception as e:
        print(f"[OpenCV] Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opencv_module.py <image_path>")
    else:
        cv_analyze(sys.argv[1])
