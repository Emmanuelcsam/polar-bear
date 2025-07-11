#!/usr/bin/env python3
"""
Demo script showing the functionality of my_image_lab
"""
import os
import sys
import time
import numpy as np
import cv2

print("=" * 60)
print("MY IMAGE LAB - Demo")
print("=" * 60)

# 1. Generate some random images
print("\n1. Generating random images...")
from modules.random_pixel import gen
from core.logger import log

for i in range(5):
    img = gen()
    log("demo", f"Generated image {i+1}: shape={img.shape}, mean={img.mean():.1f}")

# 2. Process images to learn distribution
print("\n2. Learning intensity distribution from generated images...")
from modules.cv_module import batch
from modules.intensity_reader import learn

batch("data")  # Process all images in data folder
learn()  # Learn distribution

# 3. Generate guided image based on learned distribution
print("\n3. Generating guided image based on learned distribution...")
from modules.random_pixel import guided
guided_img = guided()
log("demo", f"Guided image: shape={guided_img.shape}, mean={guided_img.mean():.1f}")

# 4. Detect anomalies in an image
print("\n4. Creating and detecting anomalies...")
# Create image with anomaly
test_img = np.ones((32, 32), dtype=np.uint8) * 128
test_img[10:20, 10:20] = 255  # Bright square
cv2.imwrite("data/test_anomaly.png", test_img)

from modules.anomaly_detector import detect
detect("data/test_anomaly.png")

# 5. Cluster images
print("\n5. Clustering images into categories...")
from modules.pattern_recognizer import cluster
cluster(k=3)

# 6. Show statistics
print("\n6. Database statistics...")
from core.datastore import scan

hist_count = len(scan("hist:"))
cat_count = len(scan("cat:"))
rand_count = len(scan("rand:"))
anom_count = len(scan("anom:"))

print(f"   Histograms stored: {hist_count}")
print(f"   Categories assigned: {cat_count}")
print(f"   Random images tracked: {rand_count}")
print(f"   Anomaly detections: {anom_count}")

print("\n✅ Demo completed!")
print("\nRun 'python main.py' for interactive mode.")
print("=" * 60)