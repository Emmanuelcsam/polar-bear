#!/usr/bin/env python3
"""
Read an image in grayscale and log each pixel intensity.
Usage: python intensity_reader.py path/to/image.png
"""
import sys
from PIL import Image
from data_store import save_event

def read_image(path):
    """Read image and log each pixel intensity."""
    try:
        img = Image.open(path).convert("L")
        for i, val in enumerate(img.getdata()):
            print(f"[Intensity] pixel#{i} = {val}")
            save_event({"intensity": val})
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python intensity_reader.py <image_path>")
    else:
        read_image(sys.argv[1])
