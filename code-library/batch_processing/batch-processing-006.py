#!/usr/bin/env python3
"""
Apply intensity_reader to every file in a folder.
Usage: python batch_processor.py path/to/folder
"""
import sys
import glob
import os
from intensity_reader import read_image

def process_folder(folder):
    """Process all image files in a folder."""
    if not os.path.exists(folder):
        print(f"[Batch] Error: Folder '{folder}' does not exist")
        return

    # Look for common image extensions
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"]
    image_files = []

    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(folder, pattern)))
        image_files.extend(glob.glob(os.path.join(folder, pattern.upper())))

    if not image_files:
        print(f"[Batch] No image files found in {folder}")
        return

    for path in image_files:
        try:
            print(f"[Batch] Reading {path}")
            read_image(path)
        except Exception as e:
            print(f"[Batch] Error processing {path}: {e}")

    print("[Batch] Done")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_processor.py <folder>")
    else:
        process_folder(sys.argv[1])
