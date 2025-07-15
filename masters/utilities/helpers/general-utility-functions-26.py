# src/utils.py
import cv2
from pathlib import Path

def load_image(image_path):
    """Loads an image from a file path."""
    return cv2.imread(str(image_path))

def list_images(folder_path):
    """Lists all images in a folder."""
    path = Path(folder_path)
    return list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.bmp"))