"""
Refactored Pixel Sampler Module
Separates functions from main execution for better testability
"""
import os
import numpy as np
from PIL import Image
import pickle
import random
from typing import Dict, List, Tuple, Optional


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on extension"""
    return filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image and convert to RGB numpy array"""
    try:
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def sample_pixels_from_image(image: np.ndarray, sample_size: int) -> List[np.ndarray]:
    """Sample random pixels from an image"""
    h, w = image.shape[:2]
    pixels = []

    for _ in range(sample_size):
        y, x = random.randint(0, h-1), random.randint(0, w-1)
        pixels.append(image[y, x])

    return pixels


def build_pixel_database(reference_dir: str, sample_size: int = 100) -> Dict[str, List[np.ndarray]]:
    """Build a pixel database from reference images organized in directories"""
    pixel_db = {}

    if not os.path.exists(reference_dir):
        raise ValueError(f"Reference directory does not exist: {reference_dir}")

    for root, dirs, files in os.walk(reference_dir):
        for filename in files:
            if is_image_file(filename):
                image_path = os.path.join(root, filename)
                category = os.path.relpath(root, reference_dir)

                # Handle root directory case
                if category == '.':
                    category = 'root'

                print(f"Sampling {image_path} → {category}")

                image = load_image(image_path)
                if image is None:
                    continue

                if category not in pixel_db:
                    pixel_db[category] = []

                sampled_pixels = sample_pixels_from_image(image, sample_size)
                pixel_db[category].extend(sampled_pixels)

    return pixel_db


def save_pixel_database(pixel_db: Dict[str, List[np.ndarray]], filename: str = 'pixel_db.pkl') -> bool:
    """Save pixel database to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(pixel_db, f)
        return True
    except Exception as e:
        print(f"Error saving pixel database: {e}")
        return False


def load_pixel_database(filename: str = 'pixel_db.pkl') -> Optional[Dict[str, List[np.ndarray]]]:
    """Load pixel database from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pixel database: {e}")
        return None


def get_database_stats(pixel_db: Dict[str, List[np.ndarray]]) -> Dict[str, int]:
    """Get statistics about the pixel database"""
    stats = {
        'categories': len(pixel_db),
        'total_pixels': sum(len(pixels) for pixels in pixel_db.values()),
        'pixels_per_category': {cat: len(pixels) for cat, pixels in pixel_db.items()}
    }
    return stats


if __name__ == "__main__":
    print("Pixel Sampler starting...")
    ref_dir = input("Reference directory path: ")
    sample_size = int(input("Pixels per image to sample: "))

    pixel_db = build_pixel_database(ref_dir, sample_size)

    if pixel_db:
        stats = get_database_stats(pixel_db)
        print(f"Built database with {stats['categories']} categories and {stats['total_pixels']} pixels")

        if save_pixel_database(pixel_db):
            print("✓ Pixel database saved successfully")
        else:
            print("✗ Failed to save pixel database")
    else:
        print("✗ Failed to build pixel database")
