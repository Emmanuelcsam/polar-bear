import cv2
import numpy as np
import json
import os
from typing import Optional, Tuple, Dict, Any

def _load_from_json(json_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load matrix from JSON file with bounds checking."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        channels = data['image_dimensions'].get('channels', 3)
        
        matrix = np.zeros((height, width, channels), dtype=np.uint8)
        
        oob_count = 0
        for pixel in data['pixels']:
            x, y = pixel['coordinates']['x'], pixel['coordinates']['y']
            if 0 <= x < width and 0 <= y < height:
                bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                matrix[y, x] = bgr[:channels] if isinstance(bgr, list) else [bgr] * channels
            else:
                oob_count += 1
        
        if oob_count > 0:
            print(f"⚠ Warning: Skipped {oob_count} out-of-bounds pixels")
        
        metadata = {
            'filename': data.get('filename', os.path.basename(json_path)),
            'width': width, 'height': height, 'channels': channels, 'json_path': json_path
        }
        return matrix, metadata
        
    except Exception as e:
        print(f"✗ Error loading JSON {json_path}: {e}")
        return None, None

def load_image(path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load image from JSON or standard image file."""
    if path.lower().endswith('.json'):
        return _load_from_json(path)
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"✗ Could not read image: {path}")
            return None, None
        metadata = {'filename': os.path.basename(path)}
        return img, metadata

if __name__ == '__main__':
    # Create a dummy JSON file for testing
    dummy_data = {
        "filename": "dummy.json",
        "image_dimensions": {"width": 5, "height": 5, "channels": 1},
        "pixels": [
            {"coordinates": {"x": 0, "y": 0}, "intensity": 255},
            {"coordinates": {"x": 2, "y": 2}, "intensity": 128},
            {"coordinates": {"x": 4, "y": 4}, "intensity": 255}
        ]
    }
    json_path = "dummy_image.json"
    with open(json_path, 'w') as f:
        json.dump(dummy_data, f)

    print(f"Loading image from dummy JSON: {json_path}")
    image, metadata = load_image(json_path)
    if image is not None:
        print("Image loaded successfully from JSON.")
        print("Metadata:", metadata)
        print("Image shape:", image.shape)
        # print("Image content:\n", image.squeeze())
    
    # Create a dummy PNG file for testing
    png_path = "dummy_image.png"
    cv2.imwrite(png_path, np.random.randint(0, 255, (10, 10), dtype=np.uint8))
    
    print(f"\nLoading image from dummy PNG: {png_path}")
    image, metadata = load_image(png_path)
    if image is not None:
        print("Image loaded successfully from PNG.")
        print("Metadata:", metadata)
        print("Image shape:", image.shape)
