from PIL import Image
import json
import time
import os

def read_pixels(image_path):
    print(f"[PIXEL_READER] Reading {image_path}")
    img = Image.open(image_path).convert('L')
    pixels = list(img.getdata())
    
    data = {
        'timestamp': time.time(),
        'image': image_path,
        'pixels': pixels,
        'size': img.size
    }
    
    with open('pixel_data.json', 'w') as f:
        json.dump(data, f)
    
    print(f"[PIXEL_READER] Read {len(pixels)} pixels")
    return pixels

if __name__ == "__main__":
    # Example usage - reads the first image it finds
    image_found = False
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            pixels = read_pixels(file)
            # Print all pixels continuously (like original intensity-reader.py)
            for p in pixels:
                print(p)
            image_found = True
            break
    
    if not image_found:
        print("[PIXEL_READER] No image found in current directory")