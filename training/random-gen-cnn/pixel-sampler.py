import os
import numpy as np
from PIL import Image
import pickle
import random

print("Pixel Sampler starting...")
ref_dir = input("Reference directory path: ")
sample_size = int(input("Pixels per image to sample: "))

pixel_db = {}
for root, dirs, files in os.walk(ref_dir):
    for f in files:
        if f.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(root, f)
            category = root.replace(ref_dir, '').strip(os.sep)
            print(f"Sampling {path} → {category}")
            
            img = np.array(Image.open(path).convert('RGB'))
            h, w = img.shape[:2]
            
            if category not in pixel_db:
                pixel_db[category] = []
            
            for _ in range(sample_size):
                y, x = random.randint(0, h-1), random.randint(0, w-1)
                pixel_db[category].append(img[y, x])
                print(f"  Sampled pixel at ({x},{y}): {img[y,x]}")

print(f"Saving {len(pixel_db)} categories...")
with open('pixel_db.pkl', 'wb') as f:
    pickle.dump(pixel_db, f)
print("✓ Pixel database saved")