#!/usr/bin/env python3
import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_image(filepath):
    """Analyze an image and return its properties."""
    try:
        with Image.open(filepath) as img:
            # Get basic properties
            width, height = img.size
            mode = img.mode
            format = img.format
            
            # Convert to numpy array for analysis
            img_array = np.array(img.convert('L'))  # Convert to grayscale
            
            # Calculate statistics
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)
            min_intensity = np.min(img_array)
            max_intensity = np.max(img_array)
            
            # Detect if it's mostly circular content (fiber optic)
            center_y, center_x = height // 2, width // 2
            radius = min(width, height) // 4
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            center_mean = np.mean(img_array[mask])
            
            return {
                'width': width,
                'height': height,
                'mode': mode,
                'format': format,
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'min_intensity': int(min_intensity),
                'max_intensity': int(max_intensity),
                'center_mean': float(center_mean),
                'aspect_ratio': width / height
            }
    except Exception as e:
        return {'error': str(e)}

def main():
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    all_images = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                filepath = os.path.join(root, file)
                all_images.append(filepath)
    
    print(f"Found {len(all_images)} images total")
    
    # Analyze images by directory
    by_directory = defaultdict(list)
    duplicates = defaultdict(list)
    
    for img_path in all_images:
        dir_name = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        
        # Get file hash to detect duplicates
        file_hash = get_file_hash(img_path)
        duplicates[file_hash].append(img_path)
        
        # Analyze image
        analysis = analyze_image(img_path)
        
        by_directory[dir_name].append({
            'filename': filename,
            'path': img_path,
            'hash': file_hash,
            'analysis': analysis
        })
    
    # Print summary
    print("\n=== Directory Summary ===")
    for dir_path, images in sorted(by_directory.items()):
        print(f"\n{dir_path}: {len(images)} images")
        
        # Get directory classification from name
        if 'clean' in dir_path:
            condition = 'clean'
        elif 'dirty-scratched-oil-wet-blob' in dir_path:
            condition = 'heavily contaminated'
        elif 'scratched' in dir_path:
            condition = 'scratched'
        elif 'dirty-oil-wet-blob' in dir_path:
            condition = 'contaminated'
        elif 'core-separated' in dir_path:
            condition = 'core-separated'
        else:
            condition = 'unknown'
        
        print(f"  Condition: {condition}")
        
        # Calculate average properties
        valid_analyses = [img['analysis'] for img in images if 'error' not in img['analysis']]
        if valid_analyses:
            avg_mean = np.mean([a['mean_intensity'] for a in valid_analyses])
            avg_std = np.mean([a['std_intensity'] for a in valid_analyses])
            print(f"  Avg intensity: {avg_mean:.1f} ± {avg_std:.1f}")
    
    # Find duplicates
    print("\n=== Duplicate Images ===")
    duplicate_count = 0
    for file_hash, paths in duplicates.items():
        if len(paths) > 1:
            duplicate_count += 1
            print(f"\nHash: {file_hash}")
            for path in paths:
                print(f"  - {path}")
    
    print(f"\nTotal unique images: {len(all_images) - duplicate_count}")
    
    # Save detailed report
    report = {
        'total_images': len(all_images),
        'unique_images': len(all_images) - duplicate_count,
        'directories': dict(by_directory),
        'duplicates': {k: v for k, v in duplicates.items() if len(v) > 1}
    }
    
    with open('dataset_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nDetailed report saved to dataset_analysis.json")

if __name__ == "__main__":
    main()