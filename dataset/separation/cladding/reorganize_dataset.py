#!/usr/bin/env python3
import os
import shutil
import json
import hashlib
from datetime import datetime
import random

# Set random seed for reproducibility
random.seed(42)

# Define class mappings based on filename patterns
# Define filename patterns for each class; fallback is 'contaminated'
class_mappings = {
    'scratch': ['scratch'],
    'clean': ['ubet'],
    # masks includes any files to exclude from train/val/test (binary masks, analysis images)
    'masks': ['mask', 'region', 'intensity']
}

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Source directory
source_dir = '.'
dataset_dir = './dataset'

# Dataset metadata
metadata = {
    'created': datetime.now().isoformat(),
    'classes': {
        'scratch': {'description': 'Cladding surfaces with scratch defects', 'label': 0},
        'clean': {'description': 'Clean cladding surfaces with minimal defects', 'label': 1},
        'contaminated': {'description': 'Cladding surfaces with contamination or other defects', 'label': 2},
        'masks': {'description': 'Binary masks and analysis images (excluded from training)', 'label': -1}
    },
    'images': []
}

def get_file_hash(filepath):
    """Generate MD5 hash for file tracking"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

def classify_image(filename):
    """Determine class based on filename patterns"""
    filename_lower = filename.lower()
    for class_name, patterns in class_mappings.items():
        for pattern in patterns:
            if pattern in filename_lower:
                return class_name
    return 'contaminated'  # Default class for unmatched files

def rename_and_organize():
    """Rename and organize images for neural network training"""
    # Get all PNG files
    png_files = [f for f in os.listdir(source_dir) if f.endswith('.png') and f != 'fiber_optic_classification_report.md']
    
    # Classify and group files
    classified_files = {}
    for filename in png_files:
        class_name = classify_image(filename)
        if class_name not in classified_files:
            classified_files[class_name] = []
        classified_files[class_name].append(filename)
    
    # Process each class (excluding masks from train/val/test split)
    for class_name, files in classified_files.items():
        if class_name == 'masks':
            # Copy masks to a separate directory without splitting
            for idx, filename in enumerate(files):
                src = os.path.join(source_dir, filename)
                file_hash = get_file_hash(src)
                new_name = f"mask_{idx:04d}_{file_hash}.png"
                dst = os.path.join(dataset_dir, 'masks', new_name)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                
                metadata['images'].append({
                    'original_name': filename,
                    'new_name': new_name,
                    'class': class_name,
                    'label': metadata['classes'][class_name]['label'],
                    'split': 'masks',
                    'hash': file_hash
                })
                print(f"Copied mask: {filename} -> {new_name}")
        else:
            # Shuffle files for random splitting
            random.shuffle(files)
            
            # Calculate split sizes
            n_files = len(files)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            
            # Split files
            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:]
            
            # Process each split
            splits = {'train': train_files, 'val': val_files, 'test': test_files}
            
            for split_name, split_files in splits.items():
                for idx, filename in enumerate(split_files):
                    src = os.path.join(source_dir, filename)
                    file_hash = get_file_hash(src)
                    
                    # Create systematic name: class_split_index_hash.png
                    new_name = f"{class_name}_{split_name}_{idx:04d}_{file_hash}.png"
                    dst = os.path.join(dataset_dir, split_name, class_name, new_name)
                    
                    # Copy file
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    
                    # Add to metadata
                    metadata['images'].append({
                        'original_name': filename,
                        'new_name': new_name,
                        'class': class_name,
                        'label': metadata['classes'][class_name]['label'],
                        'split': split_name,
                        'hash': file_hash
                    })
                    
                    print(f"Copied {split_name}: {filename} -> {new_name}")
    
    # Save metadata
    with open(os.path.join(dataset_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create class distribution summary
    create_distribution_summary()

def create_distribution_summary():
    """Create a summary of class distribution across splits"""
    summary = {
        'total_images': len(metadata['images']),
        'class_distribution': {},
        'split_distribution': {}
    }
    
    # Count by class and split
    for img in metadata['images']:
        class_name = img['class']
        split = img['split']
        
        if class_name not in summary['class_distribution']:
            summary['class_distribution'][class_name] = 0
        summary['class_distribution'][class_name] += 1
        
        if split not in summary['split_distribution']:
            summary['split_distribution'][split] = {}
        if class_name not in summary['split_distribution'][split]:
            summary['split_distribution'][split][class_name] = 0
        summary['split_distribution'][split][class_name] += 1
    
    # Save summary
    with open(os.path.join(dataset_dir, 'distribution_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n=== Dataset Distribution Summary ===")
    print(f"Total images: {summary['total_images']}")
    print("\nClass distribution:")
    for class_name, count in summary['class_distribution'].items():
        print(f"  {class_name}: {count} ({count/summary['total_images']*100:.1f}%)")
    
    print("\nSplit distribution:")
    for split, classes in summary['split_distribution'].items():
        if split != 'masks':
            total = sum(classes.values())
            print(f"\n{split}: {total} images")
            for class_name, count in classes.items():
                print(f"  {class_name}: {count}")

if __name__ == "__main__":
    rename_and_organize()