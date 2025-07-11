#!/usr/bin/env python3
import os
import shutil
import json
import hashlib
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_condition(directory_name):
    """Extract condition from directory name."""
    if 'dirty-scratched-oil-wet-blob' in directory_name:
        return 'heavily_contaminated'
    elif 'scratched' in directory_name:
        return 'scratched'
    elif 'dirty-oil-wet-blob' in directory_name:
        return 'contaminated'
    elif 'core-separated' in directory_name:
        return 'core_separated'
    elif 'clean' in directory_name:
        return 'clean'
    else:
        return 'uncategorized'

def extract_fiber_type(directory_name):
    """Extract fiber type from directory name."""
    if '91' in directory_name:
        return '91'
    elif '50' in directory_name:
        return '50'
    elif 'sma' in directory_name.lower():
        return 'sma'
    else:
        return 'unknown'

def generate_new_filename(condition, fiber_type, index, original_ext):
    """Generate a standardized filename."""
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{fiber_type}_{condition}_{timestamp}_{index:04d}{original_ext}"
    return filename

def analyze_image_quality(image_path: str) -> Dict:
    """Analyze image quality metrics."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {}
        
        # Calculate quality metrics
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        
        # Detect blur using Laplacian variance
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # Edge detection for feature richness
        edges = cv2.Canny(img, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'sharpness': float(laplacian_var),
            'edge_ratio': float(edge_ratio),
            'is_blurry': bool(laplacian_var < 100)
        }
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return {}

def create_train_val_test_split(images: List[Dict], ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict[str, List[Dict]]:
    """Create stratified train/val/test splits."""
    random.seed(42)
    
    # Group by category
    category_images = defaultdict(list)
    for img in images:
        key = f"{img['fiber_type']}_{img['condition']}"
        category_images[key].append(img)
    
    # Create splits
    splits = {'train': [], 'val': [], 'test': []}
    
    for category, cat_images in category_images.items():
        random.shuffle(cat_images)
        n = len(cat_images)
        
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        
        splits['train'].extend(cat_images[:n_train])
        splits['val'].extend(cat_images[n_train:n_train + n_val])
        splits['test'].extend(cat_images[n_train + n_val:])
    
    return splits

def main():
    # Create ML-optimized directory structure
    base_dir = 'ml_optimized'
    
    # Main splits
    for split in ['train', 'val', 'test']:
        for condition in ['clean', 'contaminated', 'scratched', 'heavily_contaminated']:
            for fiber in ['50um', '91um', 'sma']:
                os.makedirs(f'{base_dir}/{split}/{condition}/{fiber}', exist_ok=True)
    
    # Additional directories
    os.makedirs(f'{base_dir}/metadata', exist_ok=True)
    os.makedirs(f'{base_dir}/augmented', exist_ok=True)
    os.makedirs(f'{base_dir}/visualizations', exist_ok=True)
    
    # Create directories
    os.makedirs('reorganized_dataset/by_condition/clean', exist_ok=True)
    os.makedirs('reorganized_dataset/by_condition/scratched', exist_ok=True)
    os.makedirs('reorganized_dataset/by_condition/contaminated', exist_ok=True)
    os.makedirs('reorganized_dataset/by_condition/heavily_contaminated', exist_ok=True)
    os.makedirs('reorganized_dataset/by_condition/core_separated', exist_ok=True)
    os.makedirs('reorganized_dataset/by_condition/uncategorized', exist_ok=True)
    
    os.makedirs('reorganized_dataset/by_fiber_type/91', exist_ok=True)
    os.makedirs('reorganized_dataset/by_fiber_type/50', exist_ok=True)
    os.makedirs('reorganized_dataset/by_fiber_type/sma', exist_ok=True)
    os.makedirs('reorganized_dataset/by_fiber_type/unknown', exist_ok=True)
    
    os.makedirs('reorganized_dataset/combined', exist_ok=True)
    os.makedirs('reorganized_dataset/duplicates_removed', exist_ok=True)
    
    # Track processed files and duplicates
    processed_hashes = set()
    duplicate_log = []
    rename_log = []
    
    # Counter for each category
    counters = defaultdict(int)
    
    # Process all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    processed_hashes = set()
    duplicate_count = 0
    
    print("Phase 1: Collecting and analyzing images...")
    
    for root, dirs, files in os.walk('.'):
        # Skip processed directories
        if any(skip in root for skip in ['ml_optimized', 'reorganized_dataset', '.git', '.claude']):
            continue
            
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                original_path = os.path.join(root, file)
                file_hash = get_file_hash(original_path)
                
                # Check for duplicates
                if file_hash in processed_hashes:
                    duplicate_count += 1
                    continue
                
                processed_hashes.add(file_hash)
                
                # Extract metadata
                condition = extract_condition(root)
                fiber_type = extract_fiber_type(root)
                
                # Map fiber types to new convention
                fiber_map = {'91': '91um', '50': '50um', 'sma': 'sma', 'unknown': '91um'}
                fiber_type = fiber_map.get(fiber_type, '91um')
                
                # Analyze image quality
                quality_metrics = analyze_image_quality(original_path)
                
                # Store image info
                image_info = {
                    'original_path': original_path,
                    'filename': file,
                    'condition': condition,
                    'fiber_type': fiber_type,
                    'hash': file_hash,
                    'quality': quality_metrics
                }
                
                all_images.append(image_info)
                
                if len(all_images) % 100 == 0:
                    print(f"Analyzed {len(all_images)} images...")
    
    print(f"\nPhase 1 complete: Found {len(all_images)} unique images, {duplicate_count} duplicates removed.")
    
    # Phase 2: Create train/val/test splits
    print("\nPhase 2: Creating train/val/test splits...")
    splits = create_train_val_test_split(all_images)
    
    # Phase 3: Copy and organize files
    print("\nPhase 3: Organizing files...")
    split_stats = defaultdict(lambda: defaultdict(int))
    
    for split_name, images in splits.items():
        for idx, img_info in enumerate(images):
            original_path = img_info['original_path']
            condition = img_info['condition']
            fiber_type = img_info['fiber_type']
            
            # Handle special cases
            if condition == 'core_separated':
                condition = 'contaminated'
            elif condition == 'uncategorized':
                # Try to infer from path
                if 'clean' in original_path.lower():
                    condition = 'clean'
                elif 'contam' in original_path.lower() or 'dirty' in original_path.lower():
                    condition = 'contaminated'
                else:
                    condition = 'contaminated'  # default
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_id = random.randint(1000, 9999)
            ext = os.path.splitext(img_info['filename'])[1]
            new_filename = f"{fiber_type}_{condition}_{split_name}_{idx:04d}_{random_id}{ext}"
            
            # Create target path
            target_dir = f"{base_dir}/{split_name}/{condition}/{fiber_type}"
            target_path = os.path.join(target_dir, new_filename)
            
            # Copy file
            try:
                shutil.copy2(original_path, target_path)
                split_stats[split_name][f"{fiber_type}_{condition}"] += 1
                
                # Add to image info
                img_info['new_path'] = target_path
                img_info['new_filename'] = new_filename
                
            except Exception as e:
                print(f"Error copying {original_path}: {e}")
    
    # Phase 4: Generate metadata and reports
    print("\nPhase 4: Generating metadata and reports...")
    
    # Create comprehensive metadata
    metadata = {
        'dataset_info': {
            'name': 'Fiber Optic Connector End-face Dataset',
            'version': '2.0',
            'creation_date': datetime.now().isoformat(),
            'total_images': len(all_images),
            'duplicates_removed': duplicate_count,
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15}
        },
        'classes': {
            'conditions': ['clean', 'contaminated', 'scratched', 'heavily_contaminated'],
            'fiber_types': ['50um', '91um', 'sma']
        },
        'splits': {}
    }
    
    # Add split statistics
    for split_name in ['train', 'val', 'test']:
        split_data = [img for img in all_images if img.get('new_path', '').find(f'/{split_name}/') != -1]
        metadata['splits'][split_name] = {
            'total': len(split_data),
            'distribution': dict(split_stats[split_name])
        }
    
    # Save metadata
    with open(f'{base_dir}/metadata/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save detailed image catalog
    with open(f'{base_dir}/metadata/image_catalog.json', 'w') as f:
        json.dump(all_images, f, indent=2)
    
    # Create augmentation config
    augmentation_config = {
        'transformations': {
            'rotation': {'degrees': 15, 'probability': 0.5},
            'flip': {'horizontal': True, 'vertical': True, 'probability': 0.5},
            'brightness': {'limit': 0.2, 'probability': 0.5},
            'contrast': {'limit': 0.2, 'probability': 0.5},
            'gaussian_noise': {'var': 0.01, 'probability': 0.3},
            'blur': {'limit': 3, 'probability': 0.2}
        },
        'fiber_specific': {
            'simulate_dust': {'probability': 0.2, 'severity': [0.1, 0.3]},
            'simulate_oil': {'probability': 0.15, 'severity': [0.1, 0.4]},
            'simulate_scratch': {'probability': 0.1, 'width': [1, 3], 'length': [20, 100]}
        }
    }
    
    with open(f'{base_dir}/metadata/augmentation_config.json', 'w') as f:
        json.dump(augmentation_config, f, indent=2)
    
    # Create training config
    training_config = {
        'model': {
            'architecture': 'ResNet50',
            'pretrained': True,
            'num_classes': 4,
            'input_size': [224, 224]
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 10
            },
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.001
            }
        },
        'loss': {
            'type': 'CrossEntropyLoss',
            'class_weights': 'balanced'
        }
    }
    
    with open(f'{base_dir}/metadata/training_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Create a README
    readme_content = f"""# Reorganized Fiber Optic Dataset

## Summary
- Total unique images: {len(rename_log)}
- Duplicates removed: {len(duplicate_log)}
- Date reorganized: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Directory Structure

### by_condition/
Images organized by their condition:
- **clean/**: Clean fiber optic images with no defects
- **scratched/**: Images showing scratch defects
- **contaminated/**: Images with oil, wet spots, or blob contamination
- **heavily_contaminated/**: Images with multiple defects (scratches + contamination)
- **core_separated/**: Images showing core separation
- **uncategorized/**: Images that couldn't be categorized

### by_fiber_type/
Images organized by fiber type:
- **91/**: Type 91 fiber optic cables
- **50/**: Type 50 fiber optic cables
- **sma/**: SMA connector fiber optic cables
- **unknown/**: Unknown fiber type

### combined/
All unique images in a single directory with standardized naming.

## Naming Convention
`[fiber_type]_[condition]_[date]_[index].[ext]`

Example: `91_clean_20250705_0001.jpg`

## Category Distribution
"""
    
    for category, count in sorted(counters.items()):
        readme_content += f"- {category}: {count} images\n"
    
    with open('reorganized_dataset/README.md', 'w') as f:
        f.write(readme_content)
    
    # Generate comprehensive report
    report_content = f"""# ML-Optimized Fiber Optic Dataset Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total unique images: {len(all_images)}
- Duplicates removed: {duplicate_count}
- Image quality analysis: Complete

## Split Distribution

### Train Set ({metadata['splits']['train']['total']} images)
"""
    for category, count in sorted(split_stats['train'].items()):
        report_content += f"- {category}: {count}\n"
    
    report_content += f"\n### Validation Set ({metadata['splits']['val']['total']} images)\n"
    for category, count in sorted(split_stats['val'].items()):
        report_content += f"- {category}: {count}\n"
    
    report_content += f"\n### Test Set ({metadata['splits']['test']['total']} images)\n"
    for category, count in sorted(split_stats['test'].items()):
        report_content += f"- {category}: {count}\n"
    
    report_content += """\n## Quality Metrics Summary

Images have been analyzed for:
- Sharpness (Laplacian variance)
- Mean and standard deviation of intensity
- Edge feature richness
- Blur detection

## Augmentation Strategy

1. **Geometric Transformations**
   - Random rotation (±15°)
   - Horizontal and vertical flips

2. **Color/Intensity Adjustments**
   - Brightness variation (±20%)
   - Contrast variation (±20%)

3. **Noise Simulation**
   - Gaussian noise
   - Fiber-specific contamination patterns

4. **Domain-Specific Augmentations**
   - Dust particle simulation
   - Oil contamination patterns
   - Scratch generation

## Next Steps

1. Run `generate_augmentations.py` to create augmented training data
2. Use `train_model.py` with the provided configuration
3. Monitor training with TensorBoard logs
4. Evaluate model performance on test set

## Directory Structure
```
ml_optimized/
├── train/
│   ├── clean/
│   │   ├── 50um/
│   │   ├── 91um/
│   │   └── sma/
│   ├── contaminated/
│   ├── scratched/
│   └── heavily_contaminated/
├── val/
├── test/
├── metadata/
│   ├── dataset_metadata.json
│   ├── image_catalog.json
│   ├── augmentation_config.json
│   └── training_config.json
└── augmented/
```
"""
    
    with open(f'{base_dir}/metadata/dataset_report.md', 'w') as f:
        f.write(report_content)
    
    print(f"\n✅ ML-Optimized dataset reorganization complete!")
    print(f"  - Processed {len(all_images)} unique images")
    print(f"  - Removed {duplicate_count} duplicates")
    print(f"  - Train: {metadata['splits']['train']['total']} | Val: {metadata['splits']['val']['total']} | Test: {metadata['splits']['test']['total']}")
    print(f"  - Results saved in '{base_dir}/' directory")
    print(f"\n📊 Check '{base_dir}/metadata/dataset_report.md' for detailed report")

if __name__ == "__main__":
    main()