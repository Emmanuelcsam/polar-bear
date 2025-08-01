#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Augmentation parameters
AUGMENTATION_FACTOR = 5  # Generate 5 augmented versions per image
MIN_IMAGES_PER_CLASS = 50  # Target minimum images per class

def rotate_image(image, angle):
    """Rotate image by specified angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(0, 0, 0))
    return rotated

def add_gaussian_noise(image, mean=0, std=5):
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def adjust_brightness(image, factor):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def elastic_transform(image, alpha=20, sigma=3):
    """Apply elastic deformation"""
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    return cv2.remap(image, indices[1].reshape(shape).astype(np.float32), 
                     indices[0].reshape(shape).astype(np.float32), 
                     cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def augment_image(image_path, output_dir, base_name, aug_index):
    """Apply random augmentations to an image"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    # Randomly select augmentations
    augmentations = []
    
    # Rotation (random angle between -30 and 30 degrees)
    if random.random() > 0.3:
        angle = random.uniform(-30, 30)
        image = rotate_image(image, angle)
        augmentations.append(f"rot{int(angle)}")
    
    # Brightness adjustment
    if random.random() > 0.4:
        factor = random.uniform(0.7, 1.3)
        image = adjust_brightness(image, factor)
        augmentations.append(f"bright{int(factor*100)}")
    
    # Gaussian noise
    if random.random() > 0.5:
        std = random.uniform(2, 8)
        image = add_gaussian_noise(image, std=std)
        augmentations.append(f"noise{int(std)}")
    
    # Gaussian blur (mild, to simulate slight focus issues)
    if random.random() > 0.6:
        kernel = random.choice([3, 5])
        image = apply_gaussian_blur(image, kernel)
        augmentations.append(f"blur{kernel}")
    
    # Elastic deformation (for scratch class)
    if 'scratch' in str(image_path) and random.random() > 0.5:
        image = elastic_transform(image, alpha=random.uniform(10, 30))
        augmentations.append("elastic")
    
    # Create augmented filename
    aug_str = '_'.join(augmentations) if augmentations else 'copy'
    new_name = f"{base_name}_aug{aug_index:03d}_{aug_str}.png"
    output_path = output_dir / new_name
    
    # Save augmented image
    cv2.imwrite(str(output_path), image)
    return new_name

def augment_dataset():
    """Augment the dataset to increase training samples"""
    dataset_dir = Path('./dataset')
    augmented_metadata = []
    
    # Load existing metadata
    with open(dataset_dir / 'dataset_metadata.json', 'r') as f:
        original_metadata = json.load(f)
    
    # Process each split and class
    for split in ['train', 'val']:  # Don't augment test set
        split_dir = dataset_dir / split
        
        for class_name in ['scratch', 'clean', 'contaminated']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get existing images
            existing_images = list(class_dir.glob('*.png'))
            n_existing = len(existing_images)
            
            if n_existing == 0:
                continue
            
            print(f"\nAugmenting {split}/{class_name}: {n_existing} original images")
            
            # Calculate how many augmentations needed
            if split == 'train':
                target_count = max(MIN_IMAGES_PER_CLASS, n_existing * (AUGMENTATION_FACTOR + 1))
            else:
                target_count = n_existing * 3  # Less augmentation for validation
            
            augmentations_needed = target_count - n_existing
            aug_per_image = augmentations_needed // n_existing
            
            # Augment each image
            aug_count = 0
            for img_path in existing_images:
                base_name = img_path.stem
                
                for i in range(aug_per_image):
                    new_name = augment_image(img_path, class_dir, base_name, i)
                    if new_name:
                        augmented_metadata.append({
                            'original_name': img_path.name,
                            'new_name': new_name,
                            'class': class_name,
                            'label': original_metadata['classes'][class_name]['label'],
                            'split': split,
                            'augmented': True
                        })
                        aug_count += 1
            
            print(f"  Created {aug_count} augmented images")
    
    # Save augmented metadata
    augmented_meta_path = dataset_dir / 'augmented_metadata.json'
    with open(augmented_meta_path, 'w') as f:
        json.dump({
            'original_count': len(original_metadata['images']),
            'augmented_count': len(augmented_metadata),
            'total_count': len(original_metadata['images']) + len(augmented_metadata),
            'augmentations': augmented_metadata
        }, f, indent=2)
    
    print(f"\nAugmentation complete! Created {len(augmented_metadata)} new images.")
    print(f"Augmentation metadata saved to: {augmented_meta_path}")
    
    # Create training config
    create_training_config()

def create_training_config():
    """Create a basic neural network training configuration"""
    config = {
        "model": {
            "architecture": "resnet18",
            "pretrained": True,
            "num_classes": 3,
            "input_size": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 50,
            "early_stopping_patience": 10,
            "optimizer": "adam",
            "loss_function": "cross_entropy",
            "class_weights": "balanced"
        },
        "augmentation": {
            "random_rotation": 30,
            "random_horizontal_flip": 0.5,
            "random_vertical_flip": 0.5,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05
            }
        },
        "classes": {
            "0": "scratch",
            "1": "clean",
            "2": "contaminated"
        }
    }
    
    with open('./dataset/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining configuration saved to: dataset/training_config.json")

if __name__ == "__main__":
    # Check if OpenCV is available
    try:
        import cv2
        augment_dataset()
    except ImportError:
        print("OpenCV not installed. Installing required packages...")
        os.system("pip install opencv-python numpy")
        print("Please run the script again after installation.")