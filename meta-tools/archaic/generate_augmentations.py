#!/usr/bin/env python3

import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

class FiberOpticAugmentor:
    def __init__(self, config_path: str = 'ml_optimized/metadata/augmentation_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Basic augmentations
        self.basic_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ])
        
        self.created_count = 0
        
    def simulate_scratch(self, image: np.ndarray) -> np.ndarray:
        """Simulate realistic fiber optic scratches."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Random number of scratches
        num_scratches = random.randint(1, 5)
        
        for _ in range(num_scratches):
            # Random scratch parameters
            thickness = random.randint(1, 3)
            length = random.randint(20, min(h, w) // 2)
            
            # Random start point
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            
            # Random angle
            angle = random.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Random intensity (darker scratch)
            intensity = random.randint(0, 100)
            
            # Draw the scratch
            cv2.line(img_copy, (x1, y1), (x2, y2), intensity, thickness)
            
            # Add some noise around the scratch
            if random.random() > 0.5:
                kernel = np.ones((3, 3), np.uint8)
                scratch_mask = np.zeros_like(img_copy)
                cv2.line(scratch_mask, (x1, y1), (x2, y2), 255, thickness)
                dilated = cv2.dilate(scratch_mask, kernel, iterations=1)
                noise = np.random.normal(0, 20, img_copy.shape).astype(np.uint8)
                img_copy[dilated > 0] += noise[dilated > 0]
                
        return np.clip(img_copy, 0, 255).astype(np.uint8)
    
    def simulate_oil_contamination(self, image: np.ndarray) -> np.ndarray:
        """Simulate oil or liquid contamination patterns."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Number of oil spots
        num_spots = random.randint(1, 5)
        
        for _ in range(num_spots):
            # Random center
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            
            # Random radius
            radius = random.randint(20, min(h, w) // 6)
            
            # Create interference pattern (oil creates rainbow-like patterns)
            y, x = np.ogrid[:h, :w]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            
            # Create concentric interference pattern
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            pattern = np.sin(distance * 0.3) * 30
            
            # Apply pattern with transparency
            alpha = 0.3 + random.random() * 0.4
            img_copy[mask] = (1 - alpha) * img_copy[mask] + alpha * (img_copy[mask] + pattern[mask])
            
        return np.clip(img_copy, 0, 255).astype(np.uint8)
    
    def simulate_dust_particles(self, image: np.ndarray) -> np.ndarray:
        """Simulate dust particles on the fiber surface."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Number of dust particles
        num_particles = random.randint(5, 20)
        
        for _ in range(num_particles):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random size
            size = random.randint(1, 5)
            
            # Random intensity (usually dark)
            intensity = random.randint(0, 100)
            
            # Draw dust particle
            cv2.circle(img_copy, (x, y), size, intensity, -1)
            
            # Add some blur to make it realistic
            if size > 2:
                roi_x1 = max(0, x - size - 2)
                roi_x2 = min(w, x + size + 2)
                roi_y1 = max(0, y - size - 2)
                roi_y2 = min(h, y + size + 2)
                
                roi = img_copy[roi_y1:roi_y2, roi_x1:roi_x2]
                blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
                img_copy[roi_y1:roi_y2, roi_x1:roi_x2] = blurred_roi
                
        return img_copy
    
    def create_heavily_contaminated(self, image: np.ndarray) -> np.ndarray:
        """Create heavily contaminated samples by combining defects."""
        img = image.copy()
        
        # Apply multiple contamination types
        if random.random() > 0.3:
            img = self.simulate_oil_contamination(img)
        
        if random.random() > 0.3:
            img = self.simulate_scratch(img)
        
        if random.random() > 0.5:
            img = self.simulate_dust_particles(img)
        
        # Apply additional noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def augment_image(self, image_path: str, augmentation_type: str) -> List[np.ndarray]:
        """Generate augmented versions of an image."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading {image_path}")
            return []
        
        augmented_images = []
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply basic augmentations
        for i in range(3):  # Generate 3 basic augmentations
            augmented = self.basic_transform(image=img)['image']
            augmented_images.append(augmented)
        
        # Apply specific augmentations based on type
        if augmentation_type == 'scratch':
            # Generate scratched versions
            for i in range(2):
                base = self.basic_transform(image=img)['image']
                scratched = self.simulate_scratch(base)
                augmented_images.append(scratched)
        
        elif augmentation_type == 'contaminated':
            # Generate more contaminated versions
            for i in range(2):
                base = self.basic_transform(image=img)['image']
                if random.random() > 0.5:
                    contaminated = self.simulate_oil_contamination(base)
                else:
                    contaminated = self.simulate_dust_particles(base)
                augmented_images.append(contaminated)
        
        elif augmentation_type == 'heavily_contaminated':
            # Generate heavily contaminated versions
            for i in range(3):
                base = self.basic_transform(image=img)['image']
                heavy = self.create_heavily_contaminated(base)
                augmented_images.append(heavy)
        
        return augmented_images
    
    def process_dataset(self, source_dir: str = 'ml_optimized/train', 
                       target_dir: str = 'ml_optimized/augmented'):
        """Process the entire training dataset."""
        print("Starting augmentation process...")
        
        # Create augmented directories
        for condition in ['clean', 'contaminated', 'scratched', 'heavily_contaminated']:
            for fiber in ['50um', '91um', 'sma']:
                os.makedirs(f"{target_dir}/{condition}/{fiber}", exist_ok=True)
        
        # Process clean images to create synthetic defects
        clean_dirs = [
            'ml_optimized/train/clean/50um',
            'ml_optimized/train/clean/91um',
            'ml_optimized/train/clean/sma'
        ]
        
        augmentation_stats = {
            'clean': 0,
            'scratched': 0,
            'contaminated': 0,
            'heavily_contaminated': 0
        }
        
        for clean_dir in clean_dirs:
            if not os.path.exists(clean_dir):
                continue
                
            fiber_type = os.path.basename(clean_dir)
            images = list(Path(clean_dir).glob('*.jpg')) + list(Path(clean_dir).glob('*.png'))
            
            print(f"\nProcessing {len(images)} images from {clean_dir}")
            
            for img_path in images:
                # Generate clean augmentations
                clean_augs = self.augment_image(str(img_path), 'clean')
                for i, aug_img in enumerate(clean_augs[:2]):  # Save 2 clean augmentations
                    filename = f"aug_clean_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{target_dir}/clean/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.created_count += 1
                    augmentation_stats['clean'] += 1
                
                # Generate scratched versions
                scratch_augs = self.augment_image(str(img_path), 'scratch')
                for i, aug_img in enumerate(scratch_augs):
                    filename = f"aug_scratched_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{target_dir}/scratched/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.created_count += 1
                    augmentation_stats['scratched'] += 1
                
                # Generate contaminated versions
                contam_augs = self.augment_image(str(img_path), 'contaminated')
                for i, aug_img in enumerate(contam_augs[:2]):
                    filename = f"aug_contaminated_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{target_dir}/contaminated/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.created_count += 1
                    augmentation_stats['contaminated'] += 1
                
                # Generate heavily contaminated versions
                heavy_augs = self.augment_image(str(img_path), 'heavily_contaminated')
                for i, aug_img in enumerate(heavy_augs):
                    filename = f"aug_heavy_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{target_dir}/heavily_contaminated/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.created_count += 1
                    augmentation_stats['heavily_contaminated'] += 1
                
                if self.created_count % 50 == 0:
                    print(f"Generated {self.created_count} augmented images...")
        
        # Also augment existing contaminated images
        contam_dirs = ['ml_optimized/train/contaminated/91um']
        
        for contam_dir in contam_dirs:
            if not os.path.exists(contam_dir):
                continue
                
            fiber_type = os.path.basename(contam_dir)
            images = list(Path(contam_dir).glob('*.jpg'))[:20]  # Limit to balance dataset
            
            print(f"\nAugmenting {len(images)} contaminated images from {contam_dir}")
            
            for img_path in images:
                # Generate more contaminated variations
                contam_augs = self.augment_image(str(img_path), 'contaminated')
                for i, aug_img in enumerate(contam_augs[:1]):
                    filename = f"aug_contaminated_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{target_dir}/contaminated/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    self.created_count += 1
                    augmentation_stats['contaminated'] += 1
        
        # Save augmentation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_generated': self.created_count,
            'distribution': augmentation_stats,
            'config_used': self.config
        }
        
        with open(f'{target_dir}/augmentation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Augmentation complete!")
        print(f"Generated {self.created_count} augmented images")
        print(f"Distribution: {augmentation_stats}")


if __name__ == "__main__":
    augmentor = FiberOpticAugmentor()
    augmentor.process_dataset()