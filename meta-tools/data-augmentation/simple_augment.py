#!/usr/bin/env python3

import os
import cv2
import json
import numpy as np
import random
from pathlib import Path
from datetime import datetime

class SimpleAugmentor:
    def __init__(self):
        self.created_count = 0
        random.seed(42)
        np.random.seed(42)
        
    def rotate_image(self, image, angle):
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def adjust_brightness(self, image, factor):
        """Adjust image brightness."""
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def add_gaussian_noise(self, image, var=0.01):
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, var * 255, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def simulate_scratch(self, image):
        """Add scratch-like defects."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Create 1-3 scratches
        for _ in range(random.randint(1, 3)):
            # Random line
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            thickness = random.randint(1, 3)
            color = random.randint(20, 100)
            
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
            
        return img_copy
    
    def simulate_contamination(self, image):
        """Add contamination patterns."""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Add circular contamination spots
        for _ in range(random.randint(2, 5)):
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(10, 30)
            
            # Create gradient effect
            y, x = np.ogrid[:h, :w]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            
            # Apply contamination with transparency
            contamination = np.ones_like(image) * random.randint(100, 200)
            alpha = 0.3
            img_copy[mask] = (1 - alpha) * img_copy[mask] + alpha * contamination[mask]
            
        return np.clip(img_copy, 0, 255).astype(np.uint8)
    
    def augment_batch(self, image_path, num_augmentations=5):
        """Generate multiple augmented versions of an image."""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        augmented = []
        
        for i in range(num_augmentations):
            aug_img = img.copy()
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                aug_img = self.rotate_image(aug_img, angle)
            
            # Random brightness
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                aug_img = self.adjust_brightness(aug_img, factor)
            
            # Add noise
            if random.random() > 0.3:
                aug_img = self.add_gaussian_noise(aug_img, 0.005)
            
            # Flip
            if random.random() > 0.5:
                aug_img = cv2.flip(aug_img, random.choice([0, 1]))
            
            augmented.append(aug_img)
            
        return augmented
    
    def process_dataset(self):
        """Process the dataset and create augmented samples."""
        print("Starting simple augmentation process...")
        
        # Create directories
        aug_base = 'ml_optimized/augmented'
        for condition in ['clean', 'contaminated', 'scratched', 'heavily_contaminated']:
            for fiber in ['50um', '91um', 'sma']:
                os.makedirs(f"{aug_base}/{condition}/{fiber}", exist_ok=True)
        
        stats = {
            'clean': 0,
            'scratched': 0,
            'contaminated': 0,
            'heavily_contaminated': 0
        }
        
        # Process clean images
        clean_sources = {
            '50um': list(Path('ml_optimized/train/clean/50um').glob('*.jpg')),
            '91um': list(Path('ml_optimized/train/clean/91um').glob('*.jpg')),
            'sma': list(Path('ml_optimized/train/clean/sma').glob('*.jpg'))
        }
        
        for fiber_type, images in clean_sources.items():
            print(f"\nProcessing {len(images)} {fiber_type} clean images...")
            
            for img_path in images[:10]:  # Limit to prevent too many augmentations
                # Generate clean augmentations
                clean_augs = self.augment_batch(str(img_path), 2)
                for aug_img in clean_augs:
                    filename = f"aug_clean_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{aug_base}/clean/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, aug_img)
                    self.created_count += 1
                    stats['clean'] += 1
                
                # Create synthetic scratched versions
                scratch_augs = self.augment_batch(str(img_path), 3)
                for aug_img in scratch_augs:
                    scratched = self.simulate_scratch(aug_img)
                    filename = f"aug_scratched_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{aug_base}/scratched/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, scratched)
                    self.created_count += 1
                    stats['scratched'] += 1
                
                # Create synthetic contaminated versions
                contam_augs = self.augment_batch(str(img_path), 2)
                for aug_img in contam_augs:
                    contaminated = self.simulate_contamination(aug_img)
                    filename = f"aug_contaminated_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{aug_base}/contaminated/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, contaminated)
                    self.created_count += 1
                    stats['contaminated'] += 1
                
                # Create heavily contaminated (both defects)
                heavy_augs = self.augment_batch(str(img_path), 2)
                for aug_img in heavy_augs:
                    heavy = self.simulate_contamination(self.simulate_scratch(aug_img))
                    filename = f"aug_heavy_{fiber_type}_{self.created_count:05d}.jpg"
                    save_path = f"{aug_base}/heavily_contaminated/{fiber_type}/{filename}"
                    cv2.imwrite(save_path, heavy)
                    self.created_count += 1
                    stats['heavily_contaminated'] += 1
        
        # Also augment some contaminated images
        contam_images = list(Path('ml_optimized/train/contaminated/91um').glob('*.jpg'))[:10]
        print(f"\nAugmenting {len(contam_images)} existing contaminated images...")
        
        for img_path in contam_images:
            contam_augs = self.augment_batch(str(img_path), 2)
            for aug_img in contam_augs:
                filename = f"aug_contaminated_91um_{self.created_count:05d}.jpg"
                save_path = f"{aug_base}/contaminated/91um/{filename}"
                cv2.imwrite(save_path, aug_img)
                self.created_count += 1
                stats['contaminated'] += 1
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_generated': self.created_count,
            'distribution': stats,
            'method': 'simple_augmentation'
        }
        
        with open(f'{aug_base}/augmentation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Simple augmentation complete!")
        print(f"Total augmented images: {self.created_count}")
        print(f"Distribution: {stats}")
        
        # Create summary of full dataset
        self.create_full_dataset_summary()
    
    def create_full_dataset_summary(self):
        """Create a summary of the complete ML-ready dataset."""
        summary = {
            'dataset_name': 'Fiber Optic Connector End-face ML Dataset',
            'version': '2.0',
            'last_updated': datetime.now().isoformat(),
            'structure': {},
            'total_images': 0
        }
        
        # Count all images
        for split in ['train', 'val', 'test', 'augmented']:
            split_path = f'ml_optimized/{split}'
            if not os.path.exists(split_path):
                continue
                
            summary['structure'][split] = {}
            
            for condition in ['clean', 'contaminated', 'scratched', 'heavily_contaminated']:
                cond_path = f'{split_path}/{condition}'
                if not os.path.exists(cond_path):
                    continue
                    
                summary['structure'][split][condition] = {}
                
                for fiber in ['50um', '91um', 'sma']:
                    fiber_path = f'{cond_path}/{fiber}'
                    if os.path.exists(fiber_path):
                        count = len(list(Path(fiber_path).glob('*.jpg')))
                        if count > 0:
                            summary['structure'][split][condition][fiber] = count
                            summary['total_images'] += count
        
        # Save summary
        with open('ml_optimized/metadata/full_dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Full dataset summary saved to ml_optimized/metadata/full_dataset_summary.json")
        print(f"Total images in ML dataset: {summary['total_images']}")


if __name__ == "__main__":
    augmentor = SimpleAugmentor()
    augmentor.process_dataset()