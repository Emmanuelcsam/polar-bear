#!/usr/bin/env python3
"""
MNIST-Inspired Data Preparation Pipeline for Fiber Optic Defect Detection
Applies MNIST's rigorous data preparation methodology to create a standardized
defect classification dataset.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict, Counter
import hashlib
import shutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
import random
from sklearn.model_selection import train_test_split
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

@dataclass
class DefectSample:
    """Represents a single defect sample in our MNIST-style dataset"""
    image_data: np.ndarray  # Standardized image (e.g., 64x64)
    defect_type: str        # Classification label
    severity: str           # Secondary label
    source_info: Dict       # Original detection metadata
    sample_id: str          # Unique identifier
    preprocessing_params: Dict = field(default_factory=dict)
    augmentation_applied: List[str] = field(default_factory=list)
    
class FiberMNISTCreator:
    """
    Creates an MNIST-style dataset from fiber optic defect detection results.
    Implements MNIST's preprocessing philosophy adapted for defect images.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # MNIST-inspired parameters
        self.standard_size = tuple(self.config.get('standard_size', [64, 64]))  # Larger than MNIST's 28x28
        self.padding = self.config.get('padding', 4)  # Border padding
        self.interpolation = cv2.INTER_CUBIC  # High-quality interpolation
        
        # Defect categories (your 10 classes, like MNIST's 10 digits)
        self.defect_classes = [
            'SCRATCH', 'CRACK', 'DIG', 'PIT', 'CONTAMINATION',
            'CHIP', 'BUBBLE', 'BURN', 'ANOMALY', 'UNKNOWN'
        ]
        
        # Severity levels (additional dimension beyond MNIST)
        self.severity_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE']
        
        # Dataset storage
        self.samples = []
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'version': '1.0',
            'preprocessing_params': {},
            'class_distribution': defaultdict(int),
            'severity_distribution': defaultdict(int),
            'source_distribution': defaultdict(int)
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_dataset_from_detection_results(self, results_dir: Path, output_dir: Path):
        """
        Main pipeline to create MNIST-style dataset from your detection results.
        Implements MNIST's mixing strategy for balanced difficulty.
        """
        self.logger.info("="*60)
        self.logger.info("Creating Fiber-MNIST Dataset")
        self.logger.info("="*60)
        
        # Step 1: Collect all defect samples
        self.logger.info("Step 1: Collecting defect samples from detection results...")
        raw_samples = self._collect_raw_samples(results_dir)
        self.logger.info(f"Collected {len(raw_samples)} raw defect samples")
        
        # Step 2: Apply quality filtering (MNIST excluded unclear digits)
        self.logger.info("Step 2: Filtering samples by quality criteria...")
        filtered_samples = self._filter_samples(raw_samples)
        self.logger.info(f"Retained {len(filtered_samples)} samples after quality filtering")
        
        # Step 3: Balance dataset (MNIST had balanced classes)
        self.logger.info("Step 3: Balancing dataset across classes...")
        balanced_samples = self._balance_dataset(filtered_samples)
        
        # Step 4: Preprocess images (MNIST's standardization pipeline)
        self.logger.info("Step 4: Preprocessing images to standard format...")
        processed_samples = self._preprocess_samples(balanced_samples)
        
        # Step 5: Split into training and test sets (MNIST's 60k/10k split)
        self.logger.info("Step 5: Creating train/test split with mixed difficulty...")
        train_samples, test_samples = self._create_train_test_split(processed_samples)
        
        # Step 6: Apply data augmentation to training set
        self.logger.info("Step 6: Augmenting training data...")
        augmented_train = self._augment_training_data(train_samples)
        
        # Step 7: Save dataset in multiple formats
        self.logger.info("Step 7: Saving dataset...")
        self._save_dataset(augmented_train, test_samples, output_dir)
        
        # Step 8: Generate dataset report
        self._generate_dataset_report(output_dir)
        
        self.logger.info("="*60)
        self.logger.info("Dataset creation complete!")
        self.logger.info(f"Training samples: {len(augmented_train)}")
        self.logger.info(f"Test samples: {len(test_samples)}")
        self.logger.info("="*60)
        
    def _collect_raw_samples(self, results_dir: Path) -> List[Dict]:
        """Collect all defect detections from your pipeline results"""
        raw_samples = []
        
        # Navigate through your pipeline structure
        detection_dirs = list(results_dir.rglob("**/3_detected/*"))
        
        for detection_dir in detection_dirs:
            # Find detection reports
            report_files = list(detection_dir.glob("*_report.json"))
            
            for report_file in report_files:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Find corresponding original image
                image_name = report_file.stem.replace('_report', '')
                possible_image_paths = [
                    detection_dir / f"{image_name}.png",
                    detection_dir / f"{image_name}.jpg",
                    detection_dir.parent.parent / f"{image_name}.png",
                    detection_dir.parent.parent / f"{image_name}.jpg"
                ]
                
                image_path = None
                for path in possible_image_paths:
                    if path.exists():
                        image_path = path
                        break
                
                if not image_path:
                    continue
                
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Extract each defect
                for defect in report.get('defects', []):
                    if 'bbox' in defect and defect['bbox']:
                        x, y, w, h = defect['bbox']
                        
                        # Extract defect region with context (MNIST principle)
                        context_margin = 10
                        x1 = max(0, x - context_margin)
                        y1 = max(0, y - context_margin)
                        x2 = min(image.shape[1], x + w + context_margin)
                        y2 = min(image.shape[0], y + h + context_margin)
                        
                        defect_image = image[y1:y2, x1:x2]
                        
                        # Skip if too small
                        if defect_image.shape[0] < 10 or defect_image.shape[1] < 10:
                            continue
                        
                        raw_samples.append({
                            'image': defect_image,
                            'defect_type': defect.get('defect_type', 'UNKNOWN'),
                            'severity': defect.get('severity', 'UNKNOWN'),
                            'confidence': defect.get('confidence', 0.5),
                            'area_px': defect.get('area_px', 0),
                            'source_file': str(report_file),
                            'original_bbox': [x, y, w, h],
                            'detection_metadata': defect
                        })
        
        return raw_samples
    
    def _filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter samples by quality criteria, similar to MNIST's quality control.
        MNIST excluded ambiguous or poorly written digits.
        """
        filtered = []
        
        for sample in samples:
            # Quality criteria
            if sample['confidence'] < self.config.get('min_confidence', 0.3):
                continue
                
            if sample['area_px'] < self.config.get('min_area', 20):
                continue
                
            if sample['area_px'] > self.config.get('max_area', 10000):
                continue
                
            # Check image quality
            img = sample['image']
            if img.mean() < 5 or img.mean() > 250:  # Too dark or too bright
                continue
                
            # Check contrast
            if img.std() < 10:  # Too low contrast
                continue
                
            filtered.append(sample)
            
        return filtered
    
    def _balance_dataset(self, samples: List[Dict]) -> List[Dict]:
        """
        Balance dataset across classes, following MNIST's philosophy.
        MNIST had relatively balanced classes (5.4k-6.7k per digit).
        """
        # Group by defect type
        samples_by_type = defaultdict(list)
        for sample in samples:
            samples_by_type[sample['defect_type']].append(sample)
        
        # Find target count (MNIST used ~6000 per class)
        counts = [len(samples) for samples in samples_by_type.values()]
        
        # Use median as target (more robust than mean)
        target_count = int(np.median(counts))
        
        # Apply MNIST's mixed difficulty strategy
        balanced = []
        for defect_type, type_samples in samples_by_type.items():
            # Sort by severity to ensure mix of difficulties
            type_samples.sort(key=lambda x: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE'].index(x.get('severity', 'LOW')))
            
            if len(type_samples) > target_count:
                # Downsample, but keep distribution of severities
                indices = np.linspace(0, len(type_samples)-1, target_count, dtype=int)
                balanced.extend([type_samples[i] for i in indices])
            else:
                # Keep all samples for underrepresented classes
                balanced.extend(type_samples)
                
        random.shuffle(balanced)
        return balanced
    
    def _preprocess_samples(self, samples: List[Dict]) -> List[DefectSample]:
        """
        Apply MNIST-style preprocessing to standardize all images.
        This is the core of MNIST's approach - radical standardization.
        """
        processed = []
        
        for sample in samples:
            # Step 1: Convert to grayscale (MNIST used grayscale)
            if len(sample['image'].shape) == 3:
                gray = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2GRAY)
            else:
                gray = sample['image'].copy()
            
            # Step 2: Find defect boundaries (like MNIST's digit isolation)
            processed_img, params = self._isolate_and_normalize_defect(gray)
            
            # Step 3: Center by center of mass (MNIST's centering approach)
            centered_img = self._center_by_mass(processed_img)
            
            # Step 4: Add padding (MNIST's 2-pixel border)
            padded_img = np.pad(centered_img, self.padding, mode='constant', constant_values=255)
            
            # Step 5: Normalize pixel values
            normalized_img = self._normalize_pixels(padded_img)
            
            # Create DefectSample object
            defect_sample = DefectSample(
                image_data=normalized_img,
                defect_type=sample['defect_type'],
                severity=sample['severity'],
                source_info=sample,
                sample_id=self._generate_sample_id(sample),
                preprocessing_params=params
            )
            
            processed.append(defect_sample)
            
            # Update metadata
            self.metadata['class_distribution'][sample['defect_type']] += 1
            self.metadata['severity_distribution'][sample['severity']] += 1
            
        return processed
    
    def _isolate_and_normalize_defect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Isolate defect and normalize size, following MNIST's bounding box approach.
        MNIST fit digits into 20x20 preserving aspect ratio, then placed in 28x28.
        """
        # Find defect region using thresholding
        # Invert if necessary (ensure defect is darker than background)
        if np.mean(image) < 128:
            image = 255 - image
            
        # Adaptive threshold to find defect
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No clear defect found, use whole image
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        else:
            # Find bounding box of all contours (like MNIST's digit boundary)
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            
        # Extract defect region
        defect_region = image[y:y+h, x:x+w]
        
        # Calculate scale to fit in target size while preserving aspect ratio
        # We use larger size than MNIST (64x64 vs 28x28) for more detail
        target_size = self.standard_size[0] - 2 * self.padding  # Account for padding
        
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with high-quality interpolation (MNIST used linear)
        resized = cv2.resize(defect_region, (new_w, new_h), interpolation=self.interpolation)
        
        # Place in center of standard-sized image
        output = np.full((target_size, target_size), 255, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        params = {
            'original_bbox': [x, y, w, h],
            'scale_factor': scale,
            'final_size': [new_w, new_h],
            'offset': [x_offset, y_offset]
        }
        
        return output, params
    
    def _center_by_mass(self, image: np.ndarray) -> np.ndarray:
        """
        Center image by center of mass, exactly like MNIST.
        This ensures consistent positioning regardless of defect location.
        """
        # Calculate center of mass
        # Invert intensities so defect pixels have high weight
        weights = 255 - image
        
        # Calculate moments
        M = cv2.moments(weights)
        
        if M["m00"] == 0:
            return image  # No mass found, return as is
            
        # Center of mass
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate shift to center
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        
        shift_x = center_x - cx
        shift_y = center_y - cy
        
        # Apply translation with border replication
        M_translate = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered = cv2.warpAffine(image, M_translate, (image.shape[1], image.shape[0]),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        
        return centered
    
    def _normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values following MNIST's approach.
        MNIST normalized to [0,1] with background=0, foreground=1.
        """
        # Ensure defects are dark on light background
        if np.mean(image) < 128:
            image = 255 - image
            
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Invert so background=0, defect=1 (like MNIST)
        normalized = 1.0 - normalized
        
        # Apply slight smoothing to reduce noise (anti-aliasing effect)
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0.5)
        
        return normalized
    
    def _create_train_test_split(self, samples: List[DefectSample]) -> Tuple[List[DefectSample], List[DefectSample]]:
        """
        Create train/test split following MNIST's principles:
        - 60,000 training / 10,000 test (6:1 ratio)
        - Mixed difficulty in both sets
        - Stratified by class
        """
        # Group by defect type and severity for stratification
        grouped = defaultdict(list)
        for sample in samples:
            key = (sample.defect_type, sample.severity)
            grouped[key].append(sample)
        
        train_samples = []
        test_samples = []
        
        # MNIST used 6:1 train:test ratio
        test_ratio = 1/7  # Results in 6:1 split
        
        for (defect_type, severity), group_samples in grouped.items():
            # Ensure at least one sample in test if group has enough samples
            if len(group_samples) >= 2:
                n_test = max(1, int(len(group_samples) * test_ratio))
                n_train = len(group_samples) - n_test
                
                # Random split within group
                indices = np.random.permutation(len(group_samples))
                train_idx = indices[:n_train]
                test_idx = indices[n_train:]
                
                train_samples.extend([group_samples[i] for i in train_idx])
                test_samples.extend([group_samples[i] for i in test_idx])
            else:
                # Too few samples, add to training
                train_samples.extend(group_samples)
        
        # Shuffle to ensure random order
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        
        return train_samples, test_samples
    
    def _augment_training_data(self, train_samples: List[DefectSample]) -> List[DefectSample]:
        """
        Apply MNIST's data augmentation techniques adapted for defects.
        MNIST's elastic distortions were key to achieving low error rates.
        """
        augmented = train_samples.copy()
        
        # MNIST generated additional samples through augmentation
        augmentation_factor = self.config.get('augmentation_factor', 2)
        
        for _ in range(augmentation_factor - 1):
            for sample in train_samples:
                # Apply random augmentation
                aug_image = self._apply_augmentation(sample.image_data.copy())
                
                # Create new sample
                aug_sample = DefectSample(
                    image_data=aug_image,
                    defect_type=sample.defect_type,
                    severity=sample.severity,
                    source_info=sample.source_info.copy(),
                    sample_id=self._generate_sample_id(sample.source_info) + f"_aug{len(augmented)}",
                    preprocessing_params=sample.preprocessing_params.copy(),
                    augmentation_applied=sample.augmentation_applied + ['random']
                )
                
                augmented.append(aug_sample)
        
        return augmented
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply MNIST-style augmentations adapted for defect images.
        """
        # Choose augmentation type
        aug_type = random.choice(['elastic', 'affine', 'noise', 'intensity'])
        
        if aug_type == 'elastic':
            # MNIST's elastic distortion - their secret weapon
            return self._elastic_distortion(image)
        elif aug_type == 'affine':
            # Rotation, scale, shear
            return self._affine_transform(image)
        elif aug_type == 'noise':
            # Add noise
            return self._add_noise(image)
        else:
            # Intensity variations
            return self._intensity_variation(image)
    
    def _elastic_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        MNIST's elastic distortion algorithm, exactly as described in the paper.
        This was crucial for their success.
        """
        # Parameters from MNIST paper
        alpha = 8.0  # Displacement magnitude
        sigma = 4.0  # Gaussian filter sigma
        
        shape = image.shape
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, shape) * alpha
        dy = np.random.uniform(-1, 1, shape) * alpha
        
        # Smooth with Gaussian filter
        dx = gaussian_filter(dx, sigma)
        dy = gaussian_filter(dy, sigma)
        
        # Create coordinate arrays
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Apply displacement
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Map coordinates
        distorted = map_coordinates(image, indices, order=1, mode='reflect')
        
        return distorted.reshape(shape)
    
    def _affine_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply affine transformations like MNIST"""
        # Random parameters within MNIST's ranges
        angle = random.uniform(-15, 15)  # degrees
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-0.15, 0.15)  # radians
        
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Rotation matrix
        M_rotate = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Add shear
        M_shear = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
        
        # Combine transformations
        M_combined = M_rotate  # For simplicity, just rotation+scale
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M_combined, (image.shape[1], image.shape[0]),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return transformed
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise following MNIST's approach"""
        # Gaussian noise
        sigma = 0.01  # Small noise
        noise = np.random.normal(0, sigma, image.shape)
        
        # Add noise and clip
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def _intensity_variation(self, image: np.ndarray) -> np.ndarray:
        """Vary intensity to simulate imaging conditions"""
        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-0.1, 0.1)  # Brightness
        
        adjusted = alpha * image + beta
        return np.clip(adjusted, 0, 1)
    
    def _save_dataset(self, train_samples: List[DefectSample], test_samples: List[DefectSample], output_dir: Path):
        """
        Save dataset in multiple formats, like MNIST's distribution.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in NumPy format (like MNIST's IDX format)
        self._save_numpy_format(train_samples, test_samples, output_dir)
        
        # Save in image folders (for easy inspection)
        self._save_image_folders(train_samples, test_samples, output_dir)
        
        # Save metadata
        self._save_metadata(output_dir)
        
        # Save as pickle for easy loading
        self._save_pickle_format(train_samples, test_samples, output_dir)
    
    def _save_numpy_format(self, train_samples: List[DefectSample], test_samples: List[DefectSample], output_dir: Path):
        """Save in NumPy arrays like MNIST"""
        # Training data
        train_images = np.array([s.image_data for s in train_samples])
        train_labels = np.array([self.defect_classes.index(s.defect_type) for s in train_samples])
        train_severity = np.array([self.severity_levels.index(s.severity) for s in train_samples])
        
        np.save(output_dir / 'train_images.npy', train_images)
        np.save(output_dir / 'train_labels.npy', train_labels)
        np.save(output_dir / 'train_severity.npy', train_severity)
        
        # Test data
        test_images = np.array([s.image_data for s in test_samples])
        test_labels = np.array([self.defect_classes.index(s.defect_type) for s in test_samples])
        test_severity = np.array([self.severity_levels.index(s.severity) for s in test_samples])
        
        np.save(output_dir / 'test_images.npy', test_images)
        np.save(output_dir / 'test_labels.npy', test_labels)
        np.save(output_dir / 'test_severity.npy', test_severity)
        
        # Save label mappings
        label_map = {
            'defect_classes': self.defect_classes,
            'severity_levels': self.severity_levels
        }
        with open(output_dir / 'label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)
    
    def _save_image_folders(self, train_samples: List[DefectSample], test_samples: List[DefectSample], output_dir: Path):
        """Save as images in folders for visual inspection"""
        # Create directory structure
        for split in ['train', 'test']:
            for defect_class in self.defect_classes:
                (output_dir / split / defect_class).mkdir(parents=True, exist_ok=True)
        
        # Save training images
        for i, sample in enumerate(train_samples):
            filename = f"{sample.sample_id}.png"
            path = output_dir / 'train' / sample.defect_type / filename
            
            # Convert back to uint8 for saving
            img_uint8 = (sample.image_data * 255).astype(np.uint8)
            cv2.imwrite(str(path), img_uint8)
        
        # Save test images
        for i, sample in enumerate(test_samples):
            filename = f"{sample.sample_id}.png"
            path = output_dir / 'test' / sample.defect_type / filename
            
            img_uint8 = (sample.image_data * 255).astype(np.uint8)
            cv2.imwrite(str(path), img_uint8)
    
    def _save_pickle_format(self, train_samples: List[DefectSample], test_samples: List[DefectSample], output_dir: Path):
        """Save as pickle for easy Python loading"""
        dataset = {
            'train': train_samples,
            'test': test_samples,
            'metadata': self.metadata,
            'config': self.config
        }
        
        with open(output_dir / 'fiber_mnist_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    
    def _save_metadata(self, output_dir: Path):
        """Save comprehensive metadata about the dataset"""
        # Calculate statistics
        self.metadata['preprocessing_params'] = {
            'standard_size': list(self.standard_size),
            'padding': self.padding,
            'interpolation': 'INTER_CUBIC',
            'normalization': 'background=0, defect=1'
        }
        
        # Save metadata
        with open(output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_dataset_report(self, output_dir: Path):
        """Generate a comprehensive report about the dataset"""
        report_path = output_dir / 'dataset_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FIBER-MNIST DATASET REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Creation Date: {self.metadata['creation_date']}\n")
            f.write(f"Version: {self.metadata['version']}\n\n")
            
            f.write("DATASET STRUCTURE\n")
            f.write("-"*30 + "\n")
            f.write(f"Image Size: {self.standard_size[0]}x{self.standard_size[1]}\n")
            f.write(f"Number of Classes: {len(self.defect_classes)}\n")
            f.write(f"Classes: {', '.join(self.defect_classes)}\n\n")
            
            f.write("CLASS DISTRIBUTION\n")
            f.write("-"*30 + "\n")
            for defect_type, count in sorted(self.metadata['class_distribution'].items()):
                f.write(f"{defect_type:15s}: {count:5d}\n")
            
            f.write("\nSEVERITY DISTRIBUTION\n")
            f.write("-"*30 + "\n")
            for severity, count in sorted(self.metadata['severity_distribution'].items()):
                f.write(f"{severity:15s}: {count:5d}\n")
            
            f.write("\nPREPROCESSING PARAMETERS\n")
            f.write("-"*30 + "\n")
            for param, value in self.metadata['preprocessing_params'].items():
                f.write(f"{param}: {value}\n")
    
    def _generate_sample_id(self, sample_info: Dict) -> str:
        """Generate unique ID for each sample"""
        # Create hash from sample properties
        hash_input = f"{sample_info.get('source_file', '')}_{sample_info.get('original_bbox', '')}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def visualize_dataset_samples(dataset_path: Path, num_samples: int = 25):
    """
    Visualize samples from the created dataset, like MNIST visualization.
    """
    # Load dataset
    with open(dataset_path / 'fiber_mnist_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    train_samples = dataset['train']
    
    # Create grid visualization
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    # Randomly sample
    indices = np.random.choice(len(train_samples), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = train_samples[idx]
        axes[i].imshow(sample.image_data, cmap='gray')
        axes[i].set_title(f"{sample.defect_type}\n{sample.severity}", fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Fiber-MNIST Dataset Samples', fontsize=16)
    plt.savefig(dataset_path / 'dataset_visualization.png', dpi=150)
    plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("FIBER-MNIST DATASET CREATOR")
    print("Applying MNIST's data preparation methodology")
    print("="*60)
    
    # Configuration
    config = {
        'standard_size': [64, 64],  # Larger than MNIST for defect detail
        'padding': 4,
        'min_confidence': 0.3,
        'min_area': 20,
        'max_area': 10000,
        'augmentation_factor': 3,  # Generate 3x training data
        'test_split_ratio': 0.15
    }
    
    # Save config
    config_path = 'fiber_mnist_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get paths from user
    results_dir = input("Enter path to your pipeline results directory: ").strip()
    output_dir = input("Enter path for output dataset directory: ").strip()
    
    # Create dataset
    creator = FiberMNISTCreator(config_path)
    creator.create_dataset_from_detection_results(
        Path(results_dir),
        Path(output_dir)
    )
    
    # Visualize samples
    print("\nGenerating dataset visualization...")
    visualize_dataset_samples(Path(output_dir))
    
    print("\nDataset creation complete!")
    print(f"Dataset saved to: {output_dir}")
    print("\nYou can now use this dataset to train classifiers using standard")
    print("machine learning frameworks, just like MNIST!")


if __name__ == "__main__":
    main()