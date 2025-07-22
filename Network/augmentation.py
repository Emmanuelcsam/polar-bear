#!/usr/bin/env python3
"""
Advanced Augmentation Pipeline for Fiber Optics Neural Network
Implements advanced augmentation strategies from statistical analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import random
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

from statistical_config import get_statistical_config
from logger import get_logger


class ElasticTransform:
    """Elastic deformation of images"""
    
    def __init__(self, alpha: float = 50, sigma: float = 5, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        # Convert to numpy for processing
        img_np = img.numpy()
        shape = img_np.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.random(shape[-2:]) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.random(shape[-2:]) * 2 - 1), self.sigma) * self.alpha
        
        # Generate meshgrid
        x, y = np.meshgrid(np.arange(shape[-1]), np.arange(shape[-2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation
        if len(shape) == 3:  # CHW format
            distorted = np.zeros_like(img_np)
            for c in range(shape[0]):
                distorted[c] = map_coordinates(img_np[c], indices, order=1, mode='reflect').reshape(shape[-2:])
        else:  # HW format
            distorted = map_coordinates(img_np, indices, order=1, mode='reflect').reshape(shape)
        
        return torch.from_numpy(distorted)


class GridDistortion:
    """Grid distortion augmentation"""
    
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        h, w = img.shape[-2:]
        
        # Create grid
        step_x = w // self.num_steps
        step_y = h // self.num_steps
        
        # Generate random distortions
        xx = torch.zeros(h, w)
        yy = torch.zeros(h, w)
        
        for i in range(h):
            for j in range(w):
                dx = random.uniform(-self.distort_limit, self.distort_limit) * step_x
                dy = random.uniform(-self.distort_limit, self.distort_limit) * step_y
                xx[i, j] = j + dx
                yy[i, j] = i + dy
        
        # Normalize coordinates
        xx = (xx / (w - 1)) * 2 - 1
        yy = (yy / (h - 1)) * 2 - 1
        
        # Create sampling grid
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        
        # Apply grid sampling
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
            output = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            return output.squeeze(0)
        else:
            img = img.unsqueeze(0).unsqueeze(0)
            output = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            return output.squeeze(0).squeeze(0)


class OpticalDistortion:
    """Optical distortion (barrel/pincushion)"""
    
    def __init__(self, distort_limit: float = 0.05, shift_limit: float = 0.05, p: float = 0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        h, w = img.shape[-2:]
        
        # Camera matrix
        fx = w
        fy = h
        cx = w * 0.5 + random.uniform(-self.shift_limit, self.shift_limit) * w
        cy = h * 0.5 + random.uniform(-self.shift_limit, self.shift_limit) * h
        
        # Distortion coefficients
        k1 = random.uniform(-self.distort_limit, self.distort_limit)
        k2 = random.uniform(-self.distort_limit, self.distort_limit)
        p1 = random.uniform(-self.distort_limit, self.distort_limit)
        p2 = random.uniform(-self.distort_limit, self.distort_limit)
        
        # Create meshgrid
        x, y = torch.meshgrid(torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32))
        x = (x - cx) / fx
        y = (y - cy) / fy
        
        # Apply distortion
        r2 = x * x + y * y
        r4 = r2 * r2
        
        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r4
        
        # Tangential distortion
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        
        # Convert back to pixel coordinates
        x_distorted = x_distorted * fx + cx
        y_distorted = y_distorted * fy + cy
        
        # Normalize for grid_sample
        x_distorted = (x_distorted / (w - 1)) * 2 - 1
        y_distorted = (y_distorted / (h - 1)) * 2 - 1
        
        # Create sampling grid
        grid = torch.stack([x_distorted.T, y_distorted.T], dim=-1).unsqueeze(0)
        
        # Apply distortion
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
            output = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            return output.squeeze(0)
        else:
            img = img.unsqueeze(0).unsqueeze(0)
            output = F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            return output.squeeze(0).squeeze(0)


class AddSyntheticDefects:
    """Add synthetic defects (scratches, digs, blobs) to images"""
    
    def __init__(self, config: Optional[Dict] = None, p: float = 0.3):
        self.config = config or get_statistical_config().augmentation_settings
        self.p = p
        self.logger = get_logger("AddSyntheticDefects")
    
    def add_scratch(self, img: torch.Tensor) -> torch.Tensor:
        """Add synthetic scratch defect"""
        h, w = img.shape[-2:]
        scratch_params = self.config['synthetic_scratch_params']
        
        # Random scratch parameters
        length = random.randint(scratch_params['min_length'], scratch_params['max_length'])
        width = random.randint(scratch_params['min_width'], scratch_params['max_width'])
        
        # Random position and angle
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        angle = random.uniform(0, 2 * np.pi)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Create scratch mask
        mask = torch.zeros(h, w)
        
        # Draw line (simplified - in practice use cv2 for better quality)
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps > 0:
            for i in range(steps):
                x = int(x1 + (x2 - x1) * i / steps)
                y = int(y1 + (y2 - y1) * i / steps)
                if 0 <= x < w and 0 <= y < h:
                    for dx in range(-width//2, width//2 + 1):
                        for dy in range(-width//2, width//2 + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                mask[ny, nx] = 1.0
        
        # Apply scratch
        intensity = random.uniform(0.3, 0.7)
        if len(img.shape) == 3:
            mask = mask.unsqueeze(0).expand_as(img)
        
        return img * (1 - mask * intensity)
    
    def add_dig(self, img: torch.Tensor) -> torch.Tensor:
        """Add synthetic dig defect"""
        h, w = img.shape[-2:]
        dig_params = self.config['synthetic_dig_params']
        
        # Random dig parameters
        radius = random.randint(dig_params['min_radius'], dig_params['max_radius'])
        intensity = random.uniform(*dig_params['intensity_range'])
        
        # Random position
        cx = random.randint(radius, w - radius - 1)
        cy = random.randint(radius, h - radius - 1)
        
        # Create circular mask
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (dist <= radius).float()
        
        # Smooth edges
        mask = F.gaussian_blur(mask.unsqueeze(0).unsqueeze(0), kernel_size=5, sigma=1.0)
        mask = mask.squeeze(0).squeeze(0)
        
        # Apply dig
        if len(img.shape) == 3:
            mask = mask.unsqueeze(0).expand_as(img)
        
        return img * (1 - mask * intensity)
    
    def add_blob(self, img: torch.Tensor) -> torch.Tensor:
        """Add synthetic blob defect"""
        h, w = img.shape[-2:]
        blob_params = self.config['synthetic_blob_params']
        
        # Random blob parameters
        area = random.randint(blob_params['min_area'], blob_params['max_area'])
        irregularity = blob_params['irregularity']
        
        # Random center
        cx = random.randint(int(np.sqrt(area)), w - int(np.sqrt(area)) - 1)
        cy = random.randint(int(np.sqrt(area)), h - int(np.sqrt(area)) - 1)
        
        # Create irregular blob using multiple overlapping circles
        mask = torch.zeros(h, w)
        num_circles = random.randint(3, 7)
        
        for _ in range(num_circles):
            # Random offset from center
            dx = random.uniform(-irregularity * np.sqrt(area), irregularity * np.sqrt(area))
            dy = random.uniform(-irregularity * np.sqrt(area), irregularity * np.sqrt(area))
            
            # Random radius
            r = random.uniform(0.3, 0.7) * np.sqrt(area / np.pi)
            
            # Add circle to mask
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
            dist = torch.sqrt((x - (cx + dx)) ** 2 + (y - (cy + dy)) ** 2)
            mask = torch.maximum(mask, (dist <= r).float())
        
        # Smooth blob
        mask = F.gaussian_blur(mask.unsqueeze(0).unsqueeze(0), kernel_size=7, sigma=2.0)
        mask = mask.squeeze(0).squeeze(0)
        
        # Apply blob
        intensity = random.uniform(0.3, 0.6)
        if len(img.shape) == 3:
            mask = mask.unsqueeze(0).expand_as(img)
        
        return img * (1 - mask * intensity) + mask * torch.randn_like(img) * 0.1
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        # Randomly select defect type
        defect_type = random.choice(['scratch', 'dig', 'blob'])
        
        if defect_type == 'scratch':
            return self.add_scratch(img)
        elif defect_type == 'dig':
            return self.add_dig(img)
        else:
            return self.add_blob(img)


class FiberOpticsAugmentation:
    """
    Complete advanced augmentation pipeline for fiber optics images
    Implements all augmentation strategies from neural_network_config.json
    """
    
    def __init__(self, config: Optional[Dict] = None, is_training: bool = True):
        print(f"[{datetime.now()}] Initializing FiberOpticsAugmentation")
        
        self.config = config or get_statistical_config()
        self.is_training = is_training
        self.logger = get_logger("FiberOpticsAugmentation")
        
        # Build augmentation pipeline
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> T.Compose:
        """Build the complete augmentation pipeline"""
        aug_config = self.config.augmentation_settings
        transforms = []
        
        if self.is_training:
            # Basic augmentations
            if 'horizontal_flip' in aug_config['basic_augmentations']:
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            
            if 'vertical_flip' in aug_config['basic_augmentations']:
                transforms.append(T.RandomVerticalFlip(p=0.5))
            
            if 'rotation_15' in aug_config['basic_augmentations']:
                transforms.append(T.RandomRotation(degrees=15))
            
            if 'brightness_contrast' in aug_config['basic_augmentations']:
                transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            
            # Advanced augmentations
            if 'elastic_transform' in aug_config['advanced_augmentations']:
                transforms.append(ElasticTransform(alpha=50, sigma=5, p=0.5))
            
            if 'grid_distortion' in aug_config['advanced_augmentations']:
                transforms.append(GridDistortion(num_steps=5, distort_limit=0.3, p=0.5))
            
            if 'optical_distortion' in aug_config['advanced_augmentations']:
                transforms.append(OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5))
            
            # Color augmentations
            if any(aug in aug_config['color_augmentations'] for aug in ['hue_saturation', 'rgb_shift']):
                transforms.append(T.ColorJitter(saturation=0.2, hue=0.1))
            
            # Noise augmentations
            if 'gaussian_noise' in aug_config['noise_augmentations']:
                transforms.append(T.Lambda(lambda x: x + torch.randn_like(x) * 0.05))
            
            if 'blur' in aug_config['noise_augmentations']:
                transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3))
            
            # Synthetic defects
            if aug_config['add_synthetic_defects']:
                transforms.append(AddSyntheticDefects(config=aug_config, p=aug_config['defect_probability']))
        
        # Normalization (always applied)
        if aug_config['normalization'] == 'imagenet_stats':
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return T.Compose(transforms)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline"""
        return self.transforms(img)


class MixupAugmentation:
    """
    Mixup augmentation for fiber optics images
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch of images and labels
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B] or [B, num_classes]
            
        Returns:
            mixed_images: Mixed images
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Mixing coefficient
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Random shuffle indices
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return mixed data and both labels
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def get_augmentation_pipeline(is_training: bool = True, 
                            config: Optional[Dict] = None) -> FiberOpticsAugmentation:
    """
    Get the appropriate augmentation pipeline
    
    Args:
        is_training: Whether this is for training or validation
        config: Optional configuration dictionary
        
    Returns:
        FiberOpticsAugmentation instance
    """
    return FiberOpticsAugmentation(config=config, is_training=is_training)