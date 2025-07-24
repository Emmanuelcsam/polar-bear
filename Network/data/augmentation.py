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
from scipy.ndimage import gaussian_filter, map_coordinates

from core.statistical_config import get_statistical_config
from core.logger import get_logger


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
        
        # Generate random distortions using torch for vectorization
        # FIX: Replaced nested Python loops with torch operations for efficiency. Original loops were slow for large images (e.g., 256x256) due to Python interpreter overhead. Now uses torch.rand for batched random generation, improving speed by orders of magnitude.
        # FIX: Added device=img.device to ensure dx/dy on same device as img (avoids RuntimeError on GPU).
        dx = (torch.rand(h, w, device=img.device) * 2 - 1) * self.distort_limit * step_x
        dy = (torch.rand(h, w, device=img.device) * 2 - 1) * self.distort_limit * step_y
        
        # Normalize coordinates
        # FIX: torch.arange on same device to avoid device mismatch.
        xx = torch.arange(w, device=img.device).expand(h, w) + dx
        yy = torch.arange(h, device=img.device).unsqueeze(1).expand(h, w) + dy
        
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
        
        # Create scratch mask using vectorized operations
        # FIX: Replaced nested loops with torch.linspace and broadcasting for vectorized line drawing. Original loops were inefficient for high-resolution images, causing significant slowdowns. This change uses GPU-accelerated operations for better performance.
        mask = torch.zeros(h, w, device=img.device)
        
        # Generate points along the line
        # FIX: Added max(..., 1) to avoid zero steps if x1==x2 and y1==y2 (avoids empty tensor and indexing error).
        num_steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        num_steps = max(num_steps, 1)  # Ensure at least one point
        t = torch.linspace(0, 1, num_steps, device=img.device)
        xs = (x1 + t * (x2 - x1)).long()
        ys = (y1 + t * (y2 - y1)).long()
        
        # Clip to bounds
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid]
        ys = ys[valid]
        
        # Create width around the line using meshgrid for offsets
        dx = torch.arange(-width//2, width//2 + 1, device=img.device)
        dy = torch.arange(-width//2, width//2 + 1, device=img.device)
        dx, dy = torch.meshgrid(dx, dy, indexing='ij')
        dx = dx.flatten()
        dy = dy.flatten()
        
        # Broadcast offsets to all points
        nxs = xs.unsqueeze(1) + dx.unsqueeze(0)
        nys = ys.unsqueeze(1) + dy.unsqueeze(0)
        
        # Clip and set mask
        valid_n = (nxs >= 0) & (nxs < w) & (nys >= 0) & (nys < h)
        # FIX: Use torch.where to get indices of valid, then flatten nxs/nys only for valid (avoids shape mismatch in flattening).
        valid_indices = torch.nonzero(valid_n, as_tuple=False)
        nxs = nxs[valid_indices[:, 0], valid_indices[:, 1]]
        nys = nys[valid_indices[:, 0], valid_indices[:, 1]]
        
        mask[nys.long(), nxs.long()] = 1.0
        
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
        
        # Create circular mask using vectorized operations
        # FIX: Used torch.meshgrid for vectorized distance calculation instead of loops. Improves performance for large images by avoiding Python loops.
        y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing='ij')
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
        
        # Create irregular blob using multiple overlapping circles with vectorized operations
        # FIX: Vectorized circle addition using broadcasting instead of loops. Reduces computation time for multiple circles.
        mask = torch.zeros(h, w, device=img.device)
        num_circles = random.randint(3, 7)
        
        # Generate random offsets and radii
        dxs = (torch.rand(num_circles, device=img.device) * 2 - 1) * irregularity * np.sqrt(area)
        dys = (torch.rand(num_circles, device=img.device) * 2 - 1) * irregularity * np.sqrt(area)
        rs = torch.rand(num_circles, device=img.device) * 0.4 + 0.3  # Between 0.3 and 0.7
        rs *= np.sqrt(area / np.pi)
        
        # Meshgrid for distances
        y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing='ij')
        
        # Broadcast to compute distances for all circles
        dists = torch.sqrt((x.unsqueeze(0) - (cx + dxs).unsqueeze(1).unsqueeze(2)) ** 2 +
                           (y.unsqueeze(0) - (cy + dys).unsqueeze(1).unsqueeze(2)) ** 2)
        
        # Maximum over circles
        circle_masks = (dists <= rs.unsqueeze(1).unsqueeze(2)).float()
        mask = circle_masks.max(dim=0)[0]
        
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
                # OpticalDistortion not implemented, using GridDistortion as alternative
                transforms.append(GridDistortion(num_steps=5, distort_limit=0.05, p=0.5))
                self.logger.info("Using GridDistortion as alternative to OpticalDistortion")
            
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
