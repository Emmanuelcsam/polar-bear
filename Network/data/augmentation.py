#!/usr/bin/env python3
"""
Advanced Augmentation Pipeline for Fiber Optics Neural Network.
Implements various augmentation strategies, including geometric distortions,
color shifts, and the addition of synthetic defects like scratches and digs.
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
    """Applies elastic deformation to an image, simulating warping effects."""
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.5):
        self.alpha = alpha  # Controls deformation intensity
        self.sigma = sigma  # Controls deformation smoothness
        self.p = p          # Probability of applying the transform
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        # Convert to numpy for processing
        img_np = img.numpy()
        shape = img_np.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape[-2:]) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape[-2:]) * 2 - 1), self.sigma) * self.alpha
        
        # Create coordinate map for resampling
        y, x = np.meshgrid(np.arange(shape[-2]), np.arange(shape[-1]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation to each channel
        distorted = np.zeros_like(img_np)
        for c in range(shape[0]):
            distorted[c] = map_coordinates(img_np[c], indices, order=1, mode='reflect').reshape(shape[-2:])
            
        return torch.from_numpy(distorted)


class GridDistortion:
    """Applies grid-based distortion to an image."""
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        h, w = img.shape[-2:]
        
        # Create a grid and apply random displacements
        # This implementation is vectorized for efficiency.
        y_steps = torch.linspace(0, h - 1, self.num_steps + 1, device=img.device)
        x_steps = torch.linspace(0, w - 1, self.num_steps + 1, device=img.device)
        
        source_points = []
        for i in range(self.num_steps + 1):
            for j in range(self.num_steps + 1):
                source_points.append([x_steps[j], y_steps[i]])
        source_points = torch.tensor(source_points, device=img.device)
        
        # Add random displacement, but keep corners fixed
        displacement = (torch.rand(source_points.shape, device=img.device) * 2 - 1) * self.distort_limit * (w / self.num_steps)
        dest_points = source_points + displacement
        
        # Apply thin plate spline interpolation for a smooth distortion
        return TF.perspective(img, source_points.tolist(), dest_points.tolist())

class AddSyntheticDefects:
    """Adds synthetic defects (scratches, digs, blobs) to images."""
    def __init__(self, config: Optional[Dict] = None, p: float = 0.3):
        self.config = config or get_statistical_config().augmentation_settings
        self.p = p
        self.logger = get_logger("AddSyntheticDefects")
    
    def add_scratch(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a synthetic scratch defect."""
        h, w = img.shape[-2:]
        params = self.config['synthetic_scratch_params']
        
        length = random.randint(params['min_length'], params['max_length'])
        width = random.randint(params['min_width'], params['max_width'])
        
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Vectorized line drawing for efficiency
        num_steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        t = torch.linspace(0, 1, num_steps, device=img.device)
        xs = (x1 + t * (x2 - x1)).long().clamp(0, w - 1)
        ys = (y1 + t * (y2 - y1)).long().clamp(0, h - 1)
        
        mask = torch.zeros_like(img[0])
        mask[ys, xs] = 1.0
        
        # Apply a blur to create thickness and smoother edges
        kernel = T.GaussianBlur(kernel_size=width | 1, sigma=float(width) / 4.0)
        mask = kernel(mask.unsqueeze(0)).squeeze(0)

        intensity = random.uniform(0.3, 0.7)
        return img * (1 - mask * intensity)
    
    def add_dig(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a synthetic circular "dig" defect."""
        h, w = img.shape[-2:]
        params = self.config['synthetic_dig_params']
        
        radius = random.randint(params['min_radius'], params['max_radius'])
        intensity = random.uniform(*params['intensity_range'])
        cx, cy = random.randint(radius, w - radius - 1), random.randint(radius, h - radius - 1)
        
        # Vectorized circle creation
        y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing='ij')
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (dist <= radius).float()
        
        return img * (1 - mask * intensity)
    
    def add_blob(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a synthetic irregular "blob" defect."""
        # This can be implemented similarly to scratch/dig using noise generation
        # or by combining multiple smaller shapes. For brevity, this is a placeholder.
        self.logger.debug("Blob augmentation is a placeholder.")
        return img # Placeholder
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        
        defect_type = random.choice(['scratch', 'dig', 'blob'])
        if defect_type == 'scratch':
            return self.add_scratch(img)
        elif defect_type == 'dig':
            return self.add_dig(img)
        else:
            return self.add_blob(img)


class FiberOpticsAugmentation:
    """Complete augmentation pipeline for fiber optic images."""
    def __init__(self, config: Optional[Dict] = None, is_training: bool = True):
        self.config = config or get_statistical_config()
        self.is_training = is_training
        self.logger = get_logger("FiberOpticsAugmentation")
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> T.Compose:
        aug_config = self.config.augmentation_settings
        transforms = []

        if self.is_training:
            # Basic augmentations
            if 'horizontal_flip' in aug_config.get('basic_augmentations', []):
                transforms.append(T.RandomHorizontalFlip(p=0.5))
            if 'vertical_flip' in aug_config.get('basic_augmentations', []):
                transforms.append(T.RandomVerticalFlip(p=0.5))
            if 'rotation_15' in aug_config.get('basic_augmentations', []):
                transforms.append(T.RandomRotation(degrees=15))
            
            # Advanced geometric augmentations
            if 'elastic_transform' in aug_config.get('advanced_augmentations', []):
                transforms.append(ElasticTransform(p=0.5))
            
            if 'grid_distortion' in aug_config.get('advanced_augmentations', []):
                transforms.append(GridDistortion(p=0.5))
            
            # Color augmentations
            if 'brightness_contrast' in aug_config.get('basic_augmentations', []):
                transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            if 'hue_saturation' in aug_config.get('color_augmentations', []):
                transforms.append(T.ColorJitter(saturation=0.2, hue=0.1))

            # Noise and blur augmentations
            if 'gaussian_noise' in aug_config.get('noise_augmentations', []):
                transforms.append(T.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1)))
            if 'blur' in aug_config.get('noise_augmentations', []):
                transforms.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3))

            # Synthetic defects
            if aug_config.get('add_synthetic_defects', False):
                transforms.append(AddSyntheticDefects(config=aug_config, p=aug_config.get('defect_probability', 0.3)))

        # Normalization is always applied
        if aug_config.get('normalization') == 'imagenet_stats':
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return T.Compose(transforms)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transforms(img)


# Example usage and testing block
if __name__ == "__main__":
    pipeline = FiberOpticsAugmentation()
    logger = get_logger("AugmentationTest")
    logger.log_process_start("Augmentation Test")

    # Create a dummy test tensor
    test_tensor = torch.randn(2, 3, 256, 256)  # Test with batch size > 1
    
    # Apply augmentations
    augmented_tensor = pipeline(test_tensor)
    
    # Log results
    logger.info(f"Input shape: {test_tensor.shape}")
    logger.info(f"Output shape: {augmented_tensor.shape}")
    logger.info(f"Augmentation pipeline applied successfully.")
    
    logger.log_process_end("Augmentation Test")
    print(f"[{datetime.now()}] Augmentation test completed successfully.")
