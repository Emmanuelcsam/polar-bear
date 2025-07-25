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
        
        # Add batch dimension if it's not there
        is_batched = img.dim() == 4
        if not is_batched:
            img = img.unsqueeze(0)

        # Convert to numpy for processing, but keep device info
        device = img.device
        img_np = img.cpu().numpy()
        shape = img_np.shape
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(*shape[-2:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape[-2:]) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        
        # Create coordinate map for resampling
        y, x = np.meshgrid(np.arange(shape[-2]), np.arange(shape[-1]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation to each channel of each image in the batch
        distorted_batch = np.zeros_like(img_np)
        for i in range(shape[0]): # Iterate over batch
            for c in range(shape[1]): # Iterate over channels
                distorted_batch[i, c] = map_coordinates(img_np[i, c], indices, order=1, mode='reflect').reshape(shape[-2:])
        
        distorted_tensor = torch.from_numpy(distorted_batch).to(device)
        
        if not is_batched:
            distorted_tensor = distorted_tensor.squeeze(0)
            
        return distorted_tensor


class GridDistortion:
    """Applies grid-based distortion to an image."""
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, p: float = 0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        # Add batch dimension if it's not there for grid_sample
        is_batched = img.dim() == 4
        if not is_batched:
            img = img.unsqueeze(0)

        b, c, h, w = img.shape
        
        # Create a normalized grid (-1 to 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=img.device), 
                             torch.linspace(-1, 1, w, device=img.device), 
                             indexing='ij')
        
        # Create random displacement field
        # Use a smaller grid and interpolate to avoid memory issues
        grid_h, grid_w = self.num_steps + 1, self.num_steps + 1
        displace_y = torch.rand(grid_h, grid_w, device=img.device) * 2 - 1
        displace_x = torch.rand(grid_h, grid_w, device=img.device) * 2 - 1
        
        # Scale displacements
        displace_y *= self.distort_limit
        displace_x *= self.distort_limit
        
        # Keep borders fixed
        displace_y[0, :] = displace_y[-1, :] = displace_y[:, 0] = displace_y[:, -1] = 0
        displace_x[0, :] = displace_x[-1, :] = displace_x[:, 0] = displace_x[:, -1] = 0
        
        # Interpolate to full image size
        displace_y = F.interpolate(displace_y.unsqueeze(0).unsqueeze(0), 
                                  size=(h, w), mode='bilinear', align_corners=False).squeeze()
        displace_x = F.interpolate(displace_x.unsqueeze(0).unsqueeze(0), 
                                  size=(h, w), mode='bilinear', align_corners=False).squeeze()
        
        # Apply displacements to the grid
        y = y + displace_y
        x = x + displace_x
        
        # Stack into flow field (N, H, W, 2) format
        flow_field = torch.stack((y, x), dim=-1).unsqueeze(0)
        
        # Expand to match batch size
        flow_field = flow_field.expand(b, h, w, 2)
        
        # Apply the distortion
        distorted_img = F.grid_sample(img, flow_field, mode='bilinear', padding_mode='border', align_corners=False)
        
        if not is_batched:
            distorted_img = distorted_img.squeeze(0)

        return distorted_img

class AddSyntheticDefects:
    """Adds synthetic defects (scratches, digs, blobs) to images."""
    def __init__(self, config: Optional[Dict] = None, p: float = 0.3):
        stat_config = get_statistical_config()
        self.config = config or stat_config.get('augmentation_settings', {})
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
        # Ensure kernel size is odd
        kernel_size = width if width % 2 == 1 else width + 1
        kernel = T.GaussianBlur(kernel_size=kernel_size, sigma=float(width) / 4.0)
        mask = kernel(mask.unsqueeze(0)).squeeze(0)

        intensity = random.uniform(0.3, 0.7)
        return img * (1 - mask * intensity)
    
    def add_dig(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a synthetic circular "dig" defect."""
        h, w = img.shape[-2:]
        params = self.config['synthetic_dig_params']
        
        radius = random.randint(params['min_radius'], params['max_radius'])
        intensity = random.uniform(*params['intensity_range'])
        
        # Ensure center is within bounds
        cx = random.randint(radius, w - radius - 1) if w > 2 * radius else w // 2
        cy = random.randint(radius, h - radius - 1) if h > 2 * radius else h // 2
        
        # Vectorized circle creation
        y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing='ij')
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = (dist <= radius).float()
        
        # Add a soft edge to the dig
        mask = T.GaussianBlur(kernel_size=3, sigma=1.0)(mask.unsqueeze(0)).squeeze(0)

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
        stat_config = get_statistical_config()
        self.config = config or stat_config
        self.is_training = is_training
        self.logger = get_logger("FiberOpticsAugmentation")
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> T.Compose:
        aug_config = self.config.get('augmentation_settings', {})
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
            # Note: Custom normalization stats might be better for fiber optic images
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        return T.Compose(transforms)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transforms(img)


# Example usage and testing block
if __name__ == "__main__":
    # This requires a dummy config setup to run standalone
    from unittest.mock import MagicMock
    get_statistical_config = MagicMock()
    get_statistical_config.return_value = {
        'augmentation_settings': {
            'basic_augmentations': ['horizontal_flip', 'vertical_flip', 'rotation_15', 'brightness_contrast'],
            'advanced_augmentations': ['elastic_transform', 'grid_distortion'],
            'color_augmentations': ['hue_saturation'],
            'noise_augmentations': ['gaussian_noise', 'blur'],
            'add_synthetic_defects': True,
            'defect_probability': 0.5,
            'synthetic_scratch_params': {'min_length': 20, 'max_length': 100, 'min_width': 1, 'max_width': 3},
            'synthetic_dig_params': {'min_radius': 2, 'max_radius': 8, 'intensity_range': (0.3, 0.8)},
            'normalization': 'imagenet_stats'
        }
    }

    pipeline = FiberOpticsAugmentation()
    logger = get_logger("AugmentationTest")
    logger.log_process_start("Augmentation Test")

    # Create a dummy test tensor
    test_tensor = torch.rand(2, 3, 256, 256)  # Test with batch size > 1, use rand for [0,1] range
    
    # Test individual components
    logger.info("Testing individual augmentation components...")
    
    # Test ElasticTransform
    elastic = ElasticTransform(p=1.0)  # Force application
    elastic_result = elastic(test_tensor)
    assert elastic_result.shape == test_tensor.shape, "ElasticTransform shape mismatch"
    logger.info("ElasticTransform test passed")
    
    # Test GridDistortion
    grid_dist = GridDistortion(p=1.0)  # Force application
    grid_result = grid_dist(test_tensor)
    assert grid_result.shape == test_tensor.shape, "GridDistortion shape mismatch"
    logger.info("GridDistortion test passed")
    
    # Test AddSyntheticDefects
    defects = AddSyntheticDefects(p=1.0)  # Force application
    defects_result = defects(test_tensor)
    assert defects_result.shape == test_tensor.shape, "AddSyntheticDefects shape mismatch"
    logger.info("AddSyntheticDefects test passed")
    
    # Apply full augmentation pipeline
    augmented_tensor = pipeline(test_tensor)
    
    # Log results
    logger.info(f"Input shape: {test_tensor.shape}")
    logger.info(f"Output shape: {augmented_tensor.shape}")
    assert test_tensor.shape == augmented_tensor.shape, "Output shape must match input shape"
    logger.info("Full augmentation pipeline test passed")
    
    logger.log_process_end("Augmentation Test")
    print(f"[{datetime.now()}] Augmentation test completed successfully.")