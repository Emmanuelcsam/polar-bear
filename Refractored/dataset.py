# dataset.py
# Dataset and data loading utilities for fiber optic analysis

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import random
from typing import Dict, Any, Tuple
import logging

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    # Fallback transforms using basic torch operations
    import torchvision.transforms as transforms

class FiberOpticsDataset(Dataset):
    """
    Dataset class for loading fiber optic images and tensors.
    Supports both image files (.png, .jpg) and tensor files (.pt).
    """
    
    def __init__(self, config, mode='train'):
        self.config = config.data
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load file paths
        self._load_file_paths()
        
        # Create class mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.config.class_names)}
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        self.logger.info(f"Dataset initialized: {len(self)} samples in {mode} mode")

    def _load_file_paths(self):
        """Load and validate file paths from the data directory."""
        data_path = Path(self.config.path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Collect image and tensor files
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        tensor_extensions = ['*.pt', '*.pth']
        
        self.image_paths = []
        self.tensor_paths = []
        
        for ext in image_extensions:
            self.image_paths.extend(list(data_path.glob(f"**/{ext}")))
        
        for ext in tensor_extensions:
            self.tensor_paths.extend(list(data_path.glob(f"**/{ext}")))
        
        self.all_files = sorted(self.image_paths + self.tensor_paths)
        
        if not self.all_files:
            raise FileNotFoundError(f"No data files found in: {data_path}")
        
        self.logger.info(f"Found {len(self.image_paths)} image files and {len(self.tensor_paths)} tensor files")

    def _get_transforms(self):
        """Create data augmentation pipeline based on available libraries."""
        if ALBUMENTATIONS_AVAILABLE:
            return self._get_albumentations_transforms()
        else:
            return self._get_torchvision_transforms()

    def _get_albumentations_transforms(self):
        """Create Albumentations transform pipeline."""
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.OneOf([A.ElasticTransform(p=0.7), A.GridDistortion(p=0.5)], p=0.8),
                A.RandomBrightnessContrast(p=0.8),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def _get_torchvision_transforms(self):
        """Create torchvision transform pipeline as fallback."""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def _load_image_file(self, file_path: Path) -> np.ndarray:
        """Load image from file."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_tensor_file(self, file_path: Path) -> np.ndarray:
        """Load tensor from .pt file and convert to image array."""
        try:
            tensor_data = torch.load(file_path, map_location='cpu')
            
            if isinstance(tensor_data, dict):
                # Handle structured tensor files
                tensor = tensor_data.get('tensor', tensor_data.get('image', None))
                if tensor is None:
                    # Try to find any tensor in the dict
                    for key, value in tensor_data.items():
                        if isinstance(value, torch.Tensor) and value.dim() >= 3:
                            tensor = value
                            break
                if tensor is None:
                    raise ValueError("No valid tensor found in file")
            else:
                tensor = tensor_data
            
            # Ensure tensor is in correct format
            if tensor.dim() == 4:  # Batch dimension
                tensor = tensor.squeeze(0)
            
            if tensor.shape[0] == 3:  # CHW format
                tensor = tensor.permute(1, 2, 0)  # Convert to HWC
            
            # Convert to numpy and ensure proper range
            image = tensor.numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
            return image
            
        except Exception as e:
            raise ValueError(f"Error loading tensor file {file_path}: {e}")

    def _get_class_label(self, file_path: Path) -> int:
        """Extract class label from file path."""
        class_name = file_path.parent.name
        return self.class_to_idx.get(class_name, 0)

    def _get_reference_sample(self) -> np.ndarray:
        """Get a random reference sample for similarity comparison."""
        ref_idx = random.randint(0, len(self) - 1)
        ref_path = self.all_files[ref_idx]
        
        if ref_path.suffix == '.pt':
            return self._load_tensor_file(ref_path)
        else:
            return self._load_image_file(ref_path)

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        file_path = self.all_files[idx]
        
        try:
            # Load main image
            if file_path.suffix == '.pt':
                image = self._load_tensor_file(file_path)
            else:
                image = self._load_image_file(file_path)
            
            # Get class label
            label = self._get_class_label(file_path)
            
            # Apply transforms
            if ALBUMENTATIONS_AVAILABLE:
                augmented = self.transform(image=image)
                image_tensor = augmented['image']
            else:
                image_tensor = self.transform(image)
            
            # Get reference sample
            ref_image = self._get_reference_sample()
            if ALBUMENTATIONS_AVAILABLE:
                ref_augmented = self.transform(image=ref_image)
                ref_tensor = ref_augmented['image']
            else:
                ref_tensor = self.transform(ref_image)
            
            return {
                "image": image_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "reference": ref_tensor,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx} from {file_path}: {e}")
            # Return a fallback sample
            return {
                "image": torch.zeros(3, self.config.image_size, self.config.image_size),
                "label": torch.tensor(0, dtype=torch.long),
                "reference": torch.zeros(3, self.config.image_size, self.config.image_size),
                "file_path": str(file_path)
            }

def create_dataloaders(config, world_size=1, rank=0):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        world_size: Number of distributed processes
        rank: Current process rank
        
    Returns:
        Tuple of (train_loader, val_loader, train_sampler)
    """
    # Create datasets
    train_dataset = FiberOpticsDataset(config, mode='train')
    val_dataset = FiberOpticsDataset(config, mode='val')
    
    # Create distributed sampler if needed
    train_sampler = None
    if world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size // world_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler
