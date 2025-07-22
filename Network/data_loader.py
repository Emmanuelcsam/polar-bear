#!/usr/bin/env python3
"""
Data Loader module for Fiber Optics Neural Network
"I have a folder with a large database of .pt files of different images used as reference 
and the files are separated by folder name"
Handles loading and batching of tensorized data
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json
import random
import torchvision.transforms as T

from config_loader import get_config
from logger import get_logger
from tensor_processor import TensorProcessor

class FiberOpticsDataset(Dataset):
    """
    Dataset for fiber optic images
    "an image will be selected from a dataset folder (or multiple images for batch processing)"
    """
    
    def __init__(self, data_paths: List[Path], 
                 transform=None,
                 load_into_memory: bool = False,
                 max_samples_per_class: Optional[int] = None):
        print(f"[{datetime.now()}] Initializing FiberOpticsDataset")
        
        self.logger = get_logger("FiberOpticsDataset")
        self.config = get_config()
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("FiberOpticsDataset", 
                                  num_paths=len(data_paths),
                                  load_into_memory=load_into_memory)
        
        self.transform = transform
        self.load_into_memory = load_into_memory
        
        # Storage for samples
        self.samples = []
        self.labels = []
        self.metadata = []
        
        # Updated class mapping from data-implementation.py
        self.classes = ["19700101000222-scratch.jpg", "19700101000223-scratch.jpg", "19700101000237-scratch.jpg", "50", "50-cladding", 
                       "50_clean_20250705_0001.png", "50_clean_20250705_0003.jpg", "50_clean_20250705_0004.png", "50_clean_20250705_0005.png", 
                       "91", "91-cladding", "91-scratched", "cladding-batch-1", "cladding-batch-3", "cladding-batch-4", 
                       "cladding-batch-5", "cladding-features-batch-1", "core-batch-1", "core-batch-2", "core-batch-3", 
                       "core-batch-4", "core-batch-5", "core-batch-6", "core-batch-7", "core-batch-8", "dirty-image", 
                       "fc-50-clean-full-1.png", "fc-50-clean-full-2.png", "fc-50-clean-full-3.jpg", "fc-50-clean-full.jpg", 
                       "fc-50-clean-full.png", "ferrule-batch-1", "ferrule-batch-2", "ferrule-batch-3", "ferrule-batch-4", 
                       "large-core-batch", "scratch-library-bmp", "sma", "sma-clean", "visualizations"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # If loading into memory, store tensors
        self.tensors = {} if load_into_memory else None
        
        # Scan data paths and build dataset
        self._scan_data_paths(data_paths, max_samples_per_class)
        
        self.logger.info(f"Dataset initialized with {len(self.samples)} samples")
        print(f"[{datetime.now()}] FiberOpticsDataset initialized with {len(self.samples)} samples")
    
    def _scan_data_paths(self, data_paths: List[Path], max_samples_per_class: Optional[int]):
        """Scan data paths and collect samples"""
        self.logger.log_process_start("Scanning data paths")
        
        class_counts = {}
        
        for path in data_paths:
            if not path.exists():
                self.logger.warning(f"Path does not exist: {path}")
                continue
            
            # Find all .pt files
            pt_files = list(path.glob("*.pt"))
            self.logger.info(f"Found {len(pt_files)} .pt files in {path.name}")
            
            # Determine class label from folder structure
            class_label = self._determine_class_label(path)
            
            # Limit samples per class if specified
            if max_samples_per_class and class_label in class_counts:
                remaining = max_samples_per_class - class_counts[class_label]
                if remaining <= 0:
                    continue
                pt_files = pt_files[:remaining]
            
            for pt_file in pt_files:
                sample_info = {
                    'path': pt_file,
                    'class_name': path.name,
                    'class_label': class_label,
                    'has_anomaly': self._check_if_anomaly(path.name)
                }
                
                self.samples.append(pt_file)
                self.labels.append(class_label)
                self.metadata.append(sample_info)
                
                # Load into memory if requested
                if self.load_into_memory:
                    try:
                        tensor_data = self.tensor_processor.load_tensor_file(pt_file)
                        self.tensors[str(pt_file)] = tensor_data['tensor']
                    except Exception as e:
                        self.logger.warning(f"Failed to load {pt_file}: {e}")
                
                # Update class count
                class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        self.logger.info(f"Class distribution: {class_counts}")
        self.logger.log_process_end("Scanning data paths")
    
    def _determine_class_label(self, path: Path) -> int:
        """Determine class label from path"""
        folder_name = path.name
        
        # First check if folder name is in our class list
        if folder_name in self.class_to_idx:
            return self.class_to_idx[folder_name]
        
        # Check region categories as fallback
        for region_idx, (region, folders) in enumerate(self.config.REGION_CATEGORIES.items()):
            if folder_name in folders:
                return region_idx
        
        # Default to unknown class
        return len(self.classes)
    
    def _check_if_anomaly(self, folder_name: str) -> bool:
        """Check if folder contains anomaly data"""
        anomaly_keywords = ['defect', 'dirty', 'scratch', 'contamination', 'anomaly']
        return any(keyword in folder_name.lower() for keyword in anomaly_keywords)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get sample info
        sample_path = self.samples[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        # Load tensor
        if self.load_into_memory and str(sample_path) in self.tensors:
            tensor = self.tensors[str(sample_path)]
        else:
            try:
                tensor_data = self.tensor_processor.load_tensor_file(sample_path)
                tensor = tensor_data['tensor']
            except Exception as e:
                self.logger.error(f"Failed to load sample {idx}: {e}")
                # Return zeros tensor as fallback
                tensor = torch.zeros(3, 256, 256)
        
        # Ensure correct shape
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Ensure tensor is float and in correct range
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # Normalize to [0, 1] if needed
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        # Apply transforms if any
        if self.transform:
            tensor = self.transform(tensor)
        
        return {
            'image': tensor,
            'label': label,
            'has_anomaly': metadata['has_anomaly'],
            'metadata': metadata
        }

class FiberOpticsDataLoader:
    """
    Data loader manager for fiber optics dataset
    "or multiple images for batch processing or realtime processing"
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing FiberOpticsDataLoader")
        print(f"[{datetime.now()}] Previous script: trainer.py")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsDataLoader")
        
        self.logger.log_class_init("FiberOpticsDataLoader")
        
        # Collect all data paths
        self.data_paths = self._collect_data_paths()
        
        self.logger.info(f"Found {len(self.data_paths)} data folders")
        print(f"[{datetime.now()}] FiberOpticsDataLoader initialized")
    
    def _collect_data_paths(self) -> List[Path]:
        """Collect all relevant data paths"""
        paths = []
        
        # First check if reference folder exists
        reference_path = Path(self.config.TENSORIZED_DATA_PATH).parent / "reference"
        if reference_path.exists():
            # Scan all subdirectories in reference folder
            for folder_path in reference_path.iterdir():
                if folder_path.is_dir() and any(folder_path.glob("*.pt")):
                    paths.append(folder_path)
                    self.logger.debug(f"Found reference folder: {folder_path.name}")
        
        # Also collect paths for each region from config (fallback)
        for region, folders in self.config.REGION_CATEGORIES.items():
            for folder_name in folders:
                folder_path = self.config.TENSORIZED_DATA_PATH / folder_name
                if folder_path.exists():
                    paths.append(folder_path)
                else:
                    self.logger.debug(f"Folder not found: {folder_path}")
        
        return paths
    
    def get_data_loaders(self, 
                        train_ratio: float = 0.8,
                        batch_size: Optional[int] = None,
                        num_workers: Optional[int] = None,
                        use_weighted_sampling: bool = True,
                        use_augmentation: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation data loaders
        "the program will run entirely in hpc"
        """
        self.logger.log_function_entry("get_data_loaders", 
                                     train_ratio=train_ratio,
                                     use_weighted_sampling=use_weighted_sampling)
        
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        if num_workers is None:
            num_workers = self.config.NUM_WORKERS
        
        # Data augmentation based on recommendations from data-implementation.py
        train_transform = None
        val_transform = None
        
        if use_augmentation:
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Create full dataset
        dataset = FiberOpticsDataset(
            self.data_paths,
            transform=train_transform,
            load_into_memory=False,  # Load on demand for large datasets
            max_samples_per_class=1000  # Limit for training efficiency
        )
        
        # Split into train and validation
        total_samples = len(dataset)
        train_size = int(total_samples * train_ratio)
        val_size = total_samples - train_size
        
        # Random split
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # Apply validation transform to val dataset if using augmentation
        if use_augmentation and val_transform is not None:
            # Create a new dataset for validation with val_transform
            val_dataset_obj = FiberOpticsDataset(
                self.data_paths,
                transform=val_transform,
                load_into_memory=False,
                max_samples_per_class=1000
            )
            val_dataset = torch.utils.data.Subset(val_dataset_obj, val_indices)
        
        # Create weighted sampler for training if requested
        train_sampler = None
        if use_weighted_sampling:
            # Calculate class weights
            train_labels = [dataset.labels[i] for i in train_indices]
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            
            # Create sample weights
            sample_weights = [class_weights[label] for label in train_labels]
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_indices),
                replacement=True
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=False
        )
        
        self.logger.info(f"Created data loaders: train={len(train_loader)} batches, "
                        f"val={len(val_loader)} batches")
        
        self.logger.log_function_exit("get_data_loaders")
        
        return train_loader, val_loader
    
    def get_region_specific_loader(self, region: str, 
                                  batch_size: Optional[int] = None) -> DataLoader:
        """Get data loader for specific region"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Get folders for region
        region_folders = self.config.REGION_CATEGORIES.get(region, [])
        region_paths = []
        
        for folder_name in region_folders:
            folder_path = self.config.TENSORIZED_DATA_PATH / folder_name
            if folder_path.exists():
                region_paths.append(folder_path)
        
        if not region_paths:
            self.logger.warning(f"No data found for region: {region}")
            return None
        
        # Create dataset
        dataset = FiberOpticsDataset(region_paths)
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.logger.info(f"Created {region} loader with {len(loader)} batches")
        
        return loader
    
    def get_streaming_loader(self, batch_size: int = 1) -> DataLoader:
        """
        Get streaming data loader for real-time processing
        "or multiple images for batch processing or realtime processing"
        """
        # Create dataset with minimal loading
        dataset = FiberOpticsDataset(
            self.data_paths,
            load_into_memory=False,
            max_samples_per_class=100  # Small subset for streaming
        )
        
        # Create loader optimized for streaming
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,  # Single worker for streaming
            pin_memory=False,  # No pinning for streaming
            drop_last=False
        )
        
        self.logger.info(f"Created streaming loader with batch_size={batch_size}")
        
        return loader

class ReferenceDataLoader:
    """
    Specialized loader for reference data
    "I have reference tensorized images of these regions"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("ReferenceDataLoader")
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("ReferenceDataLoader")
        
        # Reference storage
        self.references = {
            'core': {},
            'cladding': {},
            'ferrule': {},
            'full': {}
        }
    
    def load_references(self, limit_per_region: int = 100):
        """Load reference images into memory"""
        self.logger.log_process_start("Loading reference images")
        
        # First try to load from reference folder
        reference_path = Path(self.config.TENSORIZED_DATA_PATH).parent / "reference"
        if reference_path.exists():
            # Map folder names to regions
            folder_to_region = {
                'core': ['core-batch-1', 'core-batch-2', 'core-batch-3', 'core-batch-4', 'core-batch-5', 'core-batch-6', 'core-batch-7', 'core-batch-8', 'large-core-batch'],
                'cladding': ['50-cladding', '91-cladding', 'cladding-batch-1', 'cladding-batch-3', 'cladding-batch-4', 'cladding-batch-5', 'cladding-features-batch-1'],
                'ferrule': ['ferrule-batch-1', 'ferrule-batch-2', 'ferrule-batch-3', 'ferrule-batch-4'],
                'full': ['50', '91', 'sma', 'sma-clean']
            }
            
            for region, folder_names in folder_to_region.items():
                count = 0
                for folder_name in folder_names:
                    folder_path = reference_path / folder_name
                    if folder_path.exists():
                        # Load limited number of references
                        pt_files = list(folder_path.glob("*.pt"))[:limit_per_region - count]
                        
                        for pt_file in pt_files:
                            try:
                                tensor_data = self.tensor_processor.load_tensor_file(pt_file)
                                ref_id = f"{region}_{pt_file.stem}"
                                self.references[region][ref_id] = tensor_data['tensor']
                                count += 1
                            except Exception as e:
                                self.logger.warning(f"Failed to load reference {pt_file}: {e}")
                        
                        if count >= limit_per_region:
                            break
                
                self.logger.info(f"Loaded {count} references for {region} from reference folder")
        
        # Fallback to config-based loading
        for region, folders in self.config.REGION_CATEGORIES.items():
            if region == 'defects' or len(self.references.get(region, {})) >= limit_per_region:
                continue
            
            count = len(self.references.get(region, {}))
            for folder_name in folders:
                folder_path = self.config.TENSORIZED_DATA_PATH / folder_name
                
                if not folder_path.exists():
                    continue
                
                # Load limited number of references
                pt_files = list(folder_path.glob("*.pt"))[:limit_per_region - count]
                
                for pt_file in pt_files:
                    try:
                        tensor_data = self.tensor_processor.load_tensor_file(pt_file)
                        ref_id = f"{region}_{pt_file.stem}"
                        self.references[region][ref_id] = tensor_data['tensor']
                        count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load reference {pt_file}: {e}")
                
                if count >= limit_per_region:
                    break
            
            self.logger.info(f"Loaded {count} total references for {region}")
        
        self.logger.log_process_end("Loading reference images")
    
    def get_reference_batch(self, region: str, batch_size: int = 32) -> torch.Tensor:
        """Get a batch of reference tensors"""
        if region not in self.references or not self.references[region]:
            self.logger.warning(f"No references available for region: {region}")
            return None
        
        # Sample random references
        ref_ids = list(self.references[region].keys())
        selected_ids = random.sample(ref_ids, min(batch_size, len(ref_ids)))
        
        # Stack tensors
        tensors = []
        for ref_id in selected_ids:
            tensor = self.references[region][ref_id]
            if tensor.dim() == 3:
                tensors.append(tensor)
            elif tensor.dim() == 2:
                tensors.append(tensor.unsqueeze(0).repeat(3, 1, 1))
        
        return torch.stack(tensors)

# Test the data loader
if __name__ == "__main__":
    data_loader = FiberOpticsDataLoader()
    logger = get_logger("DataLoaderTest")
    
    logger.log_process_start("Data Loader Test")
    
    # Get train and validation loaders
    train_loader, val_loader = data_loader.get_data_loaders(
        train_ratio=0.8,
        batch_size=4,
        use_weighted_sampling=True
    )
    
    # Test loading a batch
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Batch {batch_idx}:")
        logger.info(f"  Image shape: {batch['image'].shape}")
        logger.info(f"  Labels: {batch['label']}")
        logger.info(f"  Has anomaly: {batch['has_anomaly']}")
        
        if batch_idx >= 2:  # Just test a few batches
            break
    
    # Test reference loader
    ref_loader = ReferenceDataLoader()
    ref_loader.load_references(limit_per_region=10)
    
    ref_batch = ref_loader.get_reference_batch('core', batch_size=4)
    if ref_batch is not None:
        logger.info(f"Reference batch shape: {ref_batch.shape}")
    
    logger.log_process_end("Data Loader Test")
    logger.log_script_transition("data_loader.py", "main.py")
    
    print(f"[{datetime.now()}] Data loader test completed")
    print(f"[{datetime.now()}] Next script: main.py")
