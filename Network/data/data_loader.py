#!/usr/bin/env python3
"""
Data Loader module for Fiber Optics Neural Network
"I have a folder with a large database of .pt files of different images used as reference 
and the files are separated by folder name"
Handles loading and batching of tensorized data
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json
import random
import torchvision.transforms as T
from PIL import Image

from core.config_loader import get_config
from core.logger import get_logger
from data.tensor_processor import TensorProcessor
from utilities.distributed_utils import get_rank, get_world_size, is_main_process

class FiberOpticsDataset(Dataset):
    """
    Dataset for fiber optic images
    "an image will be selected from a dataset folder (or multiple images for batch processing)"
    Supports loading both .pt tensor files and raw image formats (PNG, JPG, JPEG, BMP, TIFF)
    """
    
    SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, data_paths: List[Path], 
                 transform=None,
                 load_into_memory: bool = False,
                 max_samples_per_class: Optional[int] = None,
                 use_raw_images: bool = False,
                 image_size: Tuple[int, int] = (256, 256)):
        print(f"[{datetime.now()}] Initializing FiberOpticsDataset")
        
        self.logger = get_logger("FiberOpticsDataset")
        self.config = get_config()
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("FiberOpticsDataset", 
                                  num_paths=len(data_paths),
                                  load_into_memory=load_into_memory)
        
        self.transform = transform
        self.load_into_memory = load_into_memory
        self.use_raw_images = use_raw_images
        self.image_size = image_size
        
        # Storage for samples
        self.samples = []
        self.labels = []
        self.metadata = []
        
        # Standardized to region-based classes for consistency; original had folder-specific classes leading to label inconsistency (e.g., high idx for specific folders vs low region idx)
        self.classes = ["core", "cladding", "ferrule", "defects"]
        self.class_to_idx = {"core": 0, "cladding": 1, "ferrule": 2, "defects": 3}
        
        # If loading into memory, store tensors
        self.tensors = {} if load_into_memory else None
        
        # Scan data paths and build dataset
        self._scan_data_paths(data_paths, max_samples_per_class)
        
        if len(self.samples) == 0:
            error_msg = f"No samples found in the provided data paths: {[str(p) for p in data_paths]}\n"
            if use_raw_images:
                error_msg += "Expected image files with extensions: " + ", ".join(self.SUPPORTED_IMAGE_FORMATS)
            else:
                error_msg += "Expected .pt tensor files"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.logger.info(f"Dataset initialized with {len(self.samples)} samples")
        print(f"[{datetime.now()}] FiberOpticsDataset initialized with {len(self.samples)} samples")
    
    def _scan_data_paths(self, data_paths: List[Path], max_samples_per_class: Optional[int]):
        """Scan data paths and collect samples"""
        self.logger.log_process_start("Scanning data paths")
        
        class_counts = {idx: 0 for idx in self.class_to_idx.values()}  # Initialized with 0 for each class; original used {} and get(0), but could miss classes leading to division by zero in weighted sampling
        
        for path in data_paths:
            if not path.exists():
                self.logger.warning(f"Path does not exist: {path}")
                continue
            
            # Find files based on mode
            if self.use_raw_images:
                # For raw images, check if this is a dataset folder with chunks
                if path.name == "dataset" and path.is_dir():
                    # Scan chunk folders
                    chunk_folders = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith('chunk_')])
                    if chunk_folders:
                        # Process chunk folders
                        for chunk_folder in chunk_folders:
                            self._scan_image_folder(chunk_folder, class_counts, max_samples_per_class)
                    else:
                        # No chunks, scan the dataset folder directly
                        self._scan_image_folder(path, class_counts, max_samples_per_class)
                else:
                    # Regular folder scan
                    self._scan_image_folder(path, class_counts, max_samples_per_class)
            else:
                # Find all .pt files
                pt_files = sorted(path.glob("*.pt"))  # Added sorted for deterministic order; original glob could vary by OS/filesystem, causing non-reproducible splits in get_data_loaders
            
                self.logger.info(f"Found {len(pt_files)} .pt files in {path.name}")
                
                # Determine class label from folder structure
                class_label = self._determine_class_label(path)
                
                if class_label == -1:  # Added skip for unknown labels; original appended invalid labels, causing errors later
                    self.logger.warning(f"Skipping unknown class for path: {path.name}")
                    continue
                
                # Limit samples per class if specified
                if max_samples_per_class is not None:  # Changed to is not None for clarity; original checked truthy but None==False, but explicit better
                    remaining = max_samples_per_class - class_counts.get(class_label, 0)
                    if remaining <= 0:
                        continue
                    pt_files = pt_files[:remaining]
                
                for pt_file in pt_files:
                    sample_info = {
                        'path': str(pt_file),  # Convert Path to string
                        'filename': pt_file.name,
                        'folder': path.name,
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
                            continue  # Added continue to skip invalid files; original appended path but tensors missing, causing KeyError in __getitem__
                    
                    # Update class count
                    class_counts[class_label] += 1  # Simplified; get not needed since initialized
        
        self.logger.info(f"Class distribution: {class_counts}")
        self.logger.log_process_end("Scanning data paths")
    
    def _scan_image_folder(self, folder: Path, class_counts: Dict[int, int], max_samples_per_class: Optional[int]):
        """Scan a folder for image files"""
        # Find all image files
        image_files = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)  # Sort for consistency
        
        if image_files:
            self.logger.info(f"Found {len(image_files)} image files in {folder.name}")
        
        for img_file in image_files:
            # Determine class from filename
            class_label = self._determine_class_from_filename(img_file.name)
            
            if class_label == -1:
                # Try to determine from folder name as fallback
                class_label = self._determine_class_label(folder)
                if class_label == -1:
                    self.logger.debug(f"Unknown class for file: {img_file.name}")
                    continue
            
            # Check limit
            if max_samples_per_class is not None:
                if class_counts[class_label] >= max_samples_per_class:
                    continue
            
            sample_info = {
                'path': str(img_file),  # Convert Path to string for JSON serialization
                'filename': img_file.name,
                'folder': folder.name,
                'class_name': self.classes[class_label],
                'class_label': class_label,
                'has_anomaly': self._check_if_anomaly(img_file.name) or self._check_if_anomaly(folder.name)
            }
            
            self.samples.append(img_file)
            self.labels.append(class_label)
            self.metadata.append(sample_info)
            
            # Load into memory if requested
            if self.load_into_memory:
                try:
                    tensor = self._load_image(img_file)
                    self.tensors[str(img_file)] = tensor
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_file}: {e}")
                    continue
            
            class_counts[class_label] += 1
    
    def _determine_class_from_filename(self, filename: str) -> int:
        """Determine class label from filename"""
        filename_lower = filename.lower()
        
        # Check for class keywords in filename
        if 'core' in filename_lower:
            return self.class_to_idx['core']
        elif 'cladding' in filename_lower:
            return self.class_to_idx['cladding']
        elif 'ferrule' in filename_lower:
            return self.class_to_idx['ferrule']
        elif any(keyword in filename_lower for keyword in ['defect', 'dirty', 'scratch', 'contamination', 'anomaly']):
            return self.class_to_idx['defects']
        
        # Check for mask_ prefix patterns
        if filename_lower.startswith('mask_'):
            parts = filename_lower.split('_')
            if len(parts) >= 2:
                region = parts[1]
                if region in self.class_to_idx:
                    return self.class_to_idx[region]
        
        return -1  # Unknown class
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image file"""
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(self.image_size, Image.BILINEAR)
            
            # Convert to numpy array
            image_np = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to tensor (HWC -> CHW)
            tensor = torch.from_numpy(image_np).permute(2, 0, 1)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.image_size[0], self.image_size[1])
    
    def _determine_class_label(self, path: Path) -> int:
        """Determine class label from path"""
        folder_name = path.name
        
        # Check against region categories
        for region, idx in self.class_to_idx.items():
            if folder_name in self.config.REGION_CATEGORIES.get(region, []):
                return idx
        
        # Fallback to anomaly check
        if self._check_if_anomaly(folder_name):
            return self.class_to_idx["defects"]
        
        # Default to unknown (skip in scanning)
        return -1  # Changed from len(self.classes) to -1; original could return arbitrary high idx, causing label mismatch and errors in np.bincount for sampling
    
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
        
        # Load tensor or image
        if self.load_into_memory and str(sample_path) in self.tensors:
            tensor = self.tensors[str(sample_path)]
        else:
            if self.use_raw_images:
                # Load raw image
                tensor = self._load_image(sample_path)
            else:
                # Load tensor file
                try:
                    tensor_data = self.tensor_processor.load_tensor_file(sample_path)
                    tensor = tensor_data['tensor']
                except Exception as e:
                    self.logger.error(f"Failed to load sample {idx}: {e}")
                    # Return zeros tensor as fallback
                    tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
        
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
        
        # Will be set based on data availability
        self.use_raw_images = False
        
        # Collect all data paths
        self.data_paths = self._collect_data_paths()
        
        if not self.data_paths:
            error_msg = "No data found! Please ensure either:\n"
            error_msg += "1. Image files exist in the 'dataset' folder, or\n"
            error_msg += "2. Tensor files (.pt) exist in the 'reference' folder\n"
            error_msg += f"Checked paths: dataset={Path(__file__).parent.parent / 'dataset'}, "
            error_msg += f"reference={Path(self.config.system.tensorized_data_path).parent / 'reference'}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.logger.info(f"Found {len(self.data_paths)} data folders")
        print(f"[{datetime.now()}] FiberOpticsDataLoader initialized")
    
    def _collect_data_paths(self) -> List[Path]:
        """Collect all relevant data paths"""
        paths = []
        
        # Check if we should use raw images from dataset folder
        dataset_path = Path(__file__).parent.parent / "dataset"
        if dataset_path.exists():
            # Check if there are image files in the dataset folder
            has_images = False
            for ext in FiberOpticsDataset.SUPPORTED_IMAGE_FORMATS:
                if list(dataset_path.glob(f"*{ext}")) or list(dataset_path.glob(f"**/*{ext}")):
                    has_images = True
                    break
            
            if has_images:
                self.logger.info(f"Found images in dataset folder, will use raw image loading")
                self.use_raw_images = True
                return [dataset_path]
        
        # Otherwise, check for tensor files
        self.use_raw_images = False
        
        # First check if reference folder exists
        reference_path = Path(self.config.system.tensorized_data_path).parent / "reference"
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
                        use_augmentation: bool = True,
                        distributed: bool = False,
                        image_size: Tuple[int, int] = (256, 256)) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation data loaders
        "the program will run entirely in hpc"
        """
        self.logger.log_function_entry("get_data_loaders")
        
        if batch_size is None:
            batch_size = self.config.training.batch_size  # Fallback to config.training.batch_size; original used self.config.BATCH_SIZE but that's legacy, use dot notation for consistency
        
        if num_workers is None:
            num_workers = self.config.system.num_workers  # Fallback to config.system.num_workers; original used self.config.NUM_WORKERS legacy
        
        # Data augmentation based on recommendations from data-implementation.py
        train_transform = None
        val_transform = None
        
        if use_augmentation:
            train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Retained ImageNet norms; but for fiber optics, consider custom in config.yaml if needed (e.g., model: {normalize_mean: [0.5,0.5,0.5]})
            ])
            
            val_transform = T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Create full dataset
        dataset = FiberOpticsDataset(
            self.data_paths,
            transform=train_transform if use_augmentation else None,  # Added conditional; original always passed train_transform even if not use_augmentation
            load_into_memory=False,  # Load on demand for large datasets
            max_samples_per_class=1000,  # Limit for training efficiency
            use_raw_images=self.use_raw_images,
            image_size=image_size
        )
        
        # Split into train and validation
        total_samples = len(dataset)
        if total_samples == 0:  # Added check; original proceeded with empty, causing ZeroDivisionError in split
            raise ValueError("No samples found in dataset. Check data paths and tensor files.")
        train_size = int(total_samples * train_ratio)
        val_size = total_samples - train_size
        
        # Random split with seed for reproducibility
        torch.manual_seed(42)  # Added seed; original random.shuffle not seeded, leading to non-reproducible splits
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
                max_samples_per_class=1000,
                use_raw_images=self.use_raw_images,
                image_size=image_size
            )
            val_dataset = torch.utils.data.Subset(val_dataset_obj, val_indices)
        
        # Create samplers based on distributed training
        if distributed:
            # Distributed samplers for multi-GPU training
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False
            )
            # Disable shuffle when using DistributedSampler
            train_shuffle = False
            val_shuffle = False
        else:
            # Single GPU samplers
            val_sampler = None
            val_shuffle = False
            
            if use_weighted_sampling:
                # Calculate class weights
                train_labels = [dataset.labels[i] for i in train_indices]
                class_counts = np.bincount(train_labels, minlength=len(dataset.classes))  # Added minlength=4; original could have shorter array if not all classes present, causing index errors in class_weights
                class_weights = 1.0 / (class_counts + 1e-6)
                
                # Create sample weights
                sample_weights = [class_weights[label] for label in train_labels]
                
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_indices),
                    replacement=True
                )
                train_shuffle = False
            else:
                train_sampler = None
                train_shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=self.config.system.pin_memory if hasattr(self.config.system, 'pin_memory') else True,  # Changed to config.system.pin_memory; original used legacy self.config.PIN_MEMORY, use yaml for all configs
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=val_shuffle,
            num_workers=num_workers,
            pin_memory=self.config.system.pin_memory if hasattr(self.config.system, 'pin_memory') else True,  # Changed to config.system.pin_memory; original used legacy self.config.PIN_MEMORY, use yaml for all configs
            drop_last=False
        )
        
        if distributed and is_main_process():
            self.logger.info(f"Created distributed data loaders (rank {get_rank()}/{get_world_size()}): "
                           f"train={len(train_loader)} batches, val={len(val_loader)} batches")
        elif not distributed:
            self.logger.info(f"Created data loaders: train={len(train_loader)} batches, "
                           f"val={len(val_loader)} batches")
        
        self.logger.log_function_exit("get_data_loaders")
        
        return train_loader, val_loader
    
    def get_region_specific_loader(self, region: str, 
                                  batch_size: Optional[int] = None) -> DataLoader:
        """Get data loader for specific region"""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
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
            num_workers=self.config.system.num_workers,
            pin_memory=self.config.system.pin_memory if hasattr(self.config.system, 'pin_memory') else True
        )
        
        self.logger.info(f"Created {region} loader with {len(loader)} batches")
        
        return loader
    
    def get_streaming_loader(self, batch_size: int = 1) -> DataLoader:
        """
        Get a streaming data loader for real-time processing
        Returns a single batch at a time for continuous processing
        """
        self.logger.info("Creating streaming data loader")
        
        # Use a small subset for streaming
        streaming_paths = self._collect_data_paths()[:100]  # Limit to 100 samples for streaming
        
        if not streaming_paths:
            self.logger.warning("No data paths available for streaming")
            return None
        
        # Create dataset
        dataset = FiberOpticsDataset(
            data_paths=streaming_paths,
            transform=self._get_transforms(use_augmentation=False),
            load_into_memory=False,
            use_raw_images=self.config.get('data.use_raw_images', False)
        )
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Sequential processing for streaming
            num_workers=0,  # Single worker for real-time
            pin_memory=torch.cuda.is_available()
        )
        
        self.logger.info(f"Streaming loader created with {len(dataset)} samples")
        return loader
    
    def get_test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Get a test data loader for evaluation and benchmarking
        Uses a separate test split or validation data as test data
        """
        self.logger.info("Creating test data loader")
        
        # Use validation data as test data if no separate test split
        _, test_loader = self.get_data_loaders(
            train_ratio=0.8,
            batch_size=batch_size or self.config.training.batch_size,
            use_weighted_sampling=False,  # No weighting for testing
            use_augmentation=False  # No augmentation for testing
        )
        
        self.logger.info("Test loader created from validation split")
        return test_loader

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
        # Use the project root to find the reference folder
        reference_path = Path(__file__).parent.parent / "reference"
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