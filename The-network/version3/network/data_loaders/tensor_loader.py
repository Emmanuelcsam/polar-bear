import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..utils.logger import get_logger
from ..config.config import get_config


class TensorDataLoader:
    """
    Loads and manages tensorized fiber optic images (.pt files) and associated metadata.
    "I have a folder with a large database of .pt files of different images used as reference"
    "I also have several .json files, .pth files, .csv files .pkl files and .txt files with loads of data"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Validate paths
        self.config.validate_paths()
        
        # Cache for loaded tensors
        self.tensor_cache = {}
        self.metadata_cache = {}
        
        # Reference tensors organized by region
        self.reference_tensors = {
            'core': {},
            'cladding': {},
            'ferrule': {},
            'full': {}
        }
        
        # Statistical data from existing processing
        self.statistics = {}
        
        self.logger.info("Initialized TensorDataLoader")
        
    def load_all_references(self, preload=True):
        """
        Load all reference tensors and organize by region type.
        "the files are separated by folder name"
        """
        start_time = time.time()
        self.logger.info("Starting to load all reference tensors...")
        
        total_loaded = 0
        
        # Load tensors for each region category
        for region, folders in self.config.REGION_CATEGORIES.items():
            if region == 'defects':
                continue  # Skip defect images for reference
                
            region_key = 'full' if region == 'full_images' else region
            
            for folder in folders:
                folder_path = self.config.TENSORIZED_DATA_PATH / folder
                if not folder_path.exists():
                    self.logger.warning(f"Folder not found: {folder_path}")
                    continue
                
                # Load all .pt files in the folder
                pt_files = list(folder_path.glob("*.pt"))
                self.logger.info(f"Loading {len(pt_files)} tensors from {folder}...")
                
                if preload:
                    # Load tensors in parallel for speed
                    loaded = self._parallel_load_tensors(pt_files, region_key, folder)
                    total_loaded += loaded
                else:
                    # Just store paths for lazy loading
                    for pt_file in pt_files:
                        key = f"{folder}/{pt_file.stem}"
                        self.reference_tensors[region_key][key] = pt_file
                    total_loaded += len(pt_files)
        
        # Load associated metadata files
        self._load_metadata()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Loaded {total_loaded} reference tensors in {elapsed:.2f}s")
        self.logger.log_performance_metric("reference_loading_time", elapsed * 1000)
        
        return total_loaded
    
    def _parallel_load_tensors(self, pt_files: List[Path], region: str, folder: str) -> int:
        """Load tensors in parallel using ThreadPoolExecutor"""
        loaded_count = 0
        
        with ThreadPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
            future_to_file = {
                executor.submit(self._load_single_tensor, pt_file): pt_file 
                for pt_file in pt_files
            }
            
            for future in as_completed(future_to_file):
                pt_file = future_to_file[future]
                try:
                    tensor = future.result()
                    if tensor is not None:
                        key = f"{folder}/{pt_file.stem}"
                        self.reference_tensors[region][key] = tensor
                        loaded_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading {pt_file}: {e}")
        
        return loaded_count
    
    def _load_single_tensor(self, pt_file: Path) -> Optional[torch.Tensor]:
        """Load a single tensor file with error handling"""
        try:
            tensor = torch.load(pt_file, map_location='cpu')
            self.logger.debug(f"Loaded tensor from {pt_file} with shape {tensor.shape}")
            return tensor
        except Exception as e:
            self.logger.error(f"Failed to load tensor {pt_file}: {e}")
            return None
    
    def _load_metadata(self):
        """
        Load all metadata files (.json, .csv, .pkl, .txt)
        "I also have several .json files, .pth files, .csv files .pkl files and .txt files"
        """
        self.logger.info("Loading metadata files...")
        
        # Search for metadata files in the reference directory
        metadata_extensions = ['.json', '.csv', '.pkl', '.txt', '.pth']
        
        for ext in metadata_extensions:
            files = list(self.config.REFERENCE_ROOT.rglob(f"*{ext}"))
            self.logger.info(f"Found {len(files)} {ext} files")
            
            for file in files:
                try:
                    if ext == '.json':
                        with open(file, 'r') as f:
                            data = json.load(f)
                            self.metadata_cache[file.stem] = data
                            
                    elif ext == '.csv':
                        data = pd.read_csv(file)
                        self.metadata_cache[file.stem] = data
                        
                    elif ext == '.pkl':
                        with open(file, 'rb') as f:
                            data = pickle.load(f)
                            self.metadata_cache[file.stem] = data
                            
                    elif ext == '.txt':
                        with open(file, 'r') as f:
                            data = f.readlines()
                            self.metadata_cache[file.stem] = data
                            
                    elif ext == '.pth':
                        # Could be model weights or other torch data
                        data = torch.load(file, map_location='cpu')
                        self.metadata_cache[file.stem] = data
                    
                    self.logger.debug(f"Loaded metadata from {file}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load metadata {file}: {e}")
    
    def get_tensor_batch(self, batch_size: int, region: Optional[str] = None) -> Tuple[torch.Tensor, List[str]]:
        """
        Get a batch of tensors for training.
        "an image will be selected from a dataset folder(or multiple images for batch processing"
        """
        if region and region in self.reference_tensors:
            # Get tensors from specific region
            tensor_dict = self.reference_tensors[region]
        else:
            # Get tensors from all regions
            tensor_dict = {}
            for region_tensors in self.reference_tensors.values():
                tensor_dict.update(region_tensors)
        
        # Sample batch_size tensors
        keys = list(tensor_dict.keys())
        if len(keys) < batch_size:
            self.logger.warning(f"Requested batch size {batch_size} but only {len(keys)} tensors available")
            batch_size = len(keys)
        
        import random
        selected_keys = random.sample(keys, batch_size)
        
        # Stack tensors into batch
        tensors = []
        for key in selected_keys:
            tensor = tensor_dict[key]
            if isinstance(tensor, Path):
                # Lazy load if needed
                tensor = self._load_single_tensor(tensor)
            tensors.append(tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(tensors)
        
        self.logger.log_tensor_operation("batch_created", batch_tensor.shape, 
                                        f"region={region}, keys={len(selected_keys)}")
        
        return batch_tensor, selected_keys
    
    def get_reference_by_similarity(self, input_tensor: torch.Tensor, 
                                   region: str) -> Tuple[torch.Tensor, str, float]:
        """
        Find the most similar reference tensor for a given input.
        "try to see which of the reference images in the reference folder of the database 
        that the regions most specifically represent"
        """
        if region not in self.reference_tensors:
            raise ValueError(f"Unknown region: {region}")
        
        best_similarity = -1
        best_tensor = None
        best_key = None
        
        # Compare with all reference tensors in the region
        for key, ref_tensor in self.reference_tensors[region].items():
            if isinstance(ref_tensor, Path):
                ref_tensor = self._load_single_tensor(ref_tensor)
            
            # Calculate similarity (basic cosine similarity for now)
            similarity = self._calculate_similarity(input_tensor, ref_tensor)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tensor = ref_tensor
                best_key = key
        
        self.logger.info(f"Best match for {region}: {best_key} with similarity {best_similarity:.4f}")
        
        return best_tensor, best_key, best_similarity
    
    def _calculate_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate similarity between two tensors"""
        # Ensure same shape
        if tensor1.shape != tensor2.shape:
            # Resize if needed
            tensor2 = torch.nn.functional.interpolate(
                tensor2.unsqueeze(0), 
                size=tensor1.shape[-2:], 
                mode='bilinear'
            ).squeeze(0)
        
        # Flatten and normalize
        t1_flat = tensor1.flatten()
        t2_flat = tensor2.flatten()
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0), 
            t2_flat.unsqueeze(0)
        ).item()
        
        return similarity
    
    def get_correlational_data(self, image_key: str) -> Dict:
        """
        Get all correlational data for a specific image.
        "it will go through the neural network for classification, in the neural network 
        for classification it will be split up into segments, each segment will be analyzed 
        by multiple comparisons, statistics, and correlational data from the multiple sources"
        """
        correlations = {}
        
        # Check all metadata sources
        for metadata_key, data in self.metadata_cache.items():
            if image_key in str(data):
                correlations[metadata_key] = data
        
        # Add statistics if available
        if image_key in self.statistics:
            correlations['statistics'] = self.statistics[image_key]
        
        self.logger.debug(f"Found {len(correlations)} correlation sources for {image_key}")
        
        return correlations
    
    def calculate_gradient_intensity(self, tensor: torch.Tensor) -> float:
        """
        Calculate average intensity gradient for weight initialization.
        "the weights of the neural network will be dependent on the average intensity gradient"
        """
        # Calculate gradients using Sobel filters
        if tensor.dim() == 3:  # C, H, W
            gray = tensor.mean(dim=0)  # Convert to grayscale
        else:
            gray = tensor
        
        # Simple gradient calculation
        dx = gray[1:, :] - gray[:-1, :]
        dy = gray[:, 1:] - gray[:, :-1]
        
        # Average gradient magnitude
        grad_magnitude = torch.sqrt(dx[:-1, :-1]**2 + dy[:-1, :-1]**2)
        avg_gradient = grad_magnitude.mean().item()
        
        self.logger.debug(f"Calculated average gradient intensity: {avg_gradient:.4f}")
        
        return avg_gradient
    
    def calculate_pixel_positions(self, tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate average pixel position for weight initialization.
        "another weight will be dependent on the average pixel position"
        """
        h, w = tensor.shape[-2:]
        
        # Create coordinate grids
        y_coords = torch.arange(h).float().unsqueeze(1).expand(h, w)
        x_coords = torch.arange(w).float().unsqueeze(0).expand(h, w)
        
        # Weight by pixel intensities
        if tensor.dim() == 3:
            intensity = tensor.mean(dim=0)
        else:
            intensity = tensor
        
        # Normalize intensity
        intensity = intensity / (intensity.sum() + 1e-8)
        
        # Calculate weighted average positions
        avg_y = (y_coords * intensity).sum().item()
        avg_x = (x_coords * intensity).sum().item()
        
        self.logger.debug(f"Calculated average pixel position: ({avg_x:.2f}, {avg_y:.2f})")
        
        return avg_x, avg_y